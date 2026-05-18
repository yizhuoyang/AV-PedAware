import numpy as np
import torch.nn as nn
import torch
from itertools import permutations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
import torch
import torchaudio.transforms as T
from typing import Dict, List, Optional, Tuple, Any, Union
# from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import scipy.io as sio

@dataclass
class CalibFromMat:
    K: np.ndarray
    R: np.ndarray
    T: np.ndarray
    kc: np.ndarray

    @staticmethod
    def from_mat(mat_path: Union[str, Path]) -> "CalibFromMat":
        mat_path = Path(mat_path)
        d = sio.loadmat(str(mat_path))
        if "camData" not in d:
            raise KeyError(f"'camData' not found in {mat_path}")

        camData = d["camData"][0, 0]
        K = camData["K"].astype(np.float32)
        R = camData["R"].astype(np.float32)
        T = camData["T"].astype(np.float32)
        kc = camData["kc"].astype(np.float32).reshape(-1)

        if T.shape != (3, 1):
            if T.shape == (1, 3):
                T = T.T
            elif T.shape == (3,):
                T = T.reshape(3, 1)
            else:
                raise ValueError(f"Unexpected T shape: {T.shape}")

        kc = kc[:5]
        return CalibFromMat(K=K, R=R, T=T, kc=kc)


def parse_gt_xyz(gt_arr: np.ndarray) -> np.ndarray:
    """
    Support:
      - (3,) single target
      - (3K,) concatenated targets
      - (K,3) multi-target
    Return (K,3).
    """
    gt = np.asarray(gt_arr)
    if gt.ndim == 1:
        if gt.shape[0] == 3:
            return gt.reshape(1, 3).astype(np.float32)
        if gt.shape[0] % 3 == 0:
            return gt.reshape(-1, 3).astype(np.float32)
        raise ValueError(f"Unsupported 1D gt shape: {gt.shape}")
    if gt.ndim == 2 and gt.shape[1] == 3:
        return gt.astype(np.float32)
    raise ValueError(f"Unsupported gt shape: {gt.shape}")


def doa_xz_deg_from_xyz_cav3d(
    xyz: np.ndarray,
    calib: CalibFromMat,
    assume_gt_in_camera_frame: bool = False,
    mirror_x: bool = False,   # 如果你发现左右反，就设 True
) -> np.ndarray:
    """
    xyz: (K,3) in GT coord (or camera coord if assume_gt_in_camera_frame=True)

    1) Pc = R*P + T
    2) yaw = atan2(X, Z)  (Z 是深度轴，和你投影 x=X/Z 对齐)
    3) return degrees (K,)
    """
    P = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)

    if assume_gt_in_camera_frame:
        Pc = P
    else:
        Pc = (calib.R @ P.T + calib.T).T  # (K,3)

    X = Pc[:, 0]
    Z = Pc[:, 2]

    if mirror_x:
        X = -X

    yaw = np.arctan2(X, Z)  # radians
    yaw_deg = np.degrees(yaw).astype(np.float32)
    return yaw_deg




def doa_xy_deg_from_xyz(xyz):
    """
    xyz: (3,) or (P,3)
    returns: (P,) degree in [0,360)
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim == 1:
        xyz = xyz[None, :]
    x = xyz[:, 0]
    y = xyz[:, 1]
    deg = np.degrees(np.arctan2(y, x))
    deg = (deg + 360.0) % 360.0
    return deg.astype(np.float32)

def load_gt_dict(gt_path):
    gt = np.load(str(gt_path), allow_pickle=True).item()
    if not isinstance(gt, dict):
        raise RuntimeError(f"GT is not a dict: {gt_path}")
    return gt

def stem(p):
    return p.stem

def downsample_audio(audio_tensor, target_length=1600):
    """
    Downsamples a multi-channel audio tensor to a fixed length (1600 samples).

    :param audio_tensor: Input tensor of shape (C, N), where N > 1600.
    :param target_length: The target number of samples (default = 1600).
    :return: Downsampled audio tensor of shape (C, 1600).
    """
    C, N = audio_tensor.shape  # Number of channels and original length

    # Define resampling transformation
    resampler = T.Resample(orig_freq=N, new_freq=target_length)

    # Apply resampling to all channels at once
    downsampled_audio = resampler(audio_tensor)  # Output shape: (C, 1600)

    return downsampled_audio
class ModeVector_torch_copy:
    """
    A class for look-up tables of mode vectors using PyTorch.
    This look-up table is an outer product of three vectors running along candidate locations,
    time, and frequency. If the table is too large to store in memory, it computes values on the fly.
    """
    def __init__(self, L, fs, nfft, c, grid, mode="far", precompute=True, device='cpu'):
        """
        Parameters
        ----------
        L: torch.Tensor
            Locations of the sensors (shape: [3, num_mics])
        fs: int
            Sampling frequency of the input signal
        nfft: int
            FFT length (must be even)
        c: float
            Speed of sound
        grid: object
            Grid object with attributes `x`, `y`, `z` (torch.Tensors)
        mode: str, optional
            'far' (default) or 'near', specifying the mode vector computation method
        precompute: bool, optional
            Whether to precompute the whole table (default: False)
        device: str, optional
            Device to store tensors (default: 'cpu')
        """
        if nfft % 2 == 1:
            raise ValueError("FFT length must be even.")

        self.device = torch.device(device)
        self.precompute = precompute

        # Propagation vectors
        p_x = torch.tensor(grid.x).view(1, 1, -1).to(self.device)
        p_y = torch.tensor(grid.y).view(1, 1, -1).to(self.device)
        p_z = torch.tensor(grid.z).view(1, 1, -1).to(self.device)

        # Microphone locations
        r_x = torch.tensor(L[0]).view(1, -1, 1).to(self.device)
        r_y = torch.tensor(L[1]).view(1, -1, 1).to(self.device)
        r_z = torch.tensor(L[2]).view(1, -1, 1).to(self.device) if L.shape[0] == 3 else torch.zeros((1, L.shape[1], 1), device=self.device)

        # Compute distance or projection
        if mode == "near":
            dist = torch.sqrt((p_x - r_x) ** 2 + (p_y - r_y) ** 2 + (p_z - r_z) ** 2)
        elif mode == "far":
            dist = (p_x * r_x) + (p_y * r_y) + (p_z * r_z)
        else:
            raise ValueError("Mode must be 'near' or 'far'")

        self.tau = dist / c  # Time delay
        self.omega = 2 * torch.pi * fs * torch.arange(nfft // 2 + 1, device=self.device) / nfft

        if precompute:
            self.mode_vec = torch.exp(1j * self.omega[:, None, None] * self.tau)
        else:
            self.mode_vec = None

    def __getitem__(self, ref):
        """
        Retrieve mode vector values. Computes on the fly if not precomputed.
        """
        if self.precompute:
            return self.mode_vec[ref]

        w = self.omega[ref[0]].unsqueeze(-1) if isinstance(ref[0], slice) else self.omega[ref[0]]
        tau_selected = self.tau if len(ref) == 1 else self.tau[:, ref[1], :] if len(ref) == 2 else self.tau[:, ref[1], ref[2]]
        return torch.exp(1j * w * tau_selected)

class ModeVector_torch:
    """
    A class for look-up tables of mode vectors using PyTorch.
    This look-up table is an outer product of three vectors running along candidate locations,
    time, and frequency. If the table is too large to store in memory, it computes values on the fly.
    """
    def __init__(self, L, fs, nfft, c, grid, mode="far", precompute=True):
        """
        Parameters
        ----------
        L: torch.Tensor
            Locations of the sensors (shape: [3, num_mics])
        fs: int
            Sampling frequency of the input signal
        nfft: int
            FFT length (must be even)
        c: float
            Speed of sound
        grid: object
            Grid object with attributes `x`, `y`, `z` (torch.Tensors)
        mode: str, optional
            'far' (default) or 'near', specifying the mode vector computation method
        precompute: bool, optional
            Whether to precompute the whole table (default: False)
        device: str, optional
            Device to store tensors (default: 'cpu')
        """
        if nfft % 2 == 1:
            raise ValueError("FFT length must be even.")

        self.precompute = precompute

        # Propagation vectors
        p_x = torch.tensor(grid.x).view(1, 1, -1)
        p_y = torch.tensor(grid.y).view(1, 1, -1)
        p_z = torch.tensor(grid.z).view(1, 1, -1)

        # Microphone locations
        r_x = torch.tensor(L[0]).view(1, -1, 1)
        r_y = torch.tensor(L[1]).view(1, -1, 1)
        r_z = torch.tensor(L[2]).view(1, -1, 1) if L.shape[0] == 3 else torch.zeros((1, L.shape[1], 1))

        # Compute distance or projection
        if mode == "near":
            dist = torch.sqrt((p_x - r_x) ** 2 + (p_y - r_y) ** 2 + (p_z - r_z) ** 2)
        elif mode == "far":
            dist = (p_x * r_x) + (p_y * r_y) + (p_z * r_z)
        else:
            raise ValueError("Mode must be 'near' or 'far'")

        self.tau = dist / c  # Time delay
        self.omega = 2 * torch.pi * fs * torch.arange(nfft // 2 + 1, ) / nfft

        if precompute:
            self.mode_vec = torch.exp(1j * self.omega[:, None, None] * self.tau)
        else:
            self.mode_vec = None

    def __getitem__(self, ref):
        """
        Retrieve mode vector values. Computes on the fly if not precomputed.
        """
        if self.precompute:
            return self.mode_vec[ref]

        w = self.omega[ref[0]].unsqueeze(-1) if isinstance(ref[0], slice) else self.omega[ref[0]]
        tau_selected = self.tau if len(ref) == 1 else self.tau[:, ref[1], :] if len(ref) == 2 else self.tau[:, ref[1], ref[2]]
        return torch.exp(1j * w * tau_selected)

class SteeringVector:
    def __init__(self, mic_positions, angles):
        """
        Compute the steering vector for an arbitrary microphone array shape, without considering frequency.

        :param mic_positions: Microphone coordinates, shape [M, 3] (M microphones with (x, y, z) positions).
        :param angles: List of incident angles, shape [N], in radians.
        """
        self.mic_positions = torch.tensor(mic_positions, dtype=torch.float32)  # [M, 3]
        self.angles = torch.tensor(angles, dtype=torch.float32)  # [N]

    def steering_vec(self):
        """
        Compute the steering vector for the microphone array, supporting any array shape and independent of frequency.

        Returns:
        --------
            torch.Tensor: Steering vector, shape [N, M] (N angles, M microphones).
        """
        # Compute the wave vector assuming the sound wave propagates in the XY plane.
        wave_vectors = torch.stack([
            torch.cos(self.angles),
            torch.sin(self.angles),
            torch.zeros_like(self.angles)  # Assume no variation in the Z direction.
        ], dim=-1)  # [N, 3]

        # Compute phase shifts (without frequency dependence)
        phase_shifts = torch.exp(-1j * torch.matmul(self.mic_positions.to('cuda'), wave_vectors.T))  # [M, N]

        return phase_shifts.T  # [N, M]


def filter_folders(folder_list, n):
    if n == 0:
        return folder_list
    return [folder for folder in folder_list if f"_{n}" in folder]

def permute_prediction(prediction: torch.Tensor):
    """
    Generates all the available permutations of the given prediction tensor.

    Args:
        prediction (torch.Tensor): The input tensor for which permutations are generated.

    Returns:
        torch.Tensor: A tensor containing all the permutations of the input tensor.

    Examples:
        >>> prediction = torch.tensor([1, 2, 3])
        >>>> permute_prediction(prediction)
            torch.tensor([[1, 2, 3],
                          [1, 3, 2],
                          [2, 1, 3],
                          [2, 3, 1],
                          [3, 1, 2],
                          [3, 2, 1]])

    """
    torch_perm_list = []
    for p in list(permutations(range(prediction.shape[0]),prediction.shape[0])):
        torch_perm_list.append(prediction.index_select( 0, torch.tensor(list(p), dtype = torch.int64).to(device)))
    predictions = torch.stack(torch_perm_list, dim = 0)
    return predictions

class RMSPELoss(nn.Module):
    """Root Mean Square Periodic Error (RMSPE) loss function.
    This loss function calculates the RMSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the RMSPE loss between the predictions and target values.

    Example:
        criterion = RMSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """
    def __init__(self):
        super(RMSPELoss, self).__init__()
    def forward(self, doa_predictions: torch.Tensor, doa: torch.Tensor):
        """Compute the RMSPE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa.
        The predicted values and target values are expected to be in radians.
        The method iterates over the batch dimension and calculates the RMSPE loss for each sample in the batch.
        It utilizes the permute_prediction function to generate all possible permutations of the predicted values
        to consider all possible alignments. For each permutation, it calculates the error between the prediction
        and target values, applies modulo pi to ensure the error is within the range [-pi/2, pi/2], and then calculates the RMSPE.
        The minimum RMSPE value among all permutations is selected for each sample.
        Finally, the method sums up the RMSPE values for all samples in the batch and returns the result as the computed loss.

        Args:
            doa_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            doa (torch.Tensor): Target values tensor of shape (batch_size, num_targets).

        Returns:
            torch.Tensor: The computed RMSPE loss.

        Raises:
            None
        """
        rmspe = []
        for iter in range(doa_predictions.shape[0]):
            rmspe_list = []
            batch_predictions = doa_predictions[iter].to(device)
            targets = doa[iter].to(device)
            prediction_perm = permute_prediction(batch_predictions).to(device)
            for prediction in prediction_perm:
                # Calculate error with modulo pi
                # error = (((prediction - targets) + (np.pi / 2)) % np.pi) - np.pi / 2
                # error = ((prediction - targets + np.pi) % 2*np.pi) - np.
                error = ((prediction - targets + 180) % 360) - 180
                # Calculate RMSE over all permutations
                rmspe_val = (1 / np.sqrt(len(targets))) * torch.linalg.norm(error)
                # rmspe_val = torch.abs((1 / len(targets)) * torch.abs(error))
                rmspe_list.append(rmspe_val)
            rmspe_tensor = torch.stack(rmspe_list, dim = 0)
            # Choose minimal error from all permutations
            rmspe_min = torch.min(rmspe_tensor)
            rmspe.append(rmspe_min)
        result = torch.sum(torch.stack(rmspe, dim = 0))
        return result

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for spectrum prediction.
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, preds, targets):
        preds = preds.clamp(min=1e-6, max=1.0)  # 避免 log(0)
        bce_loss = - (targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))
        focal_weight = self.alpha * (1 - preds) ** self.gamma * targets + (1 - self.alpha) * preds ** self.gamma * (1 - targets)
        loss = focal_weight * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def normalize_magnitude(magnitude, method="min-max"):
    if method == "min-max":
        min_val = magnitude.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # Min over time and frequency
        max_val = magnitude.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # Max over time and frequency
        magnitude = (magnitude - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
    elif method == "standard":
        mean = magnitude.mean()
        std = magnitude.std()
        magnitude = (magnitude - mean) / (std + 1e-8)
    return magnitude

def normalize_phase(phase, method="scale"):
    if method == "scale":
        phase = phase / torch.pi  # Scale phase to [-1, 1]
    elif method == "sincos":
        phase_sin = torch.sin(phase)
        phase_cos = torch.cos(phase)
        return phase_sin, phase_cos  # Returns two tensors
    return phase


if __name__ == "__main__":
    criterion = RMSPELoss()
    predictions = torch.tensor([100.0,20]).unsqueeze(0)
    targets = torch.tensor([350.0,140]).unsqueeze(0)
    loss = criterion(predictions, targets)
    print(loss)
