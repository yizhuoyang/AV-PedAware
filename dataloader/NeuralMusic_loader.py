from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as Trans
from scipy.io import wavfile
from torch.utils.data import Dataset

from dataloader.utils import ModeVector_torch, normalize_magnitude, normalize_phase
import random


def array_aug(
    mic_offsets,
    locs,
    transform,
    grid,
    sample_rate=16000,
    n_fft=512,
    interval=None,
    mic_center=np.array([[0, 0, 0]]),
):
    if transform:
        if interval == None:
            random_integer = random.randint(0, 360)
        else:
            random_integer = random.randint(0,int(360/interval))*interval
    else:
        random_integer = 0
    rotation_degree = random_integer
    theta = np.deg2rad(rotation_degree)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta),  0],
        [0,             0,              1]
    ])
    rotated_offsets = mic_offsets @ R.T
    mic_locs_rotated = mic_center + rotated_offsets
    mic_positions = mic_locs_rotated.T
    steer_vector_calc = ModeVector_torch(mic_positions, sample_rate, n_fft, 343, grid, "far", precompute=True)
    sv = steer_vector_calc.mode_vec
    return sv,(locs+rotation_degree)%360

def compute_correlation_matrices_torch(stft: torch.Tensor) -> torch.Tensor:
    """Return the time-averaged spatial covariance matrix, shape [F, C, C]."""
    x = stft.permute(2, 1, 0)  # [T, F, C]
    cov = torch.matmul(x.unsqueeze(-1), x.unsqueeze(-2).conj())
    return cov.mean(dim=0)


class DirectionGrid:
    """Far-field 360-degree XY grid used by MUSIC steering vectors."""
    def __init__(self):
        self.x = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_x.npy')
        self.y = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_y.npy')
        self.z = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_z.npy')


class AVPedNeuralMusicLoader(Dataset):
    """NeuralMUSIC loader for the current `data/pairs/<split>/<bag>` layout."""

    def __init__(
        self,
        root_path="data/pairs",
        split="train",
        mic_offsets=None,
        audio_channels=(0, 1, 2, 3),
        feature_type="magphase",
        geometry_aug=False,
        rotation_interval=None,
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        skip_empty_labels=True,
    ):
        super().__init__()
        self.root_path = Path(root_path)
        self.split = split
        self.split_root = self.root_path / split
        self.audio_channels = tuple(audio_channels)
        self.feature_type = feature_type
        self.geometry_aug = geometry_aug
        self.rotation_interval = rotation_interval
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.skip_empty_labels = skip_empty_labels
        self.samples = self._collect_samples()

        if mic_offsets is None:
            side = 45.7 / 1000 / 2
            mic_offsets = np.asarray(
                [
                    [side, side, 0.0],
                    [-side, side, 0.0],
                    [-side, -side, 0.0],
                    [side, -side, 0.0],
                ],
                dtype=np.float32,
            )
        self.mic_offsets = np.asarray(mic_offsets, dtype=np.float32)
        if len(self.audio_channels) != len(self.mic_offsets):
            raise ValueError(
                f"audio_channels has {len(self.audio_channels)} entries, "
                f"but mic_offsets has {len(self.mic_offsets)} microphones"
            )

        self.grid = DirectionGrid()
        self.steering_vector = ModeVector_torch(
            self.mic_offsets.T,
            self.sample_rate,
            self.n_fft,
            343,
            self.grid,
            mode="far",
            precompute=True,
        ).mode_vec.to(torch.complex64)
        self.window = torch.hann_window(self.n_fft)
        self.resize = Trans.Resize((self.n_fft // 2 + 1, 64), antialias=True)

    def _collect_samples(self):
        samples = []
        for bag_dir in sorted(path for path in self.split_root.iterdir() if path.is_dir()):
            for audio_path in sorted((bag_dir / "audio").glob("*.wav")):
                stem = audio_path.stem
                label_path = bag_dir / "labels" / f"{stem}.txt"
                if not label_path.exists():
                    continue
                if self.skip_empty_labels and not label_path.read_text().strip():
                    continue
                samples.append(
                    {
                        "bag": bag_dir.name,
                        "stem": stem,
                        "audio": audio_path,
                        "label": label_path,
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, path):
        sample_rate, audio = wavfile.read(path)
        if sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {sample_rate} Hz in {path}")
        if audio.ndim == 1:
            audio = audio[:, None]
        if max(self.audio_channels) >= audio.shape[1]:
            raise ValueError(
                f"Requested audio channels {self.audio_channels}, but {path} only has {audio.shape[1]} channels"
            )
        audio = audio[:, self.audio_channels].astype(np.float32)
        scale = np.max(np.abs(audio))
        if scale > 0:
            audio = audio / scale
        return torch.from_numpy(audio.T)

    @staticmethod
    def _load_doa_deg(path):
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError(f"Expected one bbox in {path}, got {len(lines)}")
        parts = lines[0].split()
        if len(parts) != 15:
            raise ValueError(f"Expected 15 label fields in {path}, got {len(parts)}")
        x, y = map(float, parts[11:13])
        return np.float32(np.degrees(np.arctan2(y, x)) % 360.0)

    def _spectrogram_process(self, audio):
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        correlation = compute_correlation_matrices_torch(stft)
        if self.feature_type == "magphase":
            magnitude = normalize_magnitude(torch.abs(stft))
            phase = normalize_phase(torch.angle(stft))
            spectrogram = torch.stack([magnitude, phase], dim=1).view(
                2 * audio.shape[0], stft.shape[1], stft.shape[2]
            )
        elif self.feature_type == "ipd":
            magnitude = normalize_magnitude(torch.abs(stft[0:1]))
            ref_phase = torch.angle(stft[0])
            features = [magnitude[0]]
            for channel in range(1, stft.shape[0]):
                phase_diff = torch.angle(stft[channel]) - ref_phase
                features.append(torch.sin(phase_diff))
                features.append(torch.cos(phase_diff))
            spectrogram = torch.stack(features, dim=0)
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")
        return self.resize(spectrogram).float(), correlation

    def __getitem__(self, index):
        sample = self.samples[index]
        audio = self._load_audio(sample["audio"])
        spectrogram, correlation = self._spectrogram_process(audio)
        doa = np.asarray([self._load_doa_deg(sample["label"])], dtype=np.float32)
        use_aug = self.geometry_aug and self.split == "train"
        steering_vector, doa = array_aug(
            self.mic_offsets,
            doa,
            transform=use_aug,
            grid=self.grid,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            interval=self.rotation_interval,
        )
        return spectrogram, torch.from_numpy(doa).float(), steering_vector.to(torch.complex64), correlation


# Backward-compatible name used by older scripts.
DAMUSIC_Loader = AVPedNeuralMusicLoader
