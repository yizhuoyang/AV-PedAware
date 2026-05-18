import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_music_gt(centers: torch.Tensor, sigma=10):
    batch_size = centers.shape[0]
    angles = torch.arange(360, device=centers.device).unsqueeze(0)  # [1, 360]
    centers_expanded = centers.repeat(1, 3)  # [batch_size, 3]
    centers_expanded[:, 1] += 360  # center+360
    centers_expanded[:, 2] -= 360  # center-360
    gauss = torch.exp(-((angles - centers_expanded.unsqueeze(-1)) ** 2) / (2 * sigma ** 2))  # [batch_size, 3, 360]
    gt_spectrum = gauss.sum(dim=1)

    gt_spectrum /= gt_spectrum.max(dim=1, keepdim=True)[0]

    return gt_spectrum


def generate_music_gt_class(doa_list: list, sigma=10):

    device = doa_list[0].device if isinstance(doa_list[0], torch.Tensor) else "cpu"

    doa_all = torch.cat(doa_list)  # [sum_sources]
    batch_sizes = torch.tensor([len(d) for d in doa_list], device=device)
    batch_idx = torch.repeat_interleave(torch.arange(len(doa_list), device=device), batch_sizes)
    angles = torch.arange(360, device=device).unsqueeze(0)
    doa_expanded = torch.cat([doa_all, doa_all + 360, doa_all - 360], dim=0).unsqueeze(-1)  # [3N, 1]
    batch_idx_expanded = batch_idx.repeat(3)  # [3N]
    gauss = torch.exp(-((angles - doa_expanded) ** 2) / (2 * sigma ** 2))  # [3N, 360]
    gt = torch.zeros(len(doa_list), 360, device=device)
    gt = gt.index_add(0, batch_idx_expanded, gauss)
    gt /= gt.max(dim=1, keepdim=True)[0].clamp(min=1e-6)

    return gt  # [B, 360]

def autocorrelation_matrix(X: torch.Tensor, lag: int):
    """
    Computes the autocorrelation matrix for a given lag of the input samples.

    Args:
    -----
        X (torch.Tensor): Samples matrix input with shape [N, T].
        lag (int): The requested delay of the autocorrelation calculation.

    Returns:
    --------
        torch.Tensor: The autocorrelation matrix for the given lag.

    """
    Rx_lag = torch.zeros(X.shape[0], X.shape[0], dtype=torch.complex128).to(device)
    for t in range(X.shape[1] - lag):
        # meu = torch.mean(X,1)
        x1 = torch.unsqueeze(X[:, t], 1).to(device)
        x2 = torch.t(torch.unsqueeze(torch.conj(X[:, t + lag]), 1)).to(device)
        Rx_lag += torch.matmul(x1 - torch.mean(X), x2 - torch.mean(X)).to(device)
    Rx_lag = Rx_lag / (X.shape[-1] - lag)
    Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), 0)
    return Rx_lag


# def create_autocorrelation_tensor(X: torch.Tensor, tau: int) -> torch.Tensor:
def create_autocorrelation_tensor(X: torch.Tensor, tau: int):
    """
    Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (BS, N, T).
        tau (int): Maximal time difference for the autocorrelation tensor.

    Returns:
    --------
        torch.Tensor: Tensor containing all the autocorrelation matrices,
                    with size (Batch size, tau, 2N, N).

    Raises:
    -------
        None

    """
    Rx_tau = []
    for i in range(tau):
        Rx_tau.append(autocorrelation_matrix(X, lag=i))
    Rx_autocorr = torch.stack(Rx_tau, dim=0)
    return Rx_autocorr

def create_cov_tensor(X: torch.Tensor):
    """
    Creates a 3D tensor of size (NxNx3) containing the real part, imaginary part, and phase component of the covariance matrix.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (N, T).

    Returns:
    --------
        Rx_tensor (torch.Tensor): Tensor containing the auto-correlation matrices, with size (Batch size, N, N, 3).

    Raises:
    -------
        None

    """
    Rx = torch.cov(X)
    Rx_tensor = torch.stack((torch.real(Rx), torch.imag(Rx), torch.angle(Rx)), 2)
    return Rx_tensor
