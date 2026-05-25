import torch

def regression_loss(y_pred,y_true):
    # print(y_true.shape,y_pred.shape)
    mse_loss = torch.nn.L1Loss()
    mse_loss1 = mse_loss(y_pred[:,:3],y_true[:,:3])
    mse_loss2 = mse_loss(y_pred[:,3:],y_true[:,3:])
    total_loss = mse_loss1 + 0.1 * mse_loss2
    return total_loss


def angle_vector_loss(y_pred, y_true):
    y_pred = torch.nn.functional.normalize(y_pred, dim=1)
    y_true = torch.nn.functional.normalize(y_true, dim=1)
    return (1.0 - torch.sum(y_pred * y_true, dim=1)).mean()


def angle_distribution_loss(y_pred, y_true):
    """Soft-label cross entropy for azimuth probability distributions."""
    return -(y_true * torch.log(y_pred.clamp_min(1e-8))).sum(dim=1).mean()


def distribution_to_angle_rad(distribution):
    """Decode each probability distribution using its most likely angle bin."""
    num_bins = distribution.shape[1]
    peak_bin = torch.argmax(distribution, dim=1)
    return peak_bin.to(distribution.dtype) * (2.0 * torch.pi / num_bins)


def angular_error_from_distribution_deg(prediction, target):
    pred_angle = distribution_to_angle_rad(prediction)
    target_angle = distribution_to_angle_rad(target)
    diff = torch.atan2(torch.sin(pred_angle - target_angle), torch.cos(pred_angle - target_angle))
    return torch.rad2deg(torch.abs(diff))
