import torch

def regression_loss(y_pred,y_true):
    # print(y_true.shape,y_pred.shape)
    mse_loss = torch.nn.L1Loss()
    mse_loss1 = mse_loss(y_pred[:,:3],y_true[:,:3])
    mse_loss2 = mse_loss(y_pred[:,3:],y_true[:,3:])
    total_loss = mse_loss1 + 0.1 * mse_loss2
    return total_loss

