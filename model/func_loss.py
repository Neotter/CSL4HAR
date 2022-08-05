import torch
import torch.nn as nn

def L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def mse_loss(reduction = 'none'):
    return nn.MSELoss(reduction=reduction)