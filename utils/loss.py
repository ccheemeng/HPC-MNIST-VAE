import torch
import torch.nn as nn

def _kl_divergence(mean, log_var):
    return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

class BCELoss():
    @staticmethod
    def loss(x, x_hat, mean, log_var):
        reconstruction_error = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        return _kl_divergence(mean, log_var) + reconstruction_error
    
class MSELoss():
    @staticmethod
    def loss(x, x_hat, mean, log_var):
        reconstruction_error = nn.functional.mse_loss(x_hat, x, reduction="sum")
        return _kl_divergence(mean, log_var) + reconstruction_error