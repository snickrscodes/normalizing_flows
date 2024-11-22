import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BatchNorm(nn.Module):
    def __init__(self, features: int, ndim: int, dim=1, momentum=0.99, eps=1.0e-6):
        super(BatchNorm, self).__init__()
        self.conditional = False
        self.momentum = momentum
        self.eps = eps
        self.axes = list(range(ndim)); del self.axes[dim]
        param_shape = [1] * ndim; param_shape[dim] = features
        self.beta = nn.Parameter(torch.zeros(param_shape))
        self.log_gamma = nn.Parameter(torch.zeros(param_shape))
        self.register_buffer('moving_mean', torch.zeros(param_shape))
        self.register_buffer('moving_var', torch.ones(param_shape))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.beta, 0.0)
        nn.init.constant_(self.log_gamma, 0.0)

    def forward(self, x):
        with torch.no_grad():
            if self.training:
                var, mean = torch.var_mean(x, dim=self.axes, keepdim=True)
                var.add_(self.eps)
                self.moving_mean.mul_(self.momentum).add_(mean*(1.0-self.momentum))
                self.moving_var.mul_(self.momentum).add_(var*(1.0-self.momentum))
            else:
                var, mean = self.moving_var+self.eps, self.moving_mean
        x_hat = (x - mean) / torch.sqrt(var) * torch.exp(self.log_gamma) + self.beta # affine transformation
        log_det = torch.sum(self.log_gamma - 0.5 * torch.log(var))
        return x_hat, log_det
    
    def inverse(self, x):
        with torch.no_grad():
            if self.training: # this should never happen though
                var, mean = torch.var_mean(x, dim=self.axes, keepdim=True)
                var.add_(self.eps)
            else:
                var, mean = self.moving_var+self.eps, self.moving_mean
        x_hat = (x - self.beta) * torch.exp(-self.log_gamma) * torch.sqrt(var) + mean
        log_det = -torch.sum(self.log_gamma - 0.5 * torch.log(var))
        return x_hat, log_det

# for 4d inputs
class ActNorm(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conditional = False
        self.beta = nn.Parameter(torch.zeros((1, ch, 1, 1), dtype=torch.float32))
        self.log_gamma = nn.Parameter(torch.zeros((1, ch, 1, 1), dtype=torch.float32))

    def forward(self, x):
        _, _, h, w = x.size()
        z = x * torch.exp(self.log_gamma) + self.beta
        log_det = torch.sum(self.log_gamma) * h * w
        return z, log_det

    def inverse(self, z):
        _, _, h, w = x.size()
        x = (z - self.beta) / torch.exp(self.log_gamma)
        log_det = -torch.sum(self.log_gamma) * h * w
        return x, log_det