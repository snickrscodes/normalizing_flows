import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sum_no_batch

class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()
        self.conditional = False

    def forward(self, x):
        y = torch.exp(x)
        log_det = sum_no_batch(x)
        return y, log_det
    
    def inverse(self, x):
        y = torch.log(torch.abs(x))
        log_det = -sum_no_batch(y)
        return y, log_det
    
class Softplus(nn.Module):
    def __init__(self):
        super(Softplus, self).__init__()
        self.conditional = False

    def forward(self, x):
        y = F.softplus(x)
        log_det = sum_no_batch(torch.log(F.sigmoid(x)))
        return y, log_det
    
    def inverse(self, x):
        y = torch.log(torch.exp(x)-1)
        log_det = -sum_no_batch(torch.log(F.sigmoid(y)))
        return y, log_det
    
class Sigmoid(nn.Module):
    def __init__(self, eps=1.0e-6):
        super(Sigmoid, self).__init__()
        self.conditional = False
        self.eps = eps

    def forward(self, x):
        y = torch.sigmoid(x)
        log_det = sum_no_batch(torch.log(y*(1.0-y)))
        return y, log_det
    
    def inverse(self, x):
        x = torch.clamp(x, 0+self.eps, 1-self.eps)
        y = torch.log(x) - torch.log(1.0-x)
        log_det = -sum_no_batch(torch.log(x*(1.0-x)))
        return y, log_det

class Logit(nn.Module):
    def __init__(self, eps=1.0e-6):
        super(Logit, self).__init__()
        self.conditional = False
        self.eps = eps

    def forward(self, x):
        x = self.eps + (1-2*self.eps) * x
        y = torch.log(x/(1.0-x))
        log_det = -sum_no_batch(torch.log(x*(1.0-x)))
        return y, log_det
    
    def inverse(self, x):
        y = torch.sigmoid(x)
        log_det = sum_no_batch(torch.log(y*(1.0-y)))
        return y, log_det

class Tanh(nn.Module):
    def __init__(self, eps=1.0e-6):
        super(Tanh, self).__init__()
        self.conditional = False
        self.eps = eps

    def forward(self, x):
        y = torch.tanh(x)
        log_det = sum_no_batch(torch.log(1.0-torch.pow(y, 2)))
        return y, log_det
    
    def inverse(self, x):
        x = torch.clamp(x, 0+self.eps, 1-self.eps)
        y = torch.atanh(x)
        log_det = -sum_no_batch(torch.log(1.0-torch.pow(x, 2)))
        return y, log_det

class LogTanh(nn.Module):
    def __init__(self, eps=1.0e-6):
        super(LogTanh, self).__init__()
        self.conditional = False
        self.eps = eps

    def forward(self, x):
        # will only be defined if x > 0
        t = torch.clamp_min(torch.tanh(x), self.eps)
        y = torch.log(t)
        log_det = sum_no_batch(torch.log(1-torch.pow(t, 2))-y)
        return y, log_det
    
    def inverse(self, x):
        x = torch.clamp_max(x, -self.eps)
        ex = torch.exp(x)
        y = torch.atanh(ex)
        log_det = -sum_no_batch(torch.log(1-torch.pow(ex, 2))-x)
        return y, log_det
    
class LeakyRelu(nn.Module):
    def __init__(self, a=0.01):
        super(LeakyRelu, self).__init__()
        self.conditional = False
        self.a = a

    def forward(self, x):
        y = F.leaky_relu(x, negative_slope=self.a)
        log_det = sum_no_batch(torch.where(x >= 0, 0.0, torch.log(self.a)))
        return y, log_det

    def inverse(self, x):
        y = torch.where(x >= 0, x, x / self.a)
        log_det = -sum_no_batch(torch.where(x >= 0, 0.0, torch.log(self.a)))
        return y, log_det
    
class Softsign(nn.Module):
    def __init__(self):
        super(Softsign, self).__init__()
        self.conditional = False

    def forward(self, x):
        y = F.softsign(x)
        log_det = -2*sum_no_batch(torch.log(torch.abs(x)+1))
        return y, log_det

    def inverse(self, x):
        y = x / (1 - torch.abs(x))
        log_det = 2*sum_no_batch(torch.log(torch.abs(y)+1))
        return y, log_det