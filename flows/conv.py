import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# need to come up with a better name for this (supposed to be a 1x1 invertible conv)
class LUConv(nn.Module):
    def __init__(self, ch: int):
        super(LUConv, self).__init__()
        self.conditional = False
        self.ch = ch
        w = torch.randn(ch, ch)
        q, _ = torch.linalg.qr(w) # orthogonalize to ensure invertibility
        self.weight = nn.Parameter(q)
        p, lower, upper = torch.linalg.lu(q) # parameterize the lu decomposition
        s = torch.diag(upper)
        self.register_buffer("p", p)
        self.register_buffer("sign_s", torch.sign(s))
        self.log_s = nn.Parameter(torch.log(torch.abs(s))) # numerical stability
        self.lower = nn.Parameter(lower)
        self.upper = nn.Parameter(torch.triu(upper, 1))
        self.eye = torch.eye(ch, ch)
    
    def forward(self, x):
        lower = torch.tril(self.lower, -1) + self.eye # main diagonal must be 1 for LU to be valid
        upper = torch.triu(self.upper, diagonal=1) + torch.diag(self.sign_s*torch.exp(self.log_s)) # |det(W)| = |det(U)|
        weight = torch.matmul(self.p, torch.matmul(lower, upper))
        log_det = torch.sum(self.log_s)
        return torch.matmul(x, weight), log_det

    def inverse(self, x):
        lower = torch.tril(self.lower, -1) + self.eye # main diagonal must be 1 for LU to be valid
        upper = torch.triu(self.upper, diagonal=1) + torch.diag(self.sign_s*torch.exp(self.log_s)) # |det(W)| = |det(U)|
        lower_inv = torch.linalg.inv(lower) # easier to invert triangular matrices
        upper_inv = torch.linalg.inv(upper)
        p_inv = torch.linalg.inv(self.p)
        weight_inv = torch.matmul(upper_inv, torch.matmul(lower_inv, p_inv))
        log_det = -torch.sum(self.log_s)
        return torch.matmul(x, weight_inv), log_det