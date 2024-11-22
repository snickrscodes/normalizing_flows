import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flows.spline import RQS, QS, CBS

class AffineScalar(nn.Module):
    def __init__(self, scale, shift):
        super(AffineScalar, self).__init__()
        self.register_buffer('scale', torch.tensor(scale if scale is not None else 1.0))
        self.register_buffer('shift', torch.tensor(shift if shift is not None else 0.0))

    def forward(self, input):
        x = input * self.scale + self.shift
        log_det = torch.log(torch.abs(self.scale)) * input.numel() / input.size(0)
        return x, log_det
    
    def inverse(self, input):
        x = (input - self.shift) / self.scale
        log_det = -torch.log(torch.abs(self.scale)) * input.numel() / input.size(0)
        return x, log_det

# regular ffn
class FFN(nn.Module):
    def __init__(self, n_in, hidden_dims: tuple[int], out_mult, act):
        super(FFN, self).__init__()
        self.conditional = False
        self.lins = nn.ModuleList()
        self.act = act
        self.lins.append(nn.Linear(n_in, hidden_dims[0]))
        for i in range(len(hidden_dims)-1):
            self.lins.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.lins.append(nn.Linear(hidden_dims[-1], n_in*out_mult))
    
    def forward(self, x):
        for i in range(len(self.lins)-1):
            x = self.act(self.lins[i](x))
        x = self.lins[-1](x)
        return x
    
# ffn for conditional inputs
class CFFN(nn.Module):
    def __init__(self, n_in, context_in, hidden_dims: tuple[int], out_mult, act):
        super(CFFN, self).__init__()
        self.conditional = False
        self.lins = nn.ModuleList()
        self.act = act
        self.context_lin = nn.Linear(context_in, hidden_dims[0], bias=False)
        self.lins.append(nn.Linear(n_in, hidden_dims[0]))
        for i in range(len(hidden_dims)-1):
            self.lins.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.lins.append(nn.Linear(hidden_dims[-1], n_in*out_mult))
    
    def forward(self, input, context):
        x = self.act(self.context_lin(context) + self.lins[0](input))
        for i in range(1, len(self.lins)-1):
            x = self.act(self.lins[i](x))
        x = self.lins[-1](x)
        return x

class Coupling(nn.Module):
    def __init__(self, n_in):
        super(Coupling, self).__init__()
        self.conditional = False
        self.register_buffer('mask', (torch.arange(0, n_in) < n_in//2))
        self.inv_mask = ~self.mask

    def forward(self, x):
        id_x = x[:, self.mask] # identity (unchanged) features
        y = torch.zeros_like(x)
        log_det = torch.zeros_like(x)
        y[:, self.mask] = id_x
        params = self.get_params(id_x)
        y[:, self.inv_mask], log_det[:, self.inv_mask] = self.transform(x[:, self.inv_mask], *params)
        return y, torch.sum(log_det, dim=-1)
    
    def inverse(self, x):
        id_x = x[:, self.mask] # identity (unchanged) features
        y = torch.zeros_like(x)
        log_det = torch.zeros_like(x)
        y[:, self.mask] = id_x
        params = self.get_params(id_x)
        # the inverse method should then return a negative log abs det
        y[:, self.inv_mask], log_det[:, self.inv_mask] = self.inverse_transform(x[:, self.inv_mask], *params)
        return y, torch.sum(log_det, dim=-1)
    
    def transform(self, x, params):
        raise NotImplementedError()
    
    def inverse_transform(self, x, params):
        raise NotImplementedError()
    
    def get_params(self, x):
        raise NotImplementedError()

class AffineCoupling(Coupling):
    def __init__(self, n_in, hidden_dims, act=F.relu):
        super().__init__(n_in)
        self.t = FFN(n_in//2, hidden_dims, 1, act)
        self.s = FFN(n_in//2, hidden_dims, 1, act)
        self.scale = nn.Parameter(torch.zeros(n_in//2))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.scale)

    def get_params(self, x): # x is half the input (identity features)
        s = self.s(x) * self.scale
        t = self.t(x)
        return s, t
    
    def transform(self, x, s, t): # x is the transform half of the input
        y = x * torch.exp(s) + t
        return y, s
    
    def inverse_transform(self, x, s, t):
        y = (x - t) * torch.exp(-s)
        return y, -s

class AdditiveCoupling(Coupling):
    def __init__(self, n_in, hidden_dims, act=F.relu):
        super().__init__(n_in)
        self.t = FFN(n_in//2, hidden_dims, 1, act)

    def get_params(self, x):
        return self.t(x)
    
    def transform(self, x, t): # x is the transform half of the input
        return x+t, torch.zeros(x.size(0), device=x.device)
    
    def inverse_transform(self, x, s, t):
        return x-t, torch.zeros(x.size(0), device=x.device)

class CouplingSpline(Coupling):
    def __init__(self, n_in, fn, num_bins, tail_bound, hidden_dims, act=F.relu):
        super().__init__(n_in)
        self.num_bins = num_bins
        self.n_in = n_in
        self.scale = np.sqrt(np.prod(hidden_dims) // len(hidden_dims)) # avg length of each hidden dim
        self.spline, self.out_mult = self.get_spline(fn, num_bins, tail_bound)
        self.ffn = FFN(n_in//2, hidden_dims, self.out_mult, act)

    def get_spline(self, fn: str, num_bins: int, tail_bound: float):
        match fn.lower():
            case 'rqs':
                return RQS(tail_bound, num_bins), 3*num_bins-1
            case 'cbs':
                return CBS(tail_bound, num_bins), 2*num_bins+2
            case 'qs':
                return QS(tail_bound, num_bins), 2*num_bins-1

    def get_params(self, u):
        x = self.ffn(u)
        x = x.view(-1, self.n_in//2, self.out_mult)
        params = list(torch.split(x, self.num_bins, dim=-1))
        params[0] = params[0] / self.scale
        params[1] = params[1] / self.scale
        return params
    
    def transform(self, x, *params): # x is the transform half of the input
        return self.spline(x, *params)
    
    def inverse_transform(self, x, *params):
        return self.spline.inverse(x, *params)