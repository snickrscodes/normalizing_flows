import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flows.spline import RQS, CBS, QS
from utils import generate_masks

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask: torch.Tensor):
        super(MaskedLinear, self).__init__()
        self.conditional = False
        self.weight = nn.Parameter(torch.zeros((out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros((out_features,)))
        self.register_buffer('mask', mask.to(self.weight.dtype).t()) # transpose the mask to work with weight
        self.reset_parameters()

    def reset_parameters(self, variance=1.0):
        fan_in = self.weight.size(1) # special case
        std = np.sqrt(variance / fan_in)
        nn.init.normal_(self.weight, mean=0.0, std=std)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        return F.linear(x, self.weight*self.mask, self.bias)

class MADE(nn.Module):
    # n_out assumed to equal n_in
    def __init__(self, n_in, hidden_dims: tuple[int], out_mult, mode, act):
        super(MADE, self).__init__()
        self.conditional = False
        self.lins = nn.ModuleList()
        self.act = act
        masks, self.input_order = generate_masks(n_in, hidden_dims, out_mult, mode)
        self.lins.append(MaskedLinear(n_in, hidden_dims[0], masks[0]))
        for i in range(len(hidden_dims)-1):
            self.lins.append(MaskedLinear(hidden_dims[i], hidden_dims[i+1], masks[i+1]))
        self.lins.append(MaskedLinear(hidden_dims[-1], n_in*out_mult, masks[-1]))

    def forward(self, input):
        x = self.act(self.lins[0](input))
        for i in range(1, len(self.lins)-1):
            x = self.act(self.lins[i](x))
        x = self.lins[-1](x)
        return x
    
class ConditionalMADE(nn.Module):
    # n_out assumed to equal n_in
    def __init__(self, n_in, context_in, hidden_dims: tuple[int], out_mult, mode, act):
        super(ConditionalMADE, self).__init__()
        self.conditional = True
        self.lins = nn.ModuleList()
        self.act = act
        self.context_lin = nn.Linear(context_in, hidden_dims[0], bias=False)
        masks, self.input_order = generate_masks(n_in, hidden_dims, out_mult, mode)
        self.lins.append(MaskedLinear(n_in, hidden_dims[0], masks[0]))
        for i in range(len(hidden_dims)-1):
            self.lins.append(MaskedLinear(hidden_dims[i], hidden_dims[i+1], masks[i+1]))
        self.lins.append(MaskedLinear(hidden_dims[-1], n_in*out_mult, masks[-1]))

    def forward(self, input, context):
        x = self.act(self.context_lin(context) + self.lins[0](input))
        for i in range(1, len(self.lins)-1):
            x = self.act(self.lins[i](x))
        x = self.lins[-1](x)
        return x
    
# made with random numbers
class MAFLayer(nn.Module):
    def __init__(self, n_in, hidden_dims: tuple[int], mode='random', act=F.relu):
        super(MAFLayer, self).__init__()
        self.conditional = False
        self.made = MADE(n_in, hidden_dims, 2, mode, act)

    def forward(self, input):
        mean, logp = torch.chunk(self.made(input), 2, -1)
        u = (input - mean) * torch.exp(0.5 * logp)
        log_det = 0.5 * torch.sum(logp, dim=-1)
        return u, log_det

    def inverse(self, u):
        x = torch.zeros_like(u)
        log_det = torch.zeros(u.size(0))
        for dim in range(u.size(1)):
            mean, logp = torch.chunk(self.made(x), 2, -1)
            logp = torch.minimum(-0.5 * logp, torch.tensor(10.0))
            idx = torch.argwhere(self.made.input_order == dim)[0, 0]
            x[:, idx] = mean[:, idx] + u[:, idx] * torch.exp(logp[:, idx])
            log_det += logp[:, dim] # copy relevant data
        return x, log_det
    
class AutoregressiveSpline(nn.Module):
    def __init__(self, fn, num_bins, tail_bound, n_in, hidden_dims, mode='sequential', act=F.relu):
        super(AutoregressiveSpline, self).__init__()
        self.conditional = False
        self.num_bins = num_bins
        self.n_in = n_in
        self.scale = np.sqrt(np.prod(hidden_dims) // len(hidden_dims)) # avg length of each hidden dim
        self.spline, self.out_mult = self.get_spline(fn, num_bins, tail_bound)
        self.made = MADE(n_in, hidden_dims, self.out_mult, mode, act)

    def get_spline(self, fn: str, num_bins: int, tail_bound: float):
        match fn.lower():
            case 'rqs':
                return RQS(tail_bound, num_bins), 3*num_bins-1
            case 'cbs':
                return CBS(tail_bound, num_bins), 2*num_bins+2
            case 'qs':
                return QS(tail_bound, num_bins), 2*num_bins-1
    
    def get_params(self, u):
        x = self.made(u)
        b = u.size(0)
        x = x.view(b, self.n_in, self.out_mult)
        params = list(torch.split(x, self.num_bins, dim=-1))
        # scale the widths and heights (first 2 params)
        # authors of nsf repo suggested it pre softmax
        params[0] = params[0] / self.scale
        params[1] = params[1] / self.scale
        return params

    def forward(self, x):
        params = self.get_params(x)
        y, log_det = self.spline(x, *params)
        return y, torch.sum(log_det, dim=-1)
    
    def inverse(self, x):
        y = torch.zeros_like(x)
        log_det = torch.zeros_like(x)
        for _ in range(self.n_in):
            params = self.get_params(y)
            y, log_det = self.spline.inverse(x, *params)
        return y, torch.sum(log_det, dim=-1)
    
class ConditionalAutoregressiveSpline(nn.Module):
    def __init__(self, fn, num_bins, tail_bound, n_in, context_in, hidden_dims, mode='sequential', act=F.relu):
        super(ConditionalAutoregressiveSpline, self).__init__()
        self.conditional = True
        self.num_bins = num_bins
        self.n_in = n_in
        self.scale = np.sqrt(np.prod(hidden_dims) // len(hidden_dims)) # avg length of each hidden dim
        self.spline, self.out_mult = self.get_spline(fn, num_bins, tail_bound)
        self.made = ConditionalMADE(n_in, context_in, hidden_dims, self.out_mult, mode, act)

    def get_spline(self, fn: str, num_bins: int, tail_bound: float):
        match fn.lower():
            case 'rqs':
                return RQS(tail_bound, num_bins), 3*num_bins-1
            case 'cbs':
                return CBS(tail_bound, num_bins), 2*num_bins+2
            case 'qs':
                return QS(tail_bound, num_bins), 2*num_bins-1
    
    def get_params(self, u, context):
        x = self.made(u, context)
        b = u.size(0)
        x = x.view(b, self.n_in, self.out_mult)
        params = list(torch.split(x, self.num_bins, dim=-1))
        params[0] = params[0] / self.scale
        params[1] = params[1] / self.scale
        return params

    def forward(self, x, context):
        params = self.get_params(x, context)
        y, log_det = self.spline(x, *params)
        return y, torch.sum(log_det, dim=-1)
    
    def inverse(self, x, context):
        y = torch.zeros_like(x)
        log_det = torch.zeros_like(x)
        for _ in range(self.n_in):
            params = self.get_params(y, context)
            y, log_det = self.spline.inverse(x, *params)
        return y, torch.sum(log_det, dim=-1)