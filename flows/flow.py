import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# in this implementation, prior is a multivariate normal
class Flow(nn.Module):
    def __init__(self, features, transforms):
        super(Flow, self).__init__()
        self.features = features
        if isinstance(transforms, (list, tuple)):
            self.transforms = nn.ModuleList(transforms)
        elif isinstance(transforms, nn.Module):
            self.transforms = nn.ModuleList([transforms])
        else: # it has to be a module list
            self.transforms = transforms
        if not all(hasattr(transform, 'conditional') for transform in self.transforms):
            missing_attrs = [type(transform).__name__ for transform in self.transforms if not hasattr(transform, 'conditional')]
            raise AttributeError(f"some transforms missing 'conditional' attribute: {', '.join(missing_attrs)}")
    
    def forward(self, x, context=None):
        z = x
        log_det = torch.zeros(x.size(0))
        for transform in self.transforms:
            if transform.conditional:
                z, logd = transform(z, context)
            else:
                z, logd = transform(z)
            log_det = log_det + logd
        logp = self.log_prob_standard(z)
        return z, logp, log_det
    
    def inverse(self, x, context=None):
        z = x
        log_det = torch.zeros(x.size(0))
        for transform in reversed(self.transforms):
            if transform.conditional:
                z, logd = transform.inverse(z, context)
            else:
                z, logd = transform.inverse(z)
            log_det = log_det + logd # they're all negative, don't subtract
        return z, log_det

    def loss(self, x, context=None):
        z, log_prob, log_det = self(x, context)
        nll = -torch.mean(log_prob + log_det)
        return nll
    
    def log_prob(self, z, mu, cov):
        d = z.size(-1)
        diff = z - mu
        dist = torch.einsum('bi,ij,bj->b', diff, torch.linalg.inv(cov), diff)
        return -0.5*(d*np.log(2*np.pi) + torch.logdet(cov) + dist)

    # simplified - mean = 0, cov = I
    def log_prob_standard(self, z):
        d = z.size(-1)
        return -0.5*d*np.log(2*np.pi) - 0.5*torch.sum(z.pow(2), dim=-1)