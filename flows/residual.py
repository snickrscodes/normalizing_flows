import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import neumann_logdet_estimator, basic_logdet_estimator, geometric_1mcdf, geometric_sample
import numpy as np

# as of now this will only work with 2d inputs - simplified
class Resnet(nn.Module):
    def __init__(self, transform, geom_p=0.5, lmbda=2.0, n_samples=1, n_exact_terms=2):
        super(Resnet, self).__init__()
        self.transform = transform
        self.conditional = transform.conditional
        self.n_samples = n_samples
        self.n_exact_terms = n_exact_terms
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1.0 - geom_p)))
        self.register_buffer('last_n_samples', torch.zeros(self.n_samples))
        self.register_buffer('last_firmom', torch.zeros(1))
        self.register_buffer('last_secmom', torch.zeros(1))

    def forward(self, *args):
        g, logdetgrad = self._logdetgrad(args)
        return args[0] + g, -logdetgrad

    def inverse(self, y):
        x = self._inverse_fixed_point(y)
        return x, self._logdetgrad(x)[1]

    def _inverse_fixed_point(self, y, atol=1e-5, rtol=1e-5):
        z, logd = self.transform(y)
        x, x_prev = y - self.transform(y), y
        i = 0
        tol = atol + y.abs() * rtol
        while not torch.all((x - x_prev) ** 2 / tol < 1):
            x, x_prev = y - self.nnet(x), x
            i += 1
            if i > 1000:
                break
        return x

    def _logdetgrad(self, *x):
        with torch.enable_grad():
            geom_p = torch.sigmoid(self.geom_p).item()
            sample_fn = lambda m: geometric_sample(geom_p, m)
            rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)
            n_samples = sample_fn(self.n_samples)
            if self.training:
                n_power_series = max(n_samples) + self.n_exact_terms
                coeff_fn = (
                    lambda k: 1
                    / rcdf_fn(k, self.n_exact_terms)
                    * sum(n_samples >= k - self.n_exact_terms)
                    / len(n_samples)
                )
            else:
                n_power_series = max(n_samples) + 20
                coeff_fn = (
                    lambda k: 1
                    / rcdf_fn(k, 20)
                    * sum(n_samples >= k - 20)
                    / len(n_samples)
                )

            vareps = torch.randn_like(x)
            if self.training:
                estimator_fn = neumann_logdet_estimator
            else:
                estimator_fn = basic_logdet_estimator

            x = x.requires_grad_(True)
            g, log_det = self.transform(*x)
            logdetgrad = estimator_fn(g, x[0], n_power_series, vareps, coeff_fn, self.training)
            if self.training:
                self.last_n_samples.copy_(torch.tensor(n_samples).to(self.last_n_samples))
                estimator = logdetgrad.detach()
                self.last_firmom.copy_(torch.mean(estimator).to(self.last_firmom))
                self.last_secmom.copy_(torch.mean(estimator**2).to(self.last_secmom))
            return g, log_det+logdetgrad.view(-1)