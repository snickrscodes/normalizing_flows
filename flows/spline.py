import torch
import torch.nn as nn
import utils

# base class for spline transforms
class Spline(nn.Module):
    def __init__(self):
        super(Spline, self).__init__()
        self.conditional = False
    
    def forward(self, x, params):
        raise NotImplementedError()
    
    def inverse(self, x, params):
        raise NotImplementedError()

class RQS(Spline):
    def __init__(self, tail_bound, num_bins, min_bin_width=1.0e-3, min_bin_height=1.0e-3, min_derivative=1.0e-3):
        super().__init__()
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tail_bound = tail_bound

    def forward(self, x, w_, h_, d_):
        return utils.unconstrained_rational_quadratic_spline(
            inputs=x,
            unnormalized_widths=w_,
            unnormalized_heights=h_,
            unnormalized_derivatives=d_,
            inverse=False,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative
        )
    
    def inverse(self, x, w_, h_, d_):
        return utils.unconstrained_rational_quadratic_spline(
            inputs=x,
            unnormalized_widths=w_,
            unnormalized_heights=h_,
            unnormalized_derivatives=d_,
            inverse=True,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative
        )
    
class CBS(Spline):
    def __init__(self, tail_bound, num_bins, min_bin_width=1.0e-3, min_bin_height=1.0e-3, quadratic_threshold=1e-3, eps=1e-5):
        super().__init__()
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.quadratic_threshold = quadratic_threshold
        self.eps = eps
        self.tail_bound = tail_bound

    def forward(self, x, w_, h_, dl_, dr_):
        return utils.unconstrained_cubic_spline(
            inputs=x,
            unnormalized_widths=w_,
            unnormalized_heights=h_,
            unnorm_derivatives_left=dl_,
            unnorm_derivatives_right=dr_,
            inverse=False,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            quadratic_threshold=self.quadratic_threshold,
            eps=self.eps
        )
    
    def inverse(self, x, w_, h_, d_):
        return utils.unconstrained_cubic_spline(
            inputs=x,
            unnormalized_widths=w_,
            unnormalized_heights=h_,
            unnorm_derivatives_left=d_[..., 0][..., None],
            unnorm_derivatives_right=d_[..., 1][..., None],
            inverse=True,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            quadratic_threshold=self.quadratic_threshold,
            eps=self.eps
        )
    
class QS(Spline):
    def __init__(self, tail_bound, num_bins, min_bin_width=1.0e-3, min_bin_height=1.0e-3):
        super().__init__()
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.tail_bound = tail_bound

    def forward(self, x, w_, h_):
        return utils.unconstrained_quadratic_spline(
            inputs=x,
            unnormalized_widths=w_,
            unnormalized_heights=h_,
            inverse=False,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height
        )
    
    def inverse(self, x, w_, h_):
        return utils.unconstrained_quadratic_spline(
            inputs=x,
            unnormalized_widths=w_,
            unnormalized_heights=h_,
            inverse=True,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height
        )