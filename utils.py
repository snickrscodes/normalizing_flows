import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

def generate_masks(n_in, hidden_dims: tuple[int], out_mult, mode):
    degrees = []
    # generate degrees
    if mode == 'random':
        degrees.append(torch.randperm(n_in, dtype=torch.long))
        for i in range(len(hidden_dims)):
            min_val = torch.min(degrees[i])
            degrees.append(torch.randint(min_val, n_in, (hidden_dims[i],), dtype=torch.long))
    elif mode == 'sequential':
        degrees.append(torch.arange(0, n_in, dtype=torch.long))
        for i in range(len(hidden_dims)):
            degrees.append(torch.arange(0, hidden_dims[i], dtype=torch.long) % n_in)
    else:
        raise ValueError(mode + ' is not a supported mask initialization mode')
    degrees.append(degrees[0])
    # generate masks (when sequential kinda like an upper triangular matrix)
    masks = []
    for prev_degree, next_degree in zip(degrees[:-1], degrees[1:]):
        masks.append(prev_degree[..., None] <= next_degree)
    masks[-1] = torch.tile(masks[-1], (1, out_mult)) # tile along last dimension
    return masks, degrees[0]

def sum_no_batch(x: torch.Tensor):
    return torch.sum(x, dim=x.size()[1:], keepdim=True) # all but batch dim (dim 0)

def rsq_fn(y_hat, y, w) -> float:
    num = torch.sum(w*(y-y_hat)**2)
    den = torch.sum(w*y.pow(2))
    score = 1 - num/(den+1e-9)
    return score.item()

# credit: https://github.com/rtqichen/residual-flows/blob/master/resflows/layers/iresblock.py

def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)

def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)

def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.tensor(0.).to(x)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        delta = (-1)**(k + 1) / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1)**k * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
    return logdetgrad

# credit: authors of the nsf paper and source code
# they made all of these spline implementations
# at https://github.com/bayesiains/nsf/blob/master/nde/transforms/splines/rational_quadratic.py

def cbrt(x):
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tail_bound=1.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)
        
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    if torch.any(inside_interval_mask):
        (
            outputs[inside_interval_mask],
            logabsdet[inside_interval_mask],
        ) = rational_quadratic_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative
        )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):

    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives, beta=1)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        # root = (- b + torch.sqrt(discriminant)) / (2 * a)
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet
    
def unconstrained_quadratic_spline(inputs,
                                   unnormalized_widths,
                                   unnormalized_heights,
                                   inverse=False,
                                   tail_bound=1.,
                                   min_bin_width=1e-3,
                                   min_bin_height=1e-3):

    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)
        
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height
    )

    return outputs, logabsdet

def quadratic_spline(inputs,
                     unnormalized_widths,
                     unnormalized_heights,
                     inverse=False,
                     left=0., right=1., bottom=0., top=1.,
                     min_bin_width=1e-3,
                     min_bin_height=1e-3):

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    unnorm_heights_exp = torch.exp(unnormalized_heights)

    if unnorm_heights_exp.shape[-1] == num_bins - 1:
        # Set boundary heights s.t. after normalization they are exactly 1.
        first_widths = 0.5 * widths[..., 0]
        last_widths = 0.5 * widths[..., -1]
        numerator = (0.5 * first_widths * unnorm_heights_exp[..., 0]
                     + 0.5 * last_widths * unnorm_heights_exp[..., -1]
                     + torch.sum(((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2)
                                 * widths[..., 1:-1], dim=-1))
        constant = numerator / (1 - 0.5 * first_widths - 0.5 * last_widths)
        constant = constant[..., None]
        unnorm_heights_exp = torch.cat([constant, unnorm_heights_exp, constant], dim=-1)

    unnormalized_area = torch.sum(((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2)
                                  * widths, dim=-1)[..., None]
    heights = unnorm_heights_exp / unnormalized_area
    heights = min_bin_height + (1 - min_bin_height) * heights

    bin_left_cdf = torch.cumsum(((heights[..., :-1] + heights[..., 1:]) / 2) * widths, dim=-1)
    bin_left_cdf[..., -1] = 1.
    bin_left_cdf = F.pad(bin_left_cdf, pad=(1, 0), mode='constant', value=0.0)

    bin_locations = torch.cumsum(widths, dim=-1)
    bin_locations[..., -1] = 1.
    bin_locations = F.pad(bin_locations, pad=(1, 0), mode='constant', value=0.0)

    if inverse:
        bin_idx = searchsorted(bin_left_cdf, inputs)[..., None]
    else:
        bin_idx = searchsorted(bin_locations, inputs)[..., None]

    input_bin_locations = bin_locations.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_left_cdf = bin_left_cdf.gather(-1, bin_idx)[..., 0]

    input_left_heights = heights.gather(-1, bin_idx)[..., 0]
    input_right_heights = heights.gather(-1, bin_idx+1)[..., 0]

    a = 0.5 * (input_right_heights - input_left_heights) * input_bin_widths
    b = input_left_heights * input_bin_widths
    c = input_left_cdf

    if inverse:
        c_ = c - inputs
        alpha = (-b + torch.sqrt(b.pow(2) - 4*a*c_)) / (2*a)
        outputs = alpha * input_bin_widths + input_bin_locations
        outputs = torch.clamp(outputs, 0, 1)
        logabsdet = -torch.log((alpha * (input_right_heights - input_left_heights)
                                + input_left_heights))
    else:
        alpha = (inputs - input_bin_locations) / input_bin_widths
        outputs = a * alpha.pow(2) + b * alpha + c
        outputs = torch.clamp(outputs, 0, 1)
        logabsdet = torch.log((alpha * (input_right_heights - input_left_heights)
                               + input_left_heights))

    if inverse:
        outputs = outputs * (right - left) + left
        logabsdet = logabsdet - np.log(top - bottom) + np.log(right - left)
    else:
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + np.log(top - bottom) - np.log(right - left)

    return outputs, logabsdet

def unconstrained_cubic_spline(inputs,
                               unnormalized_widths,
                               unnormalized_heights,
                               unnorm_derivatives_left,
                               unnorm_derivatives_right,
                               inverse=False,
                               tail_bound=1.,
                               min_bin_width=1e-3,
                               min_bin_height=1e-3,
                               quadratic_threshold=1e-3,
                               eps=1e-5):

    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)
        
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = cubic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnorm_derivatives_left=unnorm_derivatives_left[inside_interval_mask, :],
        unnorm_derivatives_right=unnorm_derivatives_right[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        quadratic_threshold=quadratic_threshold
    )

    return outputs, logabsdet

def cubic_spline(inputs,
                 unnormalized_widths,
                 unnormalized_heights,
                 unnorm_derivatives_left,
                 unnorm_derivatives_right,
                 inverse=False,
                 left=0., right=1., bottom=0., top=1.,
                 min_bin_width=1e-3,
                 min_bin_height=1e-3,
                 quadratic_threshold=1e-3,
                 eps=1e-5):
    """
    References:
    > Blinn, J. F. (2007). How to solve a cubic equation, part 5: Back to numerics. IEEE Computer
    Graphics and Applications, 27(3):78–89.
    """

    num_bins = unnormalized_widths.shape[-1]

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths[..., -1] = 1
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights

    cumheights = torch.cumsum(heights, dim=-1)
    cumheights[..., -1] = 1
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)

    slopes = heights / widths
    min_something_1 = torch.min(torch.abs(slopes[..., :-1]),
                                torch.abs(slopes[..., 1:]))
    min_something_2 = (
            0.5 * (widths[..., 1:] * slopes[..., :-1] + widths[..., :-1] * slopes[..., 1:])
            / (widths[..., :-1] + widths[..., 1:])
    )
    min_something = torch.min(min_something_1, min_something_2)

    derivatives_left = torch.sigmoid(unnorm_derivatives_left) * 3 * slopes[..., 0][..., None]
    derivatives_right = torch.sigmoid(unnorm_derivatives_right) * 3 * slopes[..., -1][..., None]

    derivatives = min_something * (torch.sign(slopes[..., :-1]) + torch.sign(slopes[..., 1:]))
    derivatives = torch.cat([derivatives_left,
                             derivatives,
                             derivatives_right], dim=-1)

    a = (derivatives[..., :-1] + derivatives[..., 1:] - 2 * slopes) / widths.pow(2)
    b = (3 * slopes - 2 * derivatives[..., :-1] - derivatives[..., 1:]) / widths
    c = derivatives[..., :-1]
    d = cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    inputs_a = a.gather(-1, bin_idx)[..., 0]
    inputs_b = b.gather(-1, bin_idx)[..., 0]
    inputs_c = c.gather(-1, bin_idx)[..., 0]
    inputs_d = d.gather(-1, bin_idx)[..., 0]

    input_left_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_right_cumwidths = cumwidths.gather(-1, bin_idx + 1)[..., 0]

    if inverse:
        # Modified coefficients for solving the cubic.
        inputs_b_ = (inputs_b / inputs_a) / 3.
        inputs_c_ = (inputs_c / inputs_a) / 3.
        inputs_d_ = (inputs_d - inputs) / inputs_a

        delta_1 = -inputs_b_.pow(2) + inputs_c_
        delta_2 = -inputs_c_ * inputs_b_ + inputs_d_
        delta_3 = inputs_b_ * inputs_d_ - inputs_c_.pow(2)

        discriminant = 4. * delta_1 * delta_3 - delta_2.pow(2)

        depressed_1 = -2. * inputs_b_ * delta_1 + delta_2
        depressed_2 = delta_1

        three_roots_mask = discriminant >= 0  # Discriminant == 0 might be a problem in practice.
        one_root_mask = discriminant < 0

        outputs = torch.zeros_like(inputs)

        # Deal with one root cases.

        p = cbrt((-depressed_1[one_root_mask] + torch.sqrt(-discriminant[one_root_mask])) / 2.)
        q = cbrt((-depressed_1[one_root_mask] - torch.sqrt(-discriminant[one_root_mask])) / 2.)

        outputs[one_root_mask] = ((p + q)
                                  - inputs_b_[one_root_mask]
                                  + input_left_cumwidths[one_root_mask])

        # Deal with three root cases.

        theta = torch.atan2(torch.sqrt(discriminant[three_roots_mask]), -depressed_1[three_roots_mask])
        theta /= 3.

        cubic_root_1 = torch.cos(theta)
        cubic_root_2 = torch.sin(theta)

        root_1 = cubic_root_1
        root_2 = -0.5 * cubic_root_1 - 0.5 * np.sqrt(3) * cubic_root_2
        root_3 = -0.5 * cubic_root_1 + 0.5 * np.sqrt(3) * cubic_root_2

        root_scale = 2 * torch.sqrt(-depressed_2[three_roots_mask])
        root_shift = (-inputs_b_[three_roots_mask] + input_left_cumwidths[three_roots_mask])

        root_1 = root_1 * root_scale + root_shift
        root_2 = root_2 * root_scale + root_shift
        root_3 = root_3 * root_scale + root_shift

        root1_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_1).float()
        root1_mask *= (root_1 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        root2_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_2).float()
        root2_mask *= (root_2 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        root3_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_3).float()
        root3_mask *= (root_3 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        roots = torch.stack([root_1, root_2, root_3], dim=-1)
        masks = torch.stack([root1_mask, root2_mask, root3_mask], dim=-1)
        mask_index = torch.argsort(masks, dim=-1, descending=True)[..., 0][..., None]
        outputs[three_roots_mask] = torch.gather(roots, dim=-1, index=mask_index).view(-1)

        # Deal with a -> 0 (almost quadratic) cases.

        quadratic_mask = inputs_a.abs() < quadratic_threshold
        a = inputs_b[quadratic_mask]
        b = inputs_c[quadratic_mask]
        c = (inputs_d[quadratic_mask] - inputs[quadratic_mask])
        alpha = (-b + torch.sqrt(b.pow(2) - 4*a*c)) / (2*a)
        outputs[quadratic_mask] = alpha + input_left_cumwidths[quadratic_mask]

        shifted_outputs = (outputs - input_left_cumwidths)
        logabsdet = -torch.log((3 * inputs_a * shifted_outputs.pow(2) +
                                2 * inputs_b * shifted_outputs +
                                inputs_c))
    else:
        shifted_inputs = (inputs - input_left_cumwidths)
        outputs = (inputs_a * shifted_inputs.pow(3) +
                   inputs_b * shifted_inputs.pow(2) +
                   inputs_c * shifted_inputs +
                   inputs_d)

        logabsdet = torch.log((3 * inputs_a * shifted_inputs.pow(2) +
                               2 * inputs_b * shifted_inputs +
                               inputs_c))

    if inverse:
        outputs = outputs * (right - left) + left
        logabsdet = logabsdet - np.log(top - bottom) + np.log(right - left)
    else:
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + np.log(top - bottom) - np.log(right - left)

    return outputs, logabsdet