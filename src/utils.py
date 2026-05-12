import torch
import torch.nn.functional as F

import numpy as np

from scipy.signal import savgol_coeffs

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

def savgol_smooth(curve, window, order):
    coeffs = torch.tensor(
        savgol_coeffs(window, order),
        dtype=curve.dtype,
        device=curve.device
    ).view(1, 1, -1)
    shape = curve.size()
    curve = curve.view(-1, 1, shape[-1])

    smoothed = F.conv1d(curve, coeffs, padding=window // 2)
    return smoothed.view(shape)


def interpolate(x, y, x_new, mode='linear', fill_value=torch.nan, device: str = 'cpu'):
    """
    Linear interpolation function in PyTorch with additional modes for out-of-bounds values.

    Parameters:
    - x: Known x-coordinates (1D tensor)
    - y: Known y-coordinates (1D tensor)
    - x_new: New x-coordinates to interpolate (1D tensor)
    - mode: Mode for handling out-of-bounds values ('linear', 'fixed', 'border')
    - fill_value: Value to fill in fixed mode
    - device: Device for calculation

    Returns:
    - y_new: Interpolated y-coordinates (1D tensor)
    """
    # Convert inputs to torch tensors if they are not already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32, device=device)
    if not isinstance(x_new, torch.Tensor):
        x_new = torch.tensor(x_new, dtype=torch.float32, device=device)
    
    # Sort the input tensors based on x values
    sorted_indices = torch.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    
    # Initialize the result tensor
    y_new = torch.zeros_like(x_new)
    
    # Find the indices in x where elements should be inserted to maintain order
    indices = torch.searchsorted(x, x_new, right=True)
    
    # Handle edge cases based on the specified mode
    within_bounds = (x_new >= x.min()) & (x_new <= x.max())
    if mode == 'linear':
        indices = torch.clamp(indices, 1, len(x) - 1)
        x0 = x[indices - 1]
        x1 = x[indices]
        y0 = y[indices - 1]
        y1 = y[indices]
        slope = (y1 - y0) / (x1 - x0)
        y_new = y0 + slope * (x_new - x0)
    elif mode == 'fixed':
        indices = torch.clamp(indices, 1, len(x) - 1)
        x0 = x[indices - 1]
        x1 = x[indices]
        y0 = y[indices - 1]
        y1 = y[indices]
        slope = (y1 - y0) / (x1 - x0)
        y_new[within_bounds] = y0[within_bounds] + slope[within_bounds] * (x_new[within_bounds] - x0[within_bounds])
        y_new[~within_bounds] = torch.tensor(fill_value, dtype=x.dtype)
    elif mode == 'border':
        indices = torch.clamp(indices, 1, len(x) - 1)
        x0 = x[indices - 1]
        x1 = x[indices]
        y0 = y[indices - 1]
        y1 = y[indices]
        slope = (y1 - y0) / (x1 - x0)
        y_new = y0 + slope * (x_new - x0)
        y_new[x_new < x.min()] = y[0]
        y_new[x_new > x.max()] = y[-1]
    
    return y_new, within_bounds


def calc_capacity(I, t):
    return np.cumsum(np.diff(t, prepend=0) * I) / 3600.


def calc_V_of_Q(V, Q, interp_dims=1000, Q_min=0.0, Q_max=2.2):
    Q_new = np.linspace(Q_min, Q_max, interp_dims)
    return interpolate(Q, V, Q_new, mode='border')


def calc_dV_over_dQ(V, Q):
    delta_V = V - np.roll(V, 1)
    delta_Q = Q - np.roll(Q, 1)
    delta_Q[abs(delta_Q) < 1e-6] = np.inf
    return delta_V / delta_Q


def diff_cycles(cycle_A, cycle_B, mask_A=None, mask_B=None):
    diff = cycle_A - cycle_B
    if mask_A is not None and mask_B is not None:
        diff[~(mask_A | mask_B)] = 0.
    return diff


def nanmean(tensor, dim):
    mask = ~tensor.isnan()
    tensor = tensor.masked_fill(~mask, 0.0)
    count = mask.sum(dim=dim, keepdim=True)
    sum_tensor = tensor.sum(dim=dim, keepdim=True)
    mean = sum_tensor / count
    return mean.squeeze(dim)


def nanvar(tensor, dim):
    mask = ~tensor.isnan()
    tensor = tensor.masked_fill(~mask, 0.0)
    count = mask.sum(dim=dim, keepdim=True)
    mean = tensor.sum(dim=dim, keepdim=True) / count
    sq_diff = (tensor - mean).pow(2).masked_fill(~mask, 0.0)
    var = sq_diff.sum(dim=dim, keepdims=True) / count
    return var.squeeze(dim)


def nanmin(tensor, dim):
    mask = tensor.isnan()
    tensor = tensor.masked_fill(mask, float('inf'))
    min_val, _ = tensor.min(dim=dim, keepdim=False)
    return min_val


def nanmax(tensor, dim):
    mask = tensor.isnan()
    tensor = tensor.masked_fill(mask, float('-inf'))
    max_val, _ = tensor.max(dim=dim, keepdim=False)
    return max_val


def nanskew(tensor, dim):
    mask = ~tensor.isnan()
    tensor = tensor.masked_fill(~mask, 0.0)
    count = mask.sum(dim=dim, keepdim=True)
    mean = tensor.sum(dim=dim, keepdim=True) / count
    diff = tensor - mean
    diff_pow_3 = diff.pow(3).masked_fill(~mask, 0.0)
    skewness = diff_pow_3.sum(dim=dim, keepdims=True) / count
    var = nanvar(tensor, dim).unsqueeze(dim)
    skewness = skewness / var.pow(1.5)
    return skewness.squeeze(dim)


def nankurtosis(tensor, dim):
    mask = ~tensor.isnan()
    tensor = tensor.masked_fill(~mask, 0.0)
    count = mask.sum(dim=dim, keepdim=True)
    mean = tensor.sum(dim=dim, keepdim=True) / count
    diff = tensor - mean
    diff_pow_4 = diff.pow(4).masked_fill(~mask, 0.0)
    kurtosis = diff_pow_4.sum(dim=dim, keepdims=True) / count
    var = nanvar(tensor, dim).unsqueeze(dim)
    kurtosis = kurtosis / var.pow(2)
    return kurtosis.squeeze(dim)



def interpolate_element(x_original, y_original, inter_num=1000, device: str = 'cpu'):

    np.random.seed(0)
    # if isinstance(x_original, list):
    #     x_original = x_original = np.array(x_original)
    #     y_original = y_original = np.array(y_original)
    # else:
    #     x_original.astype(float)
    #     y_original.astype(float)

    try:
        x_new = np.linspace(x_original.min(), x_original.max(), inter_num)

        interpolator = interp1d(x_original, y_original, kind='linear')  # You can use 'linear', 'cubic', 'quadratic', etc.
        y_new = interpolator(x_new)
        return torch.tensor(y_new, dtype=torch.float32, device=device)
    except:
        print( x_original)

    
    