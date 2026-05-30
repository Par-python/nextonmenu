"""Raw trend windows -> fixed-length, normalized curves for the 1D-CNN."""
import numpy as np


def resample(values, length=24):
    """Linearly interpolate a 1D series to exactly `length` evenly-spaced points."""
    arr = np.asarray(values, dtype=float)
    if arr.size == length:
        return arr.copy()
    src_x = np.linspace(0.0, 1.0, num=arr.size)
    dst_x = np.linspace(0.0, 1.0, num=length)
    return np.interp(dst_x, src_x, arr)


def normalize(values):
    """Min-max scale a single curve to [0, 1]. Flat curves map to all-zeros."""
    arr = np.asarray(values, dtype=float)
    lo = arr.min()
    rng = arr.max() - lo
    if rng < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / rng
