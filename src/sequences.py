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


def augment(values, rng, n=4, jitter=0.02, scale_range=0.1, max_shift=2):
    """Return `n` label-preserving variants of a normalized curve.

    Each variant applies small jitter (Gaussian noise), amplitude scaling, and a
    time-shift (edge-padded roll). All perturbations are small enough to preserve
    the curve's shape and therefore its label.
    """
    arr = np.asarray(values, dtype=float)
    variants = []
    for _ in range(n):
        v = arr.copy()
        v = v * (1.0 + rng.uniform(-scale_range, scale_range))   # amplitude scale
        v = v + rng.normal(0.0, jitter, size=v.shape)            # jitter
        shift = int(rng.integers(-max_shift, max_shift + 1))     # time-shift
        if shift != 0:
            v = np.roll(v, shift)
            if shift > 0:
                v[:shift] = v[shift]
            else:
                v[shift:] = v[shift - 1]
        variants.append(v)
    return variants
