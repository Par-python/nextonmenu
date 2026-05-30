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
        # Flat curve: intentionally map to all-zeros (a valid CNN input, avoids div-by-zero)
        return np.zeros_like(arr)
    return (arr - lo) / rng


def augment(values, rng, n=4, jitter=0.02, scale_range=0.1, max_shift=2):
    """Return `n` label-preserving variants of a normalized curve.

    Each variant applies small jitter (Gaussian noise), amplitude scaling, and a
    time-shift (edge-padded roll). All perturbations are small enough to preserve
    the curve's shape and therefore its label.

    Args:
        values: 1-D array-like of floats. Assumed non-empty and already normalized.
        rng: a numpy Generator (e.g. np.random.default_rng(seed)) for reproducibility.
        n: number of augmented variants to produce.
        jitter: std-dev of Gaussian noise added per sample.
        scale_range: max fractional amplitude perturbation (±).
        max_shift: maximum absolute time-shift in samples (edge-padded).
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


SEQ_LENGTH = 24


def build_sequence_dataset(ingredients=None, length=SEQ_LENGTH):
    """Return (X, y, groups): X is (n, length) normalized curves, y labels,
    groups ingredient names. Uses the SAME windows as the LR dataset.

    Augmentation is intentionally NOT applied here — it is applied per-fold inside
    the experiment to avoid leaking augmented copies of a held-out ingredient.
    """
    from src.config import VIRAL_INGREDIENTS, FULL_TIMEFRAME
    from src.fetch import fetch_interest_over_time
    from src.features import slice_windows

    ingredients = ingredients or VIRAL_INGREDIENTS
    X, y, groups = [], [], []
    for ing in ingredients:
        iot = fetch_interest_over_time(ing, timeframe=FULL_TIMEFRAME, tag="full")
        for w in slice_windows(iot):
            curve = normalize(resample(w["data"]["value"].to_numpy(), length))
            X.append(curve)
            y.append(w["label"])
            groups.append(ing)
    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(groups)
