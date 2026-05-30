import numpy as np
from src.sequences import resample


def test_resample_returns_exact_length():
    arr = np.arange(50, dtype=float)
    out = resample(arr, length=24)
    assert len(out) == 24


def test_resample_short_input_upsamples():
    arr = np.array([0.0, 10.0], dtype=float)
    out = resample(arr, length=24)
    assert len(out) == 24
    assert out[0] == 0.0 and abs(out[-1] - 10.0) < 1e-9  # endpoints preserved


def test_resample_preserves_monotonic_trend():
    arr = np.linspace(0, 100, 40)
    out = resample(arr, length=24)
    assert np.all(np.diff(out) >= -1e-9)  # still non-decreasing
