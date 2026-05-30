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


from src.sequences import normalize


def test_normalize_range_is_0_1():
    arr = np.array([5.0, 10.0, 20.0, 100.0])
    out = normalize(arr)
    assert out.min() == 0.0 and out.max() == 1.0


def test_normalize_preserves_order():
    arr = np.array([3.0, 1.0, 2.0])
    out = normalize(arr)
    assert out[1] < out[2] < out[0]  # ranking unchanged


def test_normalize_flat_curve_does_not_divide_by_zero():
    arr = np.array([7.0, 7.0, 7.0])
    out = normalize(arr)
    assert np.all(np.isfinite(out))


from src.sequences import augment


def test_augment_returns_requested_count():
    rng = np.random.default_rng(0)
    arr = np.linspace(0, 1, 24)
    variants = augment(arr, rng, n=5)
    assert len(variants) == 5


def test_augment_preserves_length():
    rng = np.random.default_rng(0)
    arr = np.linspace(0, 1, 24)
    for v in augment(arr, rng, n=3):
        assert len(v) == 24


def test_augment_changes_values_but_keeps_shape():
    rng = np.random.default_rng(0)
    arr = np.linspace(0, 1, 24)  # clearly rising
    variants = augment(arr, rng, n=5)
    assert any(not np.allclose(v, arr) for v in variants)
    for v in variants:
        assert v[12:].mean() > v[:12].mean()
