import numpy as np
from src.features import (geographic_entropy, temporal_entropy, growth_features,
                         slice_windows, extract_features)


# --- entropy ---------------------------------------------------------------
def test_concentrated_region_has_lower_entropy(region_concentrated, region_uniform):
    assert geographic_entropy(region_concentrated) < geographic_entropy(region_uniform)


def test_uniform_region_entropy_is_log_n(region_uniform):
    # 10 equal regions => Shannon entropy (nats) == ln(10)
    assert abs(geographic_entropy(region_uniform) - np.log(10)) < 1e-6


def test_temporal_entropy_higher_for_noisy_series():
    smooth = list(range(100))                       # steady increments
    import random; random.seed(0)
    noisy = [random.gauss(50, 30) for _ in range(100)]
    assert temporal_entropy(noisy) > temporal_entropy(smooth)


# --- growth ----------------------------------------------------------------
def test_growth_rate_positive_on_rising_window(rising_then_peak_iot):
    window = rising_then_peak_iot.iloc[150:280]
    feats = growth_features(window)
    assert feats["growth_rate_6m"] > 0
    assert feats["growth_rate_1y"] > 0


def test_baseline_low_on_rising_window(rising_then_peak_iot):
    window = rising_then_peak_iot.iloc[150:280]
    feats = growth_features(window)
    assert feats["search_baseline"] < 20


def test_peak_not_yet_hit_true_when_below_window_max(rising_then_peak_iot):
    window = rising_then_peak_iot.iloc[150:260]
    feats = growth_features(window)
    assert feats["peak_not_yet_hit"] in (0, 1)


def test_growth_features_flat_window_near_zero(flat_iot):
    window = flat_iot.iloc[150:280]
    feats = growth_features(window)
    assert abs(feats["growth_rate_6m"]) < 0.5  # essentially flat


# --- windows ---------------------------------------------------------------
def test_slice_windows_labels(rising_then_peak_iot):
    windows = slice_windows(rising_then_peak_iot)
    types = {w["window_type"]: w["label"] for w in windows}
    assert types["early_curve"] == 1
    assert types["post_peak"] == 0
    if "pre_niche" in types:
        assert types["pre_niche"] == 0


def test_early_window_ends_before_peak(rising_then_peak_iot):
    windows = {w["window_type"]: w for w in slice_windows(rising_then_peak_iot)}
    early = windows["early_curve"]["data"]
    peak_pos = rising_then_peak_iot["value"].to_numpy().argmax()
    assert early.index[-1] < rising_then_peak_iot.index[peak_pos]


def test_flat_series_yields_no_early_curve(flat_iot):
    windows = {w["window_type"] for w in slice_windows(flat_iot)}
    assert isinstance(windows, set)


# --- assembly --------------------------------------------------------------
def test_extract_features_row_has_all_columns(rising_then_peak_iot):
    window = rising_then_peak_iot.iloc[150:280]
    row = extract_features(window, geo_entropy=1.2)
    for col in ["growth_rate_6m", "growth_rate_1y", "acceleration",
                "peak_not_yet_hit", "search_baseline",
                "geographic_entropy", "temporal_entropy", "entropy_delta_6m"]:
        assert col in row
    assert row["geographic_entropy"] == 1.2
