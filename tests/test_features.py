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


# --- sample entropy --------------------------------------------------------
def test_sample_entropy_higher_for_noisy_than_smooth():
    from src.features import sample_entropy
    smooth = list(np.linspace(0, 100, 100))          # smooth ramp
    import random; random.seed(0)
    noisy = [random.gauss(50, 30) for _ in range(100)]
    assert sample_entropy(noisy) > sample_entropy(smooth)


def test_sample_entropy_flat_series_is_finite():
    from src.features import sample_entropy
    assert np.isfinite(sample_entropy([7.0] * 50))


def test_sample_entropy_short_series_returns_zero():
    from src.features import sample_entropy
    assert sample_entropy([1.0, 2.0]) == 0.0


# --- non-breakout slicing --------------------------------------------------
def test_non_breakout_window_is_single_label_0(rising_then_peak_iot):
    from src.features import slice_non_breakout_window
    out = slice_non_breakout_window(rising_then_peak_iot)
    assert len(out) == 1
    assert out[0]["window_type"] == "non_breakout"
    assert out[0]["label"] == 0


def test_non_breakout_window_ends_before_steepest_rise(rising_then_peak_iot):
    from src.features import slice_non_breakout_window
    out = slice_non_breakout_window(rising_then_peak_iot)
    win = out[0]["data"]
    assert len(win) > 0
    assert win.index[-1] < rising_then_peak_iot.index[-1]


def test_non_breakout_flat_curve_yields_nothing(flat_iot):
    from src.features import slice_non_breakout_window
    assert slice_non_breakout_window(flat_iot) == []


def test_non_breakout_flat_noise_midseries_yields_nothing():
    # A purely flat noisy curve (no real trend) whose noisy max-slope lands mid-series.
    # Must still yield NO window — a flat term must not contribute a label-0 "early curve".
    import pandas as pd
    from src.features import slice_non_breakout_window
    n = 520
    idx = pd.date_range("2010-01-03", periods=n, freq="W")
    rng = np.random.default_rng(7)
    flat = np.clip(np.full(n, 4.0) + rng.normal(0, 1.0, n), 0, 100)
    assert slice_non_breakout_window(pd.DataFrame({"value": flat}, index=idx)) == []


def test_non_breakout_modest_rise_yields_window():
    # The real use case: a "rose then fizzled" curve (5 -> 40, never breaks out) must
    # produce exactly one label-0 window of the configured length, ending before its rise.
    import pandas as pd
    from src.features import slice_non_breakout_window, _months_to_rows, find_steepest_rise_pos
    from src.config import WINDOW_MONTHS, EARLY_GAP_MONTHS
    n = 520
    idx = pd.date_range("2010-01-03", periods=n, freq="W")
    vals = np.concatenate([np.full(150, 5.0), np.linspace(5, 40, 200), np.full(170, 38.0)])
    df = pd.DataFrame({"value": vals}, index=idx)
    out = slice_non_breakout_window(df)
    assert len(out) == 1 and out[0]["label"] == 0
    win = out[0]["data"]
    assert len(win) == _months_to_rows(WINDOW_MONTHS, idx)            # full-length window
    rise_pos = find_steepest_rise_pos(df)
    assert win.index[-1] < idx[rise_pos]                             # ends before the rise


def test_non_breakout_short_history_before_rise_yields_nothing():
    # A curve that rises almost immediately has no room for a full window before the
    # rise (start < 0) and must yield [] via the geometry guard, not the flatness guard.
    import pandas as pd
    from src.features import slice_non_breakout_window
    n = 520
    idx = pd.date_range("2010-01-03", periods=n, freq="W")
    vals = np.concatenate([np.linspace(5, 60, 30), np.full(n - 30, 58.0)])  # rises at the start
    df = pd.DataFrame({"value": vals}, index=idx)
    assert slice_non_breakout_window(df) == []
