"""Window slicing and feature extraction for NextOnMenu."""
import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy

from src.config import (WINDOW_MONTHS, EARLY_GAP_MONTHS, VIRAL_INGREDIENTS,
                        FEATURES_PATH, PROCESSED_DIR, FULL_TIMEFRAME)

WEEKS_PER_MONTH = 4.345


def _rows_per_month(index):
    """Infer how many rows correspond to one month from a datetime index.

    pytrends returns monthly data for long (multi-year) timeframes and weekly
    data for short ones, so window math must adapt to the actual cadence rather
    than assume a fixed step.
    """
    if len(index) < 2:
        return 1
    median_days = pd.Series(index).diff().dropna().dt.days.median()
    if median_days is None or median_days <= 0:
        return 1
    return max(1, int(round(30.44 / median_days)))


def _months_to_rows(months, index):
    return max(1, int(round(months * _rows_per_month(index))))


FEATURE_COLUMNS = [
    "growth_rate_6m", "growth_rate_1y", "acceleration",
    "peak_not_yet_hit", "search_baseline",
    "geographic_entropy", "temporal_entropy", "entropy_delta_6m",
]


# --------------------------------------------------------------------------- #
# Entropy
# --------------------------------------------------------------------------- #
def geographic_entropy(region_df):
    """Shannon entropy (nats) of per-region interest distribution."""
    values = region_df["interest"].to_numpy(dtype=float) + 1e-9
    probs = values / values.sum()
    return float(shannon_entropy(probs))


def temporal_entropy(values, bins=10):
    """Entropy of week-over-week changes. Higher = noisier/less consistent."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    diffs = np.diff(arr)
    hist, _ = np.histogram(diffs, bins=bins)
    hist = hist + 1e-9
    probs = hist / hist.sum()
    return float(shannon_entropy(probs))


# --------------------------------------------------------------------------- #
# Growth
# --------------------------------------------------------------------------- #
def _pct_change(series, weeks_back):
    """% change from `weeks_back` ago to the last value, robust to small baselines."""
    if len(series) <= weeks_back:
        return 0.0
    old = float(series.iloc[-weeks_back - 1])
    new = float(series.iloc[-1])
    return (new - old) / (old + 1.0)  # +1 avoids divide-by-near-zero blowups


def growth_features(window_df):
    """Growth features for a single trend window. Expects a 'value' column."""
    s = window_df["value"].astype(float)
    w6 = _months_to_rows(6, window_df.index)
    w12 = _months_to_rows(12, window_df.index)

    growth_6m = _pct_change(s, w6)
    growth_1y = _pct_change(s, w12)

    # acceleration: is recent 6m growth faster than the prior 6m growth?
    prior = s.iloc[: max(len(s) - w6, 1)]
    prior_growth_6m = _pct_change(prior, w6) if len(prior) > w6 else 0.0
    acceleration = growth_6m - prior_growth_6m

    baseline = float(s.iloc[:w6].mean())
    peak_not_yet_hit = 1 if float(s.iloc[-1]) < float(s.max()) else 0

    return {
        "growth_rate_6m": growth_6m,
        "growth_rate_1y": growth_1y,
        "acceleration": acceleration,
        "peak_not_yet_hit": peak_not_yet_hit,
        "search_baseline": baseline,
    }


# --------------------------------------------------------------------------- #
# Window slicing
# --------------------------------------------------------------------------- #
def find_peak_pos(iot_df):
    """Row index of the trend peak, found on a SMOOTHED curve.

    Smoothing prevents a lone early spike from masquerading as the true peak
    (e.g. ube's spurious 2009 blip). This is the peak `slice_windows` builds its
    windows around — use it anywhere a "peak" must match the model's behavior.
    """
    win = _months_to_rows(WINDOW_MONTHS, iot_df.index)
    smooth = iot_df["value"].rolling(window=max(3, win // 4), min_periods=1,
                                     center=True).mean().to_numpy()
    return int(smooth.argmax())


def slice_windows(iot_df):
    """Cut up to 3 labeled windows around the all-time peak.

    Returns list of dicts: {window_type, label, data (DataFrame), start, end}.
    `iot_df` must have a 'value' column and a datetime index.
    """
    win = _months_to_rows(WINDOW_MONTHS, iot_df.index)
    gap = _months_to_rows(EARLY_GAP_MONTHS, iot_df.index)
    peak_pos = find_peak_pos(iot_df)
    out = []

    # early_curve: 24m window ending `gap` weeks before the peak
    early_end = peak_pos - gap
    early_start = early_end - win
    if early_start >= 0:
        d = iot_df.iloc[early_start:early_end]
        out.append({"window_type": "early_curve", "label": 1, "data": d,
                    "start": d.index[0], "end": d.index[-1]})

    # pre_niche: earliest 24m window (the flat era), only if it ends well before rise
    if early_start - win >= 0:
        d = iot_df.iloc[0:win]
        out.append({"window_type": "pre_niche", "label": 0, "data": d,
                    "start": d.index[0], "end": d.index[-1]})

    # post_peak: 24m window starting at the peak
    post_end = min(peak_pos + win, len(iot_df))
    if post_end - peak_pos >= win // 2:  # need at least half a window of plateau
        d = iot_df.iloc[peak_pos:post_end]
        out.append({"window_type": "post_peak", "label": 0, "data": d,
                    "start": d.index[0], "end": d.index[-1]})

    return out


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #
def extract_features(window_df, geo_entropy):
    """Full feature row for one window. `geo_entropy` precomputed by caller."""
    s = window_df["value"].astype(float)
    feats = growth_features(window_df)

    temporal = temporal_entropy(s.to_numpy())
    w6 = _months_to_rows(6, window_df.index)
    recent = s.iloc[-w6:].to_numpy() if len(s) > w6 else s.to_numpy()
    earlier = s.iloc[:-w6].to_numpy() if len(s) > w6 else s.to_numpy()
    entropy_delta_6m = temporal_entropy(recent) - temporal_entropy(earlier)

    feats["geographic_entropy"] = float(geo_entropy)
    feats["temporal_entropy"] = temporal
    feats["entropy_delta_6m"] = entropy_delta_6m
    return feats


# --------------------------------------------------------------------------- #
# Dataset orchestration
# --------------------------------------------------------------------------- #
def _window_tag(window_type):
    return {"early_curve": "early", "pre_niche": "pre", "post_peak": "post"}[window_type]


def build_dataset(ingredients=None):
    """Fetch each viral ingredient, slice windows, extract features -> features.csv."""
    from src.fetch import fetch_interest_over_time, fetch_interest_by_region

    ingredients = ingredients or VIRAL_INGREDIENTS
    rows = []
    for ing in ingredients:
        print(f"Processing {ing} ...")
        iot = fetch_interest_over_time(ing, timeframe=FULL_TIMEFRAME, tag="full")
        for w in slice_windows(iot):
            tag = _window_tag(w["window_type"])
            tf = f"{w['start'].date()} {w['end'].date()}"
            try:
                region = fetch_interest_by_region(ing, timeframe=tf, tag=tag)
                geo = geographic_entropy(region)
            except Exception as e:
                print(f"  region fetch failed for {ing}/{tag}: {e}; geo=NaN")
                geo = float("nan")
            feats = extract_features(w["data"], geo_entropy=geo)
            feats.update({"ingredient": ing, "window_type": w["window_type"],
                          "label": w["label"]})
            rows.append(feats)

    df = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"Wrote {len(df)} rows -> {FEATURES_PATH}")
    return df


if __name__ == "__main__":
    build_dataset()
