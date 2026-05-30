"""Diagnose an ingredient's lifecycle stage (not a prediction)."""
import pickle

import matplotlib
# Use the non-interactive Agg backend for headless scripts and the Gradio app, but
# DON'T override an interactive backend already chosen by a notebook (e.g. the inline
# backend) — doing so would re-trigger "FigureCanvasAgg is non-interactive" warnings.
if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (PROB_SIGNAL_MIN, GROWTH_FLAT_MAX, ACCEL_MOMENTUM_MIN,
                        PROB_RESONANCE_MIN, MODEL_PATH, DEMO_TIMEFRAME)
from src.features import (slice_windows, extract_features, geographic_entropy,
                         FEATURE_COLUMNS)


def classify_stage(prob, feats):
    """Map model probability + feature thresholds to a lifecycle stage.

    Returns (stage_label, emoji, rationale). Order matters: most-advanced first.
    """
    growth = feats["growth_rate_6m"]
    accel = feats["acceleration"]
    entropy_delta = feats["entropy_delta_6m"]

    # The model probability is the gatekeeper: if the curve doesn't look like an
    # early-viral curve, a raw acceleration/entropy blip must NOT earn a stage. This
    # prevents a noisy spike (low match) from being mislabeled "Momentum".
    if prob < PROB_SIGNAL_MIN:
        return ("No signal", "",
                f"Doesn't match an early-viral curve (match {prob:.0%}) — still noise.")

    if prob >= PROB_RESONANCE_MIN and accel > ACCEL_MOMENTUM_MIN and feats["peak_not_yet_hit"]:
        return ("Stage 3 — Resonance", "",
                f"High match ({prob:.0%}) with accelerating, pre-peak signal.")

    if accel > ACCEL_MOMENTUM_MIN and growth > GROWTH_FLAT_MAX:
        return ("Stage 2 — Momentum", "",
                f"Match {prob:.0%} and growth accelerating (accel={accel:+.2f}); early-mover window.")

    if entropy_delta < 0:
        return ("Stage 1 — Entropy Drop", "",
                f"Match {prob:.0%}, entropy falling ({entropy_delta:+.2f}) while momentum still flat — watch this.")

    return ("Stage 1 — Entropy Drop", "",
            f"Early, ambiguous signal (match {prob:.0%}); worth watching.")


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _trend_chart(iot_df, ingredient):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(iot_df.index, iot_df["value"], color="#2b8a3e")
    ax.set_title(f"Google Trends — {ingredient}")
    ax.set_ylabel("interest")
    fig.tight_layout()
    return fig


def diagnose(ingredient):
    """Fetch recent data for `ingredient`, extract features, return stage + chart."""
    from src.fetch import fetch_interest_over_time, fetch_interest_by_region

    model = _load_model()
    iot = fetch_interest_over_time(ingredient, timeframe=DEMO_TIMEFRAME, tag="demo")

    # For live diagnosis, treat the whole recent window as the feature window.
    try:
        region = fetch_interest_by_region(ingredient, timeframe=DEMO_TIMEFRAME, tag="demo")
        geo = geographic_entropy(region)
    except Exception:
        geo = float("nan")

    feats = extract_features(iot, geo_entropy=geo)
    # Impute any NaN feature with the training-time median stored on the model, so
    # inference handles rate-limited region fetches the same way training did.
    medians = model.get("medians", {})
    for c in FEATURE_COLUMNS:
        if feats[c] != feats[c]:  # NaN check
            feats[c] = medians.get(c, 0.0)

    X = [[feats[c] for c in FEATURE_COLUMNS]]
    Xs = model["scaler"].transform(X)
    prob = float(model["clf"].predict_proba(Xs)[0][1])

    stage, _emoji, rationale = classify_stage(prob, feats)
    chart = _trend_chart(iot, ingredient)
    headline = f"{ingredient.title()} — {stage}"
    return headline, rationale, prob, chart
