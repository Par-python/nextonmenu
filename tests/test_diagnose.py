from src.diagnose import classify_stage


def test_flat_low_prob_is_no_signal():
    feats = {"growth_rate_6m": 0.0, "acceleration": -0.1,
             "entropy_delta_6m": 0.0, "peak_not_yet_hit": 1}
    stage, emoji, _ = classify_stage(prob=0.2, feats=feats)
    assert stage == "No signal" and emoji == "😴"


def test_entropy_drop_only_is_stage1():
    feats = {"growth_rate_6m": 0.2, "acceleration": -0.05,
             "entropy_delta_6m": -0.5, "peak_not_yet_hit": 1}
    stage, emoji, _ = classify_stage(prob=0.55, feats=feats)
    assert stage == "Stage 1 — Entropy Drop" and emoji == "🌀"


def test_positive_acceleration_is_stage2():
    feats = {"growth_rate_6m": 0.4, "acceleration": 0.3,
             "entropy_delta_6m": -0.2, "peak_not_yet_hit": 1}
    stage, emoji, _ = classify_stage(prob=0.6, feats=feats)
    assert stage == "Stage 2 — Momentum" and emoji == "⚡"


def test_high_prob_broad_is_stage3():
    feats = {"growth_rate_6m": 0.5, "acceleration": 0.4,
             "entropy_delta_6m": -0.3, "peak_not_yet_hit": 1}
    stage, emoji, _ = classify_stage(prob=0.85, feats=feats)
    assert stage == "Stage 3 — Resonance" and emoji == "🔊"
