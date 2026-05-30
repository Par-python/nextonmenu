"""Shared constants for the NextOnMenu pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_PATH = PROCESSED_DIR / "model.pkl"
FEATURES_PATH = PROCESSED_DIR / "features.csv"

# Ingredients that already went niche -> mainstream (positive source).
# Expanded with two "waves" of confirmed crossovers for more training signal.
# Any ingredient whose fetched curve lacks a real niche->mainstream rise should be
# pruned (it would be a mislabeled positive).
VIRAL_INGREDIENTS = [
    # original set
    "matcha", "ube", "tahini", "kimchi", "miso",
    "boba", "yuzu", "sriracha", "turmeric", "gochujang",
    # Asian-pantry wave
    "togarashi", "furikake", "dashi", "ponzu", "mochi",
    # global-staple wave
    "harissa", "tamari", "kefir", "quinoa", "kombucha",
    # Dropped after vetting: "taro" (already mainstream in 2004, baseline ~52 — not
    # a niche->mainstream crossover) and "acai" (early-2009 fad that crashed, its
    # post-peak shape muddied the early-curve signal). Both slightly hurt LOO acc.
]

# Rising-but-not-mainstream. Demo/inference ONLY — never used as training labels.
NICHE_INGREDIENTS = [
    "pandan", "calamansi", "chamoy", "butterfly pea flower",
    "black sesame", "koji",
]

FULL_TIMEFRAME = "2004-01-01 2024-01-01"  # training: locate true peak
DEMO_TIMEFRAME = "today 5-y"              # live demo: recent window

WINDOW_MONTHS = 24            # length of each labeled window
EARLY_GAP_MONTHS = 6          # early-curve window ends this many months before peak

# Stage-mapping thresholds (tunable). Used by diagnose.py.
PROB_SIGNAL_MIN = 0.50        # below this + flat growth => "no signal"
GROWTH_FLAT_MAX = 0.10        # <10% 6m growth counts as flat
ACCEL_MOMENTUM_MIN = 0.0      # positive acceleration => momentum
PROB_RESONANCE_MIN = 0.70     # high prob for Stage 3
