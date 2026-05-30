# NextOnMenu

Early-signal radar for niche food ingredients. Diagnoses where an ingredient sits in
its trend lifecycle — it does **not** predict virality. Radar, not crystal ball.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline
```bash
python -m src.features   # fetch + build data/processed/features.csv
python -m src.train      # train + leave-one-ingredient-out eval, writes model.pkl
python app.py            # launch the Gradio demo
```

## How it works
1. **Fetch** — Google Trends data for ingredients that already went viral (`src/fetch.py`).
2. **Window** — for each, slice the *early-curve* window (positive), plus its flat
   pre-niche era and post-peak plateau (negatives) (`src/features.py`).
3. **Features** — growth (rate, acceleration, baseline, peak-not-hit) + entropy
   (geographic, temporal, 6-month delta).
4. **Train** — logistic regression, evaluated with leave-one-ingredient-out CV so the
   score reflects judging an *unseen* ingredient (`src/train.py`).
5. **Diagnose** — fetch an ingredient live, map model output to a lifecycle stage
   (`src/diagnose.py`), shown in a Gradio UI (`app.py`).

## Stages
- 😴 **No signal** — still noise
- 🌀 **Stage 1 — Entropy Drop** — interest organizing, watch this
- ⚡ **Stage 2 — Momentum** — growth accelerating, early-mover window
- 🔊 **Stage 3 — Resonance** — broad accelerating signal, mainstream incoming

## Notes
- pytrends may be rate-limited or blocked. Fetches are cached to `data/raw/`; if blocked,
  export CSVs manually from trends.google.com to the path named in the error and re-run.
  Cached files make re-runs incremental.
- Phase 1 is the logistic-regression baseline. LSTM/CNN are deferred to Phase 2.
