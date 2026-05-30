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
- **No signal** — still noise
- **Stage 1 — Entropy Drop** — interest organizing, watch this
- **Stage 2 — Momentum** — growth accelerating, early-mover window
- **Stage 3 — Resonance** — broad accelerating signal, mainstream incoming

## Phase 2 — 1D-CNN experiment

We tested whether a small 1D-CNN reading the raw trend curve beats the
logistic-regression baseline, under the *same* leave-one-ingredient-out evaluation on
the same 20 ingredients. Augmentation (jitter / scaling / time-shift) was applied inside
each training fold only — never to the held-out ingredient — so the comparison is
leakage-free. Run it with `python -m src.experiment`.

| Model | LOO accuracy | precision | recall |
|-------|-------------|-----------|--------|
| Logistic Regression | 0.64 | 0.54 | 0.75 |
| 1D-CNN (augmented)  | 0.62 | 0.52 | 0.70 |

**Finding:** the CNN did **not** beat the baseline (0.62 vs 0.64). On ~50 windows from
20 ingredients, even with augmentation, the deep model has too little data to gain an
edge — confirming that simple, interpretable models hold up well on small, noisy trend
data. This is a real result, reported as-is: we did not tune the CNN against the test
folds to manufacture a win. More training ingredients, not a fancier model, is the
likely path past the 0.70 target.

## Notes
- pytrends may be rate-limited or blocked. Fetches are cached to `data/raw/`; if blocked,
  set a valid `NID` browser cookie via the `TRENDS_NID` env var, or export CSVs manually
  from trends.google.com to the path named in the error and re-run. Cached files make
  re-runs incremental.
- Phase 1 is the logistic-regression baseline (`src/train.py`). Phase 2 is the 1D-CNN
  experiment (`src/sequences.py`, `src/cnn.py`, `src/experiment.py`); it is an offline
  comparison and is intentionally **not** wired into the live demo, since it did not beat
  the baseline.
