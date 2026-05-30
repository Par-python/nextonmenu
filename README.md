# NextOnMenu

An early-signal radar for niche food ingredients. It diagnoses where an ingredient sits in
its trend lifecycle. It does **not** predict virality. Radar, not crystal ball.

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

1. **Fetch.** Google Trends data for ingredients that already went viral (`src/fetch.py`).
2. **Window.** For each one, slice the _early-curve_ window (the positive example), plus its
   flat pre-niche era and its post-peak plateau (the negatives) (`src/features.py`).
3. **Features.** Growth (rate, acceleration, baseline, peak-not-hit) and entropy
   (geographic, temporal, 6-month delta).
4. **Train.** Logistic regression, evaluated with leave-one-ingredient-out CV so the score
   reflects judging an _unseen_ ingredient rather than memorizing a known one (`src/train.py`).
5. **Diagnose.** Fetch an ingredient live, map the model output to a lifecycle stage
   (`src/diagnose.py`), shown in a Gradio UI (`app.py`).

## Stages

- **No signal.** Still noise.
- **Stage 1, Entropy Drop.** Interest is organizing, worth watching.
- **Stage 2, Momentum.** Growth is accelerating, early-mover window.
- **Stage 3, Resonance.** Broad accelerating signal, mainstream incoming.

## Phase 2: the 1D-CNN experiment

The question here was whether a small 1D-CNN reading the raw trend curve could beat the
logistic-regression baseline, judged under the _same_ leave-one-ingredient-out evaluation on
the same 20 ingredients. Augmentation (jitter, scaling, time-shift) was applied inside each
training fold only, never to the held-out ingredient, so the comparison stays leakage-free.
You can reproduce it with `python -m src.experiment`.

| Model               | LOO accuracy | precision | recall |
| ------------------- | ------------ | --------- | ------ |
| Logistic Regression | 0.64         | 0.54      | 0.75   |
| 1D-CNN (augmented)  | 0.62         | 0.52      | 0.70   |

**Finding:** the CNN did **not** beat the baseline (0.62 vs 0.64). On roughly 50 windows from
20 ingredients, even with augmentation, the deep model has too little data to gain an edge.
That is a genuine result, not a failure to hide: on small, noisy trend data, the simple
interpretable model holds up. We report it as-is, and we did not tune the CNN against the test
folds to manufacture a win. The likely path past the 0.70 target is more training ingredients
or richer features, not a fancier model. A learning curve in `notebooks/analysis.ipynb` backs
that up: accuracy flattens around 0.64 well before 20 ingredients, which means the model is
signal-limited, not data-limited.

## Notes

- pytrends gets rate-limited or blocked fairly often. Fetches are cached to `data/raw/`, so
  re-runs are incremental. If you get blocked, set a valid `NID` browser cookie via the
  `TRENDS_NID` env var, or export CSVs manually from trends.google.com to the path named in
  the error and re-run.
- Phase 1 is the logistic-regression baseline (`src/train.py`). Phase 2 is the 1D-CNN
  experiment (`src/sequences.py`, `src/cnn.py`, `src/experiment.py`). It is an offline
  comparison and is intentionally **not** wired into the live demo, since it did not beat the
  baseline.
