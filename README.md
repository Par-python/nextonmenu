# NextOnMenu

An early-signal radar for niche food ingredients. It diagnoses where an ingredient sits in
its trend lifecycle. It does **not** predict virality. Radar, not crystal ball.

## Data provenance

Every model input is real Google Trends data, cached as CSVs in `data/raw/`. The headline
result — logistic regression, 0.66 leave-one-ingredient-out accuracy — uses **no synthetic
data at all** (`src/experiment.py`, `run_lr_loo`). The single place synthesis appears is
label-preserving augmentation (small jitter, amplitude scale, time-shift) used to train the
1D-CNN _comparison_ model (`src/sequences.py`). That augmentation is applied inside each
training fold only, never to the held-out ingredient (`src/experiment.py:48`), so it can't
leak — and the CNN did not beat the baseline, so it isn't responsible for the headline
number either. Reproduce the headline in five seconds:

```bash
python -c "from src.experiment import run_lr_loo; a,p,r=run_lr_loo(); print(f'LR LOO acc={a:.3f}')"
# -> LR LOO acc=0.660
```

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
3. **Features.** Growth (1-year rate, baseline, peak-not-hit) and entropy (geographic,
   temporal, 6-month delta). We compute the short-term growth rate and acceleration too, but
   the model drops them: all three growth features are correlated at r = 0.98 to 0.99, and
   keeping just the 1-year rate removes that redundancy (it nudges accuracy up and makes the
   coefficients interpretable).
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
| Logistic Regression | 0.66         | 0.55      | 0.80   |
| 1D-CNN (augmented)  | 0.62         | 0.52      | 0.70   |

**Finding:** the CNN did **not** beat the baseline (0.62 vs 0.66). On roughly 50 windows from
20 ingredients, even with augmentation, the deep model has too little data to gain an edge.
That is a genuine result, not a failure to hide: on small, noisy trend data, the simple
interpretable model holds up. We report it as-is, and we did not tune the CNN against the test
folds to manufacture a win. The likely path past the 0.70 target is more training ingredients
or richer features, not a fancier model. A learning curve in `notebooks/analysis.ipynb` backs
that up: accuracy flattens in the mid-0.60s well before 20 ingredients, which means the model
is signal-limited, not data-limited.

## Limitations

These numbers come from 20 ingredients, which is small, so read them as directional rather
than precise:

- **The 0.66 is mildly optimistic.** We iterated on the dataset and features while checking the
  same leave-one-ingredient-out holdout, so the model has implicitly fit to that particular set.
  A fresh batch of ingredients would give a cleaner estimate.
- **Small differences aren't significant.** The score wobbles by roughly plus or minus 0.02 to
  0.03 fold to fold, so 0.66 vs 0.64 vs the CNN's 0.62 are probably not statistically
  distinguishable. We can say the CNN didn't beat the baseline; we don't claim "0.66 beats 0.64"
  as a hard fact.
- **The labels are human judgment.** Which ingredients "went viral" and which months count as the
  "early curve" were our calls. Defensible and consistent, but not ground truth.

The process is sound (no leakage, real holdout, results reported as-is); the conclusions are
directional.

## Notes

- pytrends gets rate-limited or blocked fairly often. Fetches are cached to `data/raw/`, so
  re-runs are incremental. If you get blocked, set a valid `NID` browser cookie via the
  `TRENDS_NID` env var, or export CSVs manually from trends.google.com to the path named in
  the error and re-run.
- Phase 1 is the logistic-regression baseline (`src/train.py`). Phase 2 is the 1D-CNN
  experiment (`src/sequences.py`, `src/cnn.py`, `src/experiment.py`). It is an offline
  comparison and is intentionally **not** wired into the live demo, since it did not beat the
  baseline.
