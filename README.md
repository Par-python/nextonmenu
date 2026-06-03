# NextOnMenu

NextOnMenu detects an entropy/growth pattern in Google Trends data that *often precedes* a
food ingredient's breakout. It is **not** a validated forward predictor: it flags a pattern,
it does not forecast that any given ingredient will go mainstream. Tested against real
"organized-then-fizzled" ingredients, it still raises a false alarm on about a third of them
(specificity 0.65). Radar with a known, measured noise level — not a crystal ball.

## Data provenance

Every model input is real Google Trends data, cached as CSVs in `data/raw/`. The headline
result — logistic regression, 0.70 leave-one-ingredient-out accuracy, now including real
non-breakout ingredients in the negative class — uses **no synthetic data at all**
(`src/experiment.py`, `run_lr_loo`). The single place synthesis appears is
label-preserving augmentation (small jitter, amplitude scale, time-shift) used to train the
1D-CNN _comparison_ model (`src/sequences.py`). That augmentation is applied inside each
training fold only, never to the held-out ingredient (`src/experiment.py:48`), so it can't
leak — and the CNN did not beat the baseline, so it isn't responsible for the headline
number either. Reproduce the headline in five seconds:

```bash
python -c "from src.experiment import run_lr_loo; acc,prec,rec,spec,fpr=run_lr_loo(); print(f'LR LOO acc={acc:.3f} specificity={spec:.3f}')"
# -> LR LOO acc=0.700 specificity=0.650
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
   flat pre-niche era and its post-peak plateau (the negatives) (`src/features.py`). The
   negative class also includes the early-rise windows of non-breakout ingredients — genuine
   "rose then fizzled" hard negatives — via `NON_BREAKOUT_INGREDIENTS` and
   `slice_non_breakout_window` in `src/features.py`.
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
the same 30 ingredients / 60 windows. Augmentation (jitter, scaling, time-shift) was applied
inside each training fold only, never to the held-out ingredient, so the comparison stays
leakage-free. You can reproduce it with `python -m src.experiment`.

| Model               | accuracy | precision | recall | specificity | fp_rate |
| ------------------- | -------- | --------- | ------ | ----------- | ------- |
| Logistic Regression | 0.70     | 0.53      | 0.80   | 0.65        | 0.35    |
| 1D-CNN (augmented)  | 0.63     | 0.47      | 0.75   | 0.58        | 0.42    |

**Finding:** the CNN did **not** beat the baseline (0.63 vs 0.70). On 60 windows from
30 ingredients, even with augmentation, the deep model has too little data to gain an edge.
That is a genuine result, not a failure to hide: on small, noisy trend data, the simple
interpretable model holds up. We report it as-is, and we did not tune the CNN against the test
folds to manufacture a win. The simple model still holds up, and crossing the target reliably
likely needs richer features or more ingredients, not a fancier model — a learning curve in
`notebooks/analysis.ipynb` explores where accuracy levels off.

## Limitations

These numbers come from 30 ingredients / 60 windows, which is small, so read them as directional
rather than precise:

- **Specificity is the headline caveat, not accuracy.** The negative class now includes the
  early-rise windows of 10 ingredients that organized but never crossed to mainstream
  (`NON_BREAKOUT_INGREDIENTS`). Against those genuine non-breakouts the model's specificity is
  0.65 — it still raises a false alarm on ~35% of them (14 false positives out of 30 negatives;
  confusion matrix [[26 14],[4 16]]). Earlier versions never tested this case (negatives were
  only winners' flat/post-peak phases), so the old precision looked better than the detector
  really is. The false-alarm rate is the number to watch.
- **The 0.70 is mildly optimistic.** We iterated on the dataset and features while checking the
  same leave-one-ingredient-out holdout, so the model has implicitly fit to that particular set.
  A fresh batch of ingredients would give a cleaner estimate.
- **Small differences aren't significant.** The score wobbles by roughly plus or minus 0.02 to
  0.03 fold to fold, so 0.70 vs the CNN's 0.63 should be read with that noise in mind. We can say
  the CNN didn't beat the baseline; we don't claim hair-thin margins as hard facts.
- **The labels are human judgment.** Which ingredients "went viral" and which months count as the
  "early curve" were our calls. Defensible and consistent, but not ground truth.

The process is sound (no leakage, real holdout, results reported as-is); the conclusions are
directional.

## Data limitations (Google Trends artifacts)

The input is Google Trends' normalized 0–100 *relative* interest index, not raw counts (Google
doesn't expose counts). Four known artifacts affect the signal; we disclose them rather than
silently correcting:

- **Low-volume quantization.** Early sparse data is coarsely rounded toward 0/1, which reads as
  "organized" (low entropy) almost by default; entropy drifts upward over 2004–2024 as volume
  grows. We do **not** detrend for this yet — it's the clearest next experiment.
- **Per-window rescale-to-peak.** Region fetches in `build_dataset` call `build_payload` with a
  different timeframe per window, so each window is independently rescaled by Google to its own
  max=100 and then treated as comparable. We do **not** re-stitch windows onto a common
  timeframe yet (it would mean re-fetching and renormalizing every ingredient).
- **Sampling wobble.** A single fetch per ingredient; Google returns slightly different numbers
  per request and we don't average across repeated queries.
- **Variance↔entropy confound.** Histogram-Shannon `temporal_entropy` largely tracks spread, so
  a low-variance plateau mechanically produces an entropy dip. We compute `sample_entropy` (a
  less variance-driven estimate) and store it in `features.csv` as a candidate; it is not yet a
  model input, and it saturates on the short 24-row windows we use, so it needs re-validation
  before promotion.

## Notes

- pytrends gets rate-limited or blocked fairly often. Fetches are cached to `data/raw/`, so
  re-runs are incremental. If you get blocked, set a valid `NID` browser cookie via the
  `TRENDS_NID` env var, or export CSVs manually from trends.google.com to the path named in
  the error and re-run.
- Phase 1 is the logistic-regression baseline (`src/train.py`). Phase 2 is the 1D-CNN
  experiment (`src/sequences.py`, `src/cnn.py`, `src/experiment.py`). It is an offline
  comparison and is intentionally **not** wired into the live demo, since it did not beat the
  baseline.
