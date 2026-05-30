# NextOnMenu — Phase 2 Design (1D-CNN Honest Experiment)

**Date:** 2026-05-30
**Status:** Approved for planning
**Scope:** Phase 2 — a single 1D-CNN trained on raw curves, compared head-to-head
against the Phase 1 logistic-regression baseline under identical leave-one-ingredient-out
evaluation. No LSTM, no transformer (out of scope).

---

## Goal

Answer one question honestly: **does a 1D convolutional neural network reading the raw
trend curve beat the 0.64 logistic-regression baseline** under the *same*
leave-one-ingredient-out (LOO) protocol on the *same* 20 vetted ingredients?

- **Success = LOO accuracy > 0.64.**
- A tie or a loss is a legitimate, reported finding ("on this data, the simple model
  wins"), not a failure. We do not tune endlessly chasing a win on data too small to
  support it.

This is an **offline experiment**. The CNN is deliberately NOT wired into `diagnose.py`
or `app.py`. Integrating it into the live demo is a follow-up that happens only if it
beats the baseline (YAGNI).

---

## Why this is framed as an experiment, not a guaranteed upgrade

We have ~50 labeled windows (~20 positive) from 20 ingredients. That is very little data
for any deep model. A neural net with too many parameters will memorize the training set
and fail LOO — possibly scoring *worse* than logistic regression. The whole design is
shaped to give a deep model a fair chance on tiny data (small model + augmentation) while
keeping the comparison airtight (same folds, no leakage).

---

## Architecture & Module Boundaries

Additive only — Phase 1 modules are untouched and reused.

| File | Responsibility |
|------|----------------|
| `src/sequences.py` | Cached IOT windows → fixed-length, per-window-normalized curve arrays; label-preserving augmentation |
| `src/cnn.py` | The tiny 1D-CNN (PyTorch): build, train one fold, predict |
| `src/experiment.py` | Run LOO for BOTH logistic regression and CNN on the same folds; print + save a leaderboard |
| `tests/test_sequences.py` | Unit tests for resample / normalize / augment (pure functions) |

**Reused unchanged:** `fetch.py` (cached curves), `features.py` (`slice_windows` yields
the same windows + labels and the LR feature vectors), `train.py` (LR benchmark logic).

**Dependency:** add `torch` to `requirements.txt`.

---

## Module: `sequences.py`

Turns a labeled window (from `features.slice_windows`) into CNN input. Pure functions,
unit-testable with known inputs.

### resample(values, length=24)
Interpolate a window's `value` series to exactly `length` evenly-spaced points (default
24, ≈ one per month for a 24-month window). Makes every input the same size regardless of
the source cadence.

### normalize(arr)
Min-max scale a single curve to 0–1 (`(x - min) / (max - min + eps)`). Shape matters, not
absolute height — two same-shaped curves at different Google-Trends heights become
identical input. Self-contained per curve, so there is no cross-window leakage.

### augment(arr, rng) → list of variants
Label-preserving perturbations to expand the tiny training set:
- **jitter** — add small Gaussian noise
- **scaling** — multiply amplitude by a small random factor
- **time-shift** — roll the curve a few steps left/right (edge-padded)

Each real curve yields several augmented variants → ~50 windows become a few hundred
training examples.

### build_sequence_dataset()
Convenience: load all 20 ingredients' windows via `slice_windows`, resample + normalize
each, return `(X, y, groups)` arrays parallel to the LR dataset (same ordering semantics,
same `groups` = ingredient names). Augmentation is **not** applied here — it is applied
per-fold inside the experiment (see leakage safeguards).

---

## Module: `cnn.py`

A deliberately tiny 1D-CNN so parameter count stays close to the LR baseline — a fair
fight, not a giant net steamrolling small data.

### Architecture (PyTorch)
- Input: 24-point curve, 1 channel
- Conv1d(1→8, kernel=3, padding=1) → ReLU
- Conv1d(8→16, kernel=3, padding=1) → ReLU
- Global average pooling over the time axis
- Dropout(0.5)
- Linear(16→1) → logit

Only a few hundred learnable parameters.

### Training (`train_fold(X_train, y_train) -> model`)
- Loss: binary cross-entropy with logits
- Optimizer: Adam
- Fixed small epoch budget (named constant), class weighting for the 20-vs-30 imbalance
- All hyperparameters are named module-level constants (tunable, visible)
- Deterministic seed for reproducibility

### Inference (`predict(model, X) -> probs`)
Returns calibrated-ish probabilities via sigmoid on the logits. Same interface shape as
the LR path so `experiment.py` treats both models identically.

---

## Module: `experiment.py`

Produces the answer. Runs the same LOO loop for both models on the same folds.

```
for each held-out ingredient g:
    train = all windows NOT in g
    test  = windows in g

    LR path:
        fit logistic regression on the 8 extracted features of `train`
        predict `test`

    CNN path:
        take resampled+normalized curves of `train`
        AUGMENT those (training only)
        train_fold -> model
        predict resampled+normalized `test` curves (NOT augmented)

collect both models' predictions across all folds
score each: accuracy, precision, recall
```

### Leakage safeguards (explicit)
- **Augmentation is inside the fold, on training curves only** — never on the held-out
  ingredient, never before the split. (Augmented copies of the test curve leaking into
  training is the classic mistake; this prevents it.)
- **Per-window normalization uses only that window's own min/max** — no statistics shared
  across windows, so no normalization leakage.
- **Same held-out ingredient for both models each fold** — truly apples-to-apples.

### Output
A console leaderboard plus a saved metrics artifact for the notebook:

```
Model                  LOO acc   precision   recall
Logistic Regression      0.64       0.54       0.75
1D-CNN (augmented)       0.??       0.??       0.??
verdict: CNN beats / ties / loses to baseline
```

If the CNN ties or loses, `experiment.py` states that plainly. That statement is a valid
deliverable.

---

## Testing

- **TDD on `sequences.py` pure functions:**
  - `resample` returns exactly `length` points
  - `normalize` output is within [0, 1] and preserves monotonic ordering
  - `augment` returns variants that differ in values but keep the same length and label,
    and a flat curve stays flat-ish (shape preserved)
- **`cnn.py` and `experiment.py` are verified by running them** — a deep model's
  correctness lives in its measured LOO behavior, not in a unit assertion. This matches
  how `train.py` was verified in Phase 1.

---

## Out of Scope (Phase 2)

- LSTM, transformer, or any second deep architecture (CNN only this round).
- Wiring the CNN into `diagnose.py` / `app.py` — happens only if it beats the baseline.
- Fetching additional ingredients — the experiment runs on the existing 20.
- Hyperparameter sweeps beyond a sane fixed configuration.

---

## Known Limitations (documented, not hidden)

- ~50 windows / ~20 positive is small; even with augmentation the CNN may not beat LR.
  That outcome is an accepted, reportable finding.
- Augmentation creates *plausible* variants, not new real signal; it reduces overfitting
  but cannot substitute for more real ingredients.
- LOO on 20 groups is a harsh, high-variance metric — small score differences between LR
  and CNN may not be statistically meaningful, and we will say so rather than overclaim.
