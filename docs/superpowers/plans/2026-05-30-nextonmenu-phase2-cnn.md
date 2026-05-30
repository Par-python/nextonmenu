# NextOnMenu Phase 2 — 1D-CNN Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tiny 1D-CNN that reads raw trend curves and compare it head-to-head against the Phase 1 logistic-regression baseline (0.64) under identical leave-one-ingredient-out evaluation, reporting the honest result.

**Architecture:** Three additive modules — `sequences.py` (raw window → fixed-length normalized curve + augmentation), `cnn.py` (tiny PyTorch 1D-CNN), `experiment.py` (runs LOO for both models on the same folds, prints a leaderboard). Phase 1 code is reused unchanged. Augmentation happens strictly inside each training fold to prevent leakage.

**Tech Stack:** Python 3, PyTorch, numpy, pandas, scikit-learn, pytest. Reuses `src/fetch.py`, `src/features.py`, `src/config.py`, `src/train.py`.

**Naming/context note:** "LOO" = leave-one-ingredient-out CV. "LR" = the Phase 1 logistic regression. The CNN is an offline experiment — it is NOT wired into `diagnose.py`/`app.py` this phase.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `requirements.txt` | add `torch` |
| `src/sequences.py` | resample, normalize, augment, build_sequence_dataset |
| `src/cnn.py` | TrendCNN model, train_fold, predict |
| `src/experiment.py` | LOO runner for LR + CNN, leaderboard output |
| `tests/test_sequences.py` | unit tests for the pure sequence functions |

All run commands assume the venv is active: `source .venv/bin/activate`.

---

## Task 1: Add PyTorch dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add torch to requirements.txt**

Add this line after the `pytest>=7.4` line in `requirements.txt`:

```
torch>=2.0
```

- [ ] **Step 2: Install it**

Run: `source .venv/bin/activate && pip install "torch>=2.0"`
Expected: torch installs without error (large download, may take a few minutes).

- [ ] **Step 3: Verify import + reproducibility seed works**

Run:
```bash
source .venv/bin/activate && python -c "
import torch
torch.manual_seed(0)
print('torch', torch.__version__, 'tensor', torch.randn(2).shape)
"
```
Expected: prints the torch version and `tensor torch.Size([2])`.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add torch dependency for Phase 2 CNN"
```

---

## Task 2: resample (TDD)

**Files:**
- Create: `src/sequences.py`
- Test: `tests/test_sequences.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sequences.py
import numpy as np
from src.sequences import resample


def test_resample_returns_exact_length():
    arr = np.arange(50, dtype=float)
    out = resample(arr, length=24)
    assert len(out) == 24


def test_resample_short_input_upsamples():
    arr = np.array([0.0, 10.0], dtype=float)
    out = resample(arr, length=24)
    assert len(out) == 24
    assert out[0] == 0.0 and abs(out[-1] - 10.0) < 1e-9  # endpoints preserved


def test_resample_preserves_monotonic_trend():
    arr = np.linspace(0, 100, 40)
    out = resample(arr, length=24)
    assert np.all(np.diff(out) >= -1e-9)  # still non-decreasing
```

- [ ] **Step 2: Run to verify failure**

Run: `source .venv/bin/activate && pytest tests/test_sequences.py -v`
Expected: FAIL — `ImportError: cannot import name 'resample'`.

- [ ] **Step 3: Implement resample**

```python
# src/sequences.py
"""Raw trend windows -> fixed-length, normalized curves for the 1D-CNN."""
import numpy as np


def resample(values, length=24):
    """Linearly interpolate a 1D series to exactly `length` evenly-spaced points."""
    arr = np.asarray(values, dtype=float)
    if arr.size == length:
        return arr.copy()
    src_x = np.linspace(0.0, 1.0, num=arr.size)
    dst_x = np.linspace(0.0, 1.0, num=length)
    return np.interp(dst_x, src_x, arr)
```

- [ ] **Step 4: Run to verify pass**

Run: `source .venv/bin/activate && pytest tests/test_sequences.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sequences.py tests/test_sequences.py
git commit -m "feat: resample trend windows to fixed length"
```

---

## Task 3: normalize (TDD)

**Files:**
- Modify: `src/sequences.py`
- Test: `tests/test_sequences.py`

- [ ] **Step 1: Add failing tests**

```python
# append to tests/test_sequences.py
from src.sequences import normalize


def test_normalize_range_is_0_1():
    arr = np.array([5.0, 10.0, 20.0, 100.0])
    out = normalize(arr)
    assert out.min() == 0.0 and out.max() == 1.0


def test_normalize_preserves_order():
    arr = np.array([3.0, 1.0, 2.0])
    out = normalize(arr)
    assert out[1] < out[2] < out[0]  # ranking unchanged


def test_normalize_flat_curve_does_not_divide_by_zero():
    arr = np.array([7.0, 7.0, 7.0])
    out = normalize(arr)
    assert np.all(np.isfinite(out))
```

- [ ] **Step 2: Run to verify failure**

Run: `source .venv/bin/activate && pytest tests/test_sequences.py -k normalize -v`
Expected: FAIL — `cannot import name 'normalize'`.

- [ ] **Step 3: Implement normalize**

```python
# append to src/sequences.py
def normalize(values):
    """Min-max scale a single curve to [0, 1]. Flat curves map to all-zeros."""
    arr = np.asarray(values, dtype=float)
    lo = arr.min()
    rng = arr.max() - lo
    if rng < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / rng
```

- [ ] **Step 4: Run to verify pass**

Run: `source .venv/bin/activate && pytest tests/test_sequences.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/sequences.py tests/test_sequences.py
git commit -m "feat: per-curve min-max normalization"
```

---

## Task 4: augment (TDD)

**Files:**
- Modify: `src/sequences.py`
- Test: `tests/test_sequences.py`

- [ ] **Step 1: Add failing tests**

```python
# append to tests/test_sequences.py
from src.sequences import augment


def test_augment_returns_requested_count():
    rng = np.random.default_rng(0)
    arr = np.linspace(0, 1, 24)
    variants = augment(arr, rng, n=5)
    assert len(variants) == 5


def test_augment_preserves_length():
    rng = np.random.default_rng(0)
    arr = np.linspace(0, 1, 24)
    for v in augment(arr, rng, n=3):
        assert len(v) == 24


def test_augment_changes_values_but_keeps_shape():
    rng = np.random.default_rng(0)
    arr = np.linspace(0, 1, 24)  # clearly rising
    variants = augment(arr, rng, n=5)
    # at least one variant differs from the original
    assert any(not np.allclose(v, arr) for v in variants)
    # rising shape preserved: each variant's second half mean > first half mean
    for v in variants:
        assert v[12:].mean() > v[:12].mean()
```

- [ ] **Step 2: Run to verify failure**

Run: `source .venv/bin/activate && pytest tests/test_sequences.py -k augment -v`
Expected: FAIL — `cannot import name 'augment'`.

- [ ] **Step 3: Implement augment**

```python
# append to src/sequences.py
def augment(values, rng, n=4, jitter=0.02, scale_range=0.1, max_shift=2):
    """Return `n` label-preserving variants of a normalized curve.

    Each variant applies small jitter (Gaussian noise), amplitude scaling, and a
    time-shift (edge-padded roll). All perturbations are small enough to preserve
    the curve's shape and therefore its label.
    """
    arr = np.asarray(values, dtype=float)
    variants = []
    for _ in range(n):
        v = arr.copy()
        v = v * (1.0 + rng.uniform(-scale_range, scale_range))   # amplitude scale
        v = v + rng.normal(0.0, jitter, size=v.shape)            # jitter
        shift = int(rng.integers(-max_shift, max_shift + 1))     # time-shift
        if shift != 0:
            v = np.roll(v, shift)
            if shift > 0:
                v[:shift] = v[shift]
            else:
                v[shift:] = v[shift - 1]
        variants.append(v)
    return variants
```

- [ ] **Step 4: Run to verify pass**

Run: `source .venv/bin/activate && pytest tests/test_sequences.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/sequences.py tests/test_sequences.py
git commit -m "feat: label-preserving curve augmentation"
```

---

## Task 5: build_sequence_dataset (orchestration)

**Files:**
- Modify: `src/sequences.py`

Builds parallel arrays to the LR dataset: each labeled window becomes a resampled +
normalized 24-point curve. NO augmentation here (applied per-fold in the experiment).

- [ ] **Step 1: Add build_sequence_dataset**

```python
# append to src/sequences.py
SEQ_LENGTH = 24

def build_sequence_dataset(ingredients=None, length=SEQ_LENGTH):
    """Return (X, y, groups): X is (n, length) normalized curves, y labels,
    groups ingredient names. Uses the SAME windows as the LR dataset."""
    from src.config import VIRAL_INGREDIENTS, FULL_TIMEFRAME
    from src.fetch import fetch_interest_over_time
    from src.features import slice_windows

    ingredients = ingredients or VIRAL_INGREDIENTS
    X, y, groups = [], [], []
    for ing in ingredients:
        iot = fetch_interest_over_time(ing, timeframe=FULL_TIMEFRAME, tag="full")
        for w in slice_windows(iot):
            curve = normalize(resample(w["data"]["value"].to_numpy(), length))
            X.append(curve)
            y.append(w["label"])
            groups.append(ing)
    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(groups)
```

- [ ] **Step 2: Verify it builds from cached data (no network)**

Run:
```bash
source .venv/bin/activate && python -c "
from src.sequences import build_sequence_dataset
X, y, g = build_sequence_dataset()
print('X', X.shape, 'y', y.shape, 'pos', int(y.sum()), 'ingredients', len(set(g)))
"
```
Expected: `X (50, 24) y (50,) pos 20 ingredients 20` (numbers may shift slightly if cache differs; X must be (N, 24) with both labels present).

- [ ] **Step 3: Commit**

```bash
git add src/sequences.py
git commit -m "feat: build resampled+normalized sequence dataset"
```

---

## Task 6: TrendCNN model + train_fold + predict

**Files:**
- Create: `src/cnn.py`

Verified by running (deep-model correctness is in measured behavior), consistent with how
`train.py` was verified in Phase 1.

- [ ] **Step 1: Write `src/cnn.py`**

```python
"""A deliberately tiny 1D-CNN for classifying trend-curve shape."""
import numpy as np
import torch
import torch.nn as nn

SEED = 0
EPOCHS = 80
LR = 1e-3
DROPOUT = 0.5


class TrendCNN(nn.Module):
    """1 input channel -> two small conv layers -> global avg pool -> logit."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # global average pool over time
        )
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        # x: (batch, length) -> (batch, 1, length)
        x = x.unsqueeze(1)
        z = self.net(x).squeeze(-1)    # (batch, 16)
        return self.head(self.drop(z)).squeeze(-1)  # (batch,)


def train_fold(X_train, y_train):
    """Train a TrendCNN on (n, length) curves. Returns the trained model."""
    torch.manual_seed(SEED)
    model = TrendCNN()
    Xt = torch.tensor(np.asarray(X_train), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y_train), dtype=torch.float32)

    # class weighting for the positive class (handles 20-vs-30 imbalance)
    n_pos = max(float(yt.sum()), 1.0)
    n_neg = max(float((yt == 0).sum()), 1.0)
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = model(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    return model


def predict(model, X):
    """Return probabilities for (n, length) curves."""
    model.eval()
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(Xt)).numpy()
```

- [ ] **Step 2: Smoke-test the model trains and predicts on toy data**

Run:
```bash
source .venv/bin/activate && python -c "
import numpy as np
from src.cnn import train_fold, predict
rng = np.random.default_rng(0)
# 10 rising (label 1), 10 flat (label 0)
rising = [np.linspace(0,1,24)+rng.normal(0,0.02,24) for _ in range(10)]
flat   = [np.zeros(24)+rng.normal(0,0.02,24) for _ in range(10)]
X = np.array(rising+flat); y = np.array([1]*10+[0]*10)
m = train_fold(X, y)
p = predict(m, X)
print('mean prob rising:', round(p[:10].mean(),2), 'mean prob flat:', round(p[10:].mean(),2))
"
```
Expected: rising mean prob clearly higher than flat (e.g. >0.6 vs <0.4) — confirms the model learns the obvious separable case.

- [ ] **Step 3: Commit**

```bash
git add src/cnn.py
git commit -m "feat: tiny 1D-CNN with train_fold and predict"
```

---

## Task 7: Experiment runner — LOO for both models

**Files:**
- Create: `src/experiment.py`

- [ ] **Step 1: Write `src/experiment.py`**

```python
"""Run leave-one-ingredient-out for LR and the 1D-CNN on the same folds."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.config import FEATURES_PATH, PROCESSED_DIR
from src.features import FEATURE_COLUMNS
from src.sequences import build_sequence_dataset, augment
from src.cnn import train_fold, predict

AUG_PER_CURVE = 4


def _lr_xy():
    df = pd.read_csv(FEATURES_PATH)
    feats = df[FEATURE_COLUMNS].astype(float)
    feats = feats.fillna(feats.median(numeric_only=True).fillna(0.0))
    return (feats.to_numpy(float), df["label"].to_numpy(int),
            df["ingredient"].to_numpy())


def _scores(truths, preds):
    return (accuracy_score(truths, preds),
            precision_score(truths, preds, zero_division=0),
            recall_score(truths, preds, zero_division=0))


def run_lr_loo():
    X, y, g = _lr_xy()
    P, T = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        P.extend(clf.predict(sc.transform(X[te])))
        T.extend(y[te])
    return _scores(T, P)


def run_cnn_loo():
    X, y, g = build_sequence_dataset()
    rng = np.random.default_rng(0)
    P, T = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        # AUGMENT training curves only — never the held-out ingredient.
        aug_X, aug_y = [], []
        for curve, label in zip(X[tr], y[tr]):
            aug_X.append(curve); aug_y.append(label)
            for v in augment(curve, rng, n=AUG_PER_CURVE):
                aug_X.append(v); aug_y.append(label)
        model = train_fold(np.array(aug_X), np.array(aug_y))
        probs = predict(model, X[te])
        P.extend((probs >= 0.5).astype(int))
        T.extend(y[te])
    return _scores(T, P)


def main():
    lr = run_lr_loo()
    cnn = run_cnn_loo()
    print(f"{'Model':<22}{'LOO acc':>9}{'precision':>11}{'recall':>9}")
    print(f"{'Logistic Regression':<22}{lr[0]:>9.2f}{lr[1]:>11.2f}{lr[2]:>9.2f}")
    print(f"{'1D-CNN (augmented)':<22}{cnn[0]:>9.2f}{cnn[1]:>11.2f}{cnn[2]:>9.2f}")
    if cnn[0] > lr[0]:
        verdict = f"CNN BEATS baseline ({cnn[0]:.2f} > {lr[0]:.2f})"
    elif abs(cnn[0] - lr[0]) < 1e-9:
        verdict = f"CNN TIES baseline ({cnn[0]:.2f})"
    else:
        verdict = f"CNN LOSES to baseline ({cnn[0]:.2f} < {lr[0]:.2f})"
    print("verdict:", verdict)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [["logistic_regression", *lr], ["cnn_augmented", *cnn]],
        columns=["model", "accuracy", "precision", "recall"],
    ).to_csv(PROCESSED_DIR / "experiment_results.csv", index=False)
    print("Saved -> experiment_results.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

Run: `source .venv/bin/activate && python -m src.experiment`
Expected: prints the two-row leaderboard, an LR accuracy of ~0.64, a CNN accuracy, and a verdict line. Writes `data/processed/experiment_results.csv`. (The CNN number is the experimental result — whatever it is, it is reported honestly.)

- [ ] **Step 3: Verify the results artifact**

Run:
```bash
source .venv/bin/activate && python -c "
import pandas as pd; from src.config import PROCESSED_DIR
print(pd.read_csv(PROCESSED_DIR/'experiment_results.csv'))
"
```
Expected: a 2-row table with model, accuracy, precision, recall.

- [ ] **Step 4: Commit**

```bash
git add src/experiment.py
git commit -m "feat: LOO experiment runner comparing LR vs 1D-CNN"
```

---

## Task 8: Full test pass + record the finding

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the whole suite**

Run: `source .venv/bin/activate && pytest -q`
Expected: all tests pass (Phase 1 + the new `test_sequences.py`).

- [ ] **Step 2: Add a Phase 2 note to README with the ACTUAL result**

Read the verdict printed by `python -m src.experiment`, then append this section to
`README.md`, filling the bracketed numbers with the real values from
`data/processed/experiment_results.csv`:

```markdown
## Phase 2 — 1D-CNN experiment

We tested whether a small 1D-CNN reading the raw curve beats the logistic-regression
baseline, under the same leave-one-ingredient-out evaluation.

| Model | LOO accuracy | precision | recall |
|-------|-------------|-----------|--------|
| Logistic Regression | [LR_ACC] | [LR_PREC] | [LR_REC] |
| 1D-CNN (augmented)  | [CNN_ACC] | [CNN_PREC] | [CNN_REC] |

**Finding:** [one honest sentence — e.g. "the CNN did not beat the baseline on 50
windows, confirming that simple interpretable models hold up on small, noisy trend
data" OR "the CNN edged out the baseline by X points"]. Run it yourself with
`python -m src.experiment`.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: record Phase 2 CNN-vs-LR experiment result"
```

---

## Self-Review Notes (addressed)

- **Spec coverage:** sequences resample/normalize/augment (T2/3/4), build_sequence_dataset no-aug (T5), tiny CNN + train_fold/predict (T6), LOO runner for both models with in-fold-only augmentation (T7), honest verdict + saved artifact (T7), README finding (T8), torch dep (T1). ✓
- **Leakage safeguard:** augmentation is applied only to `X[tr]` inside the fold loop in T7; held-out `X[te]` is predicted un-augmented. Normalization is per-curve (T3), no cross-window stats. ✓
- **Type consistency:** `build_sequence_dataset` returns `(X, y, groups)` used identically in T7; `train_fold(X, y) -> model` and `predict(model, X) -> probs` signatures match between T6 and T7; `FEATURE_COLUMNS` reused from Phase 1. ✓
- **Honest-finding clause:** verdict logic in T7 prints beats/ties/loses; README step T8 records whatever actually happened. ✓
- **Out of scope respected:** CNN not wired into diagnose.py/app.py; no LSTM; no extra ingredients. ✓
