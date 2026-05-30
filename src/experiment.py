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
