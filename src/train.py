"""Train the logistic-regression baseline with leave-one-ingredient-out eval."""
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.config import FEATURES_PATH, MODEL_PATH, PROCESSED_DIR
from src.features import FEATURE_COLUMNS


def _load_xy():
    df = pd.read_csv(FEATURES_PATH)
    # Impute missing features (e.g. geographic_entropy when region fetches were
    # rate-limited) with the column median rather than dropping whole rows — losing
    # an ingredient costs more than approximating one feature. Median is stored on
    # the model so inference imputes identically.
    feats = df[FEATURE_COLUMNS].astype(float)
    medians = feats.median(numeric_only=True).fillna(0.0)
    feats = feats.fillna(medians)
    X = feats.to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups = df["ingredient"].to_numpy()
    return X, y, groups, medians


def evaluate(X, y, groups):
    """Leave-one-ingredient-out CV. Returns (accuracy, precision, recall)."""
    logo = LeaveOneGroupOut()
    preds, truths = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        preds.extend(clf.predict(scaler.transform(X[test_idx])))
        truths.extend(y[test_idx])
    return (accuracy_score(truths, preds),
            precision_score(truths, preds, zero_division=0),
            recall_score(truths, preds, zero_division=0))


def train_and_save():
    X, y, groups, medians = _load_xy()
    acc, prec, rec = evaluate(X, y, groups)
    print(f"LOO accuracy={acc:.2f} precision={prec:.2f} recall={rec:.2f}")
    if acc < 0.70:
        print("WARNING: accuracy below the 0.70 success criterion.")

    # Fit final model on ALL data for inference.
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(scaler.transform(X), y)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf, "columns": FEATURE_COLUMNS,
                     "medians": medians.to_dict()}, f)
    print(f"Saved model -> {MODEL_PATH}")
    return acc, prec, rec


if __name__ == "__main__":
    train_and_save()
