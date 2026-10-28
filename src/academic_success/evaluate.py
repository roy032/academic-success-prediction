"""Metrics, cross-validation and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .data import CLASS_NAMES
from .pipeline import outlier_mask

LABELS = list(range(len(CLASS_NAMES)))


def classification_metrics(y_true, y_pred, proba=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "per_class_f1": dict(
            zip(
                CLASS_NAMES,
                f1_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0),
                strict=True,
            )
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }
    if proba is not None:
        out["roc_auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", labels=LABELS)
        out["log_loss"] = log_loss(y_true, proba, labels=LABELS)
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in out.items()}


def text_report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred, labels=LABELS, target_names=CLASS_NAMES, zero_division=0
    )


def cross_validate(
    model, X: pd.DataFrame, y: pd.Series, folds: int, seed: int, z_limit: float | None
) -> dict:
    """Stratified K-fold macro-F1; outliers are removed from each training fold only."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        keep = outlier_mask(X_tr, z_limit)
        fitted = clone(model).fit(X_tr[keep], y_tr[keep])
        pred = fitted.predict(X.iloc[val_idx])
        scores.append(f1_score(y.iloc[val_idx], pred, average="macro", zero_division=0))
    scores = np.asarray(scores)
    return {
        "macro_f1_mean": float(scores.mean()),
        "macro_f1_std": float(scores.std(ddof=1)),
        "fold_scores": scores.round(4).tolist(),
    }


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for test accuracy and macro-F1."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    acc, f1 = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        acc[i] = accuracy_score(y_true[idx], y_pred[idx])
        f1[i] = f1_score(y_true[idx], y_pred[idx], average="macro", labels=LABELS, zero_division=0)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {
        "accuracy": np.percentile(acc, q).round(4).tolist(),
        "macro_f1": np.percentile(f1, q).round(4).tolist(),
        "n_boot": n_boot,
    }
