import numpy as np
import pytest

from academic_success.data import TARGET
from academic_success.evaluate import bootstrap_ci, classification_metrics, cross_validate
from academic_success.pipeline import build_model


def test_perfect_predictions():
    y = np.array([0, 1, 2, 2, 1, 0])
    proba = np.eye(3)[y] * 0.97 + 0.01
    m = classification_metrics(y, y, proba)
    assert m["accuracy"] == m["macro_f1"] == m["roc_auc_ovr"] == 1.0
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    pred = np.where(rng.random(500) < 0.7, y, rng.integers(0, 3, 500))
    ci = bootstrap_ci(y, pred, n_boot=300)
    point = (y == pred).mean()
    assert ci["accuracy"][0] < point < ci["accuracy"][1]


def test_cross_validate_returns_one_score_per_fold(synthetic_df):
    X, y = synthetic_df.drop(columns=TARGET), synthetic_df[TARGET]
    res = cross_validate(build_model("logreg", X.columns), X, y, folds=3, seed=0, z_limit=4.0)
    assert len(res["fold_scores"]) == 3
    assert res["macro_f1_mean"] == pytest.approx(np.mean(res["fold_scores"]), abs=1e-4)
