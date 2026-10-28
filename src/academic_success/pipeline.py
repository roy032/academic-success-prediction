"""Preprocessing and model definitions.

Every data-dependent step lives inside a scikit-learn ``Pipeline`` so that it is
fit on training rows only, both in cross-validation and in the final fit.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CONTINUOUS, split_columns


def make_preprocessor(columns, dense: bool = True) -> ColumnTransformer:
    categorical, numeric = split_columns(columns)
    onehot = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=not dense
    )
    return ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), onehot), categorical),
        ],
        verbose_feature_names_out=False,
    )


def _with_prep(estimator) -> Callable[[list[str], int], Pipeline]:
    def build(columns, seed):
        est = estimator(seed)
        return Pipeline([("prep", make_preprocessor(columns)), ("model", est)])

    return build


MODELS: dict[str, Callable[[list[str], int], Pipeline]] = {
    "logreg": _with_prep(lambda s: LogisticRegression(max_iter=2000, random_state=s)),
    "logreg_balanced": _with_prep(
        lambda s: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=s)
    ),
    "naive_bayes": _with_prep(lambda s: GaussianNB()),
    "mlp": _with_prep(
        lambda s: MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=1e-3,
            early_stopping=True,
            max_iter=300,
            random_state=s,
        )
    ),
    "hist_gb_balanced": _with_prep(
        lambda s: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            random_state=s,
        )
    ),
}

DISPLAY_NAMES = {
    "logreg": "Logistic Regression",
    "logreg_balanced": "Logistic Regression (balanced)",
    "naive_bayes": "Gaussian Naive Bayes",
    "mlp": "Neural Network (MLP)",
    "hist_gb_balanced": "Gradient Boosting (balanced)",
}


def build_model(name: str, columns, seed: int = 42) -> Pipeline:
    try:
        return MODELS[name](list(columns), seed)
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}") from None


def outlier_mask(X: pd.DataFrame, z_limit: float | None) -> np.ndarray:
    """Rows to keep: |z| <= z_limit on continuous columns, using X's own statistics.

    Only continuous columns are checked; z-scores on category codes or binary
    flags are meaningless. Call it on training rows only.
    """
    cols = [c for c in CONTINUOUS if c in X.columns]
    if z_limit is None or not cols:
        return np.ones(len(X), dtype=bool)
    values = X[cols].astype(float)
    values = values.fillna(values.median())
    std = values.std(ddof=0).replace(0, 1.0)
    z = ((values - values.mean()) / std).abs()
    return (z <= z_limit).all(axis=1).to_numpy()
