import numpy as np
import pandas as pd
import pytest

from academic_success.data import TARGET
from academic_success.pipeline import MODELS, build_model, make_preprocessor, outlier_mask


def test_preprocessor_uses_training_statistics_only(synthetic_df):
    X = synthetic_df.drop(columns=TARGET)
    train, test = X.iloc[:600], X.iloc[600:].copy()
    test["Admission grade"] += 1000  # a shift in held-out data must not affect the fit
    prep = make_preprocessor(X.columns).fit(train)
    scaler = prep.named_transformers_["num"][-1]
    col = prep.transformers_[0][2].index("Admission grade")
    assert scaler.mean_[col] == pytest.approx(train["Admission grade"].mean())


def test_unseen_category_codes_do_not_crash(synthetic_df):
    X = synthetic_df.drop(columns=TARGET)
    prep = make_preprocessor(X.columns).fit(X)
    odd = X.head(3).assign(Course=123456)
    assert prep.transform(odd).shape[0] == 3


def test_preprocessor_imputes_missing_values(synthetic_df):
    X = synthetic_df.drop(columns=TARGET).head(200).copy()
    X.iloc[::3, :5] = np.nan
    assert not np.isnan(make_preprocessor(X.columns).fit_transform(X)).any()


def test_outlier_mask_only_checks_continuous_columns():
    X = pd.DataFrame({"Admission grade": [120.0] * 99 + [900.0], "Course": [1] * 99 + [99999]})
    keep = outlier_mask(X, 4.0)
    assert keep.sum() == 99 and not keep[-1]
    assert outlier_mask(X.drop(columns="Admission grade"), 4.0).all()
    assert outlier_mask(X, None).all()


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_fits_and_returns_probabilities(name, synthetic_df):
    X, y = synthetic_df.drop(columns=TARGET), synthetic_df[TARGET]
    model = build_model(name, X.columns).fit(X, y)
    proba = model.predict_proba(X.head(20))
    assert proba.shape == (20, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)


def test_unknown_model_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("xgboost", ["a"])
