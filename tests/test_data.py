import pandas as pd
import pytest

from academic_success.data import (
    CLASS_NAMES,
    CURRICULAR,
    ENROLLMENT_FEATURES,
    TARGET,
    SchemaError,
    load_dataset,
    make_synthetic,
    prepare,
)


def test_synthetic_matches_uci_schema(synthetic_df):
    assert list(synthetic_df.columns) == ENROLLMENT_FEATURES + [TARGET]
    assert synthetic_df.notna().all().all()


def test_synthetic_class_balance_is_realistic():
    shares = make_synthetic(n=5000, seed=1)[TARGET].value_counts(normalize=True).sort_index()
    assert shares.tolist() == pytest.approx([0.32, 0.18, 0.50], abs=0.01)


def test_synthetic_is_deterministic():
    pd.testing.assert_frame_equal(make_synthetic(300, seed=7), make_synthetic(300, seed=7))


def test_synthetic_curricular_columns():
    df = make_synthetic(200, include_curricular=True)
    assert set(CURRICULAR) <= set(df.columns)


def test_prepare_encodes_labels_and_drops_invalid_rows():
    raw = pd.DataFrame(
        {"Age at enrollment ": [18, 19, 20, 21], TARGET: ["Graduate", "Dropout", None, "Unknown"]}
    )
    df = prepare(raw)
    assert df[TARGET].tolist() == [2, 0]
    assert "Age at enrollment" in df.columns  # header whitespace stripped


def test_prepare_drops_curricular_by_default():
    raw = make_synthetic(50, include_curricular=True)
    assert not any(c.startswith("Curricular") for c in prepare(raw).columns)
    assert set(CURRICULAR) <= set(prepare(raw, include_curricular=True).columns)


def test_prepare_requires_target():
    with pytest.raises(SchemaError):
        prepare(pd.DataFrame({"a": [1]}))


@pytest.mark.parametrize("sep", [";", ","])
def test_load_dataset_accepts_both_delimiters(tmp_path, sep):
    raw = make_synthetic(60, seed=3)
    raw[TARGET] = [CLASS_NAMES[i] for i in raw[TARGET]]
    path = tmp_path / "data.csv"
    raw.to_csv(path, sep=sep, index=False, encoding="utf-8-sig")
    df = load_dataset(path)
    assert df.shape == raw.shape
    assert df[TARGET].between(0, 2).all()


def test_synthetic_missing_rate_leaves_target_intact():
    df = make_synthetic(2000, missing_rate=0.1)
    assert df[TARGET].notna().all()
    assert df.drop(columns=TARGET).isna().mean().mean() == pytest.approx(0.1, abs=0.01)
