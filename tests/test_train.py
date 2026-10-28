import json

from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from academic_success.data import TARGET, make_synthetic
from academic_success.train import TrainConfig, run


def test_artifacts_are_written(trained):
    out, report = trained
    assert (out / "model.joblib").exists()
    assert json.loads((out / "metrics.json").read_text())["selected"]["model"]
    assert "(selected)" in (out / "results.md").read_text()
    for fig in report["figures"]:
        assert (out / fig).stat().st_size > 10_000


def test_selection_uses_cv_not_test(trained):
    _, report = trained
    best = report["selected"]["model"]
    assert best == max(report["cv"], key=lambda m: report["cv"][m]["macro_f1_mean"])


def test_selected_model_beats_majority_baseline(trained):
    _, report = trained
    df = make_synthetic(1200, seed=42)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    dummy = DummyClassifier(strategy="most_frequent").fit(X_dev, y_dev)
    baseline = f1_score(y_test, dummy.predict(X_test), average="macro")
    best = report["selected"]["model"]
    assert report["test"][best]["macro_f1"] > baseline + 0.2


def test_training_is_reproducible(tmp_path):
    kwargs = dict(synthetic_rows=600, cv_folds=2, n_boot=50, models=["logreg_balanced"])
    a = run(TrainConfig(out=str(tmp_path / "a"), **kwargs))
    b = run(TrainConfig(out=str(tmp_path / "b"), **kwargs))
    assert a["cv"] == b["cv"] and a["test"] == b["test"]
