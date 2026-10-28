import pytest
from fastapi.testclient import TestClient

from academic_success import serve
from academic_success.cli import main
from academic_success.data import TARGET, SchemaError, make_synthetic
from academic_success.predict import Predictor


@pytest.fixture()
def client(trained, monkeypatch):
    out, _ = trained
    monkeypatch.setenv("ASP_MODEL", str(out / "model.joblib"))
    serve.get_predictor.cache_clear()
    yield TestClient(serve.app)
    serve.get_predictor.cache_clear()


@pytest.fixture()
def students():
    return make_synthetic(5, seed=11).drop(columns=TARGET).to_dict("records")


def test_predictor_round_trip(trained, students):
    out, report = trained
    p = Predictor.load(out / "model.joblib")
    assert p.metadata["selected_model"] == report["selected"]["model"]
    preds = p.predict(students)
    assert len(preds) == 5
    assert all(abs(sum(r["probabilities"].values()) - 1) < 1e-3 for r in preds)


def test_predictor_rejects_missing_features(trained, students):
    p = Predictor.load(trained[0] / "model.joblib")
    with pytest.raises(SchemaError, match="Admission grade"):
        p.predict([{k: v for k, v in students[0].items() if k != "Admission grade"}])


def test_health_and_schema(client):
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/schema").json()["features"]) == 24


def test_predict_endpoint(client, students):
    students[0]["Admission grade"] = None  # missing values are imputed
    r = client.post("/predict", json={"students": students})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 5
    assert body["predictions"][0]["prediction"] in {"Dropout", "Enrolled", "Graduate"}


def test_predict_endpoint_validation(client, students):
    assert client.post("/predict", json={"students": []}).status_code == 422
    bad = [{k: v for k, v in students[0].items() if k != "Debtor"}]
    r = client.post("/predict", json={"students": bad})
    assert r.status_code == 422 and "Debtor" in r.json()["detail"]


def test_cli_predict(trained, tmp_path, capsys):
    path = tmp_path / "students.csv"
    make_synthetic(4, seed=5).drop(columns=TARGET).to_csv(path, index=False)
    assert main(["predict", str(path), "--model", str(trained[0] / "model.joblib")]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 4


def test_cli_train(tmp_path, capsys):
    out = tmp_path / "cli"
    code = main(
        ["train", "--out", str(out), "--models", "logreg", "--cv-folds", "2", "--no-outlier-filter"]
    )
    assert code == 0 and (out / "model.joblib").exists()
    assert "Logistic Regression" in capsys.readouterr().out
