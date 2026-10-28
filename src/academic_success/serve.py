"""HTTP inference service.

ASP_MODEL=artifacts/model.joblib uvicorn academic_success.serve:app
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .data import SchemaError
from .predict import Predictor

MAX_BATCH = 1000


class PredictRequest(BaseModel):
    students: list[dict[str, float | int | None]] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH,
        description="One object per student, keyed by the training feature names.",
    )


class Prediction(BaseModel):
    prediction: str
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    model: str
    predictions: list[Prediction]


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor.load(os.environ.get("ASP_MODEL", "artifacts/model.joblib"))


app = FastAPI(title="Academic Success Prediction", version=__version__)


@app.get("/health")
def health() -> dict:
    meta = get_predictor().metadata
    return {"status": "ok", "model": meta["selected_model"], "trained_at": meta["trained_at"]}


@app.get("/schema")
def schema() -> dict:
    p = get_predictor()
    return {"features": p.features, "classes": p.class_names}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    p = get_predictor()
    try:
        preds = p.predict(req.students)
    except SchemaError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return PredictResponse(model=p.metadata["selected_model"], predictions=preds)
