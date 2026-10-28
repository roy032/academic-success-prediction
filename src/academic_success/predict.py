"""Load a trained bundle and score new students."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import SchemaError, _normalise_columns


@dataclass
class Predictor:
    pipeline: object
    metadata: dict

    @classmethod
    def load(cls, path: str | Path) -> Predictor:
        bundle = joblib.load(path)
        return cls(bundle["pipeline"], bundle["metadata"])

    @property
    def features(self) -> list[str]:
        return self.metadata["features"]

    @property
    def class_names(self) -> list[str]:
        return self.metadata["class_names"]

    def _frame(self, records: pd.DataFrame | list[dict]) -> pd.DataFrame:
        df = _normalise_columns(pd.DataFrame(records))
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise SchemaError(f"missing feature columns: {missing}")
        # Unknown extra columns are ignored; NaNs are imputed by the pipeline.
        return df[self.features].apply(pd.to_numeric, errors="coerce")

    def predict_proba(self, records) -> np.ndarray:
        return self.pipeline.predict_proba(self._frame(records))

    def predict(self, records) -> list[dict]:
        proba = self.predict_proba(records)
        return [
            {
                "prediction": self.class_names[int(p.argmax())],
                "probabilities": {
                    c: round(float(v), 4) for c, v in zip(self.class_names, p, strict=True)
                },
            }
            for p in proba
        ]
