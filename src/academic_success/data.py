"""Loading, schema validation, and a schema-faithful synthetic dataset.

The real data is the UCI "Predict Students' Dropout and Academic Success"
release (https://archive.ics.uci.edu/dataset/697). It is not redistributed
here; ``make_synthetic`` produces a frame with the same columns, dtypes and
class balance so the full pipeline, the tests and CI run without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Target"
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]
TARGET_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# Integer-coded nominal columns: the codes carry no order, so they are one-hot encoded.
CATEGORICAL = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]
BINARY = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]
CONTINUOUS = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]
CURRICULAR_PREFIX = "Curricular units"
CURRICULAR = [
    f"{CURRICULAR_PREFIX} {sem} sem ({kind})"
    for sem in ("1st", "2nd")
    for kind in ("credited", "enrolled", "evaluations", "approved", "grade", "without evaluations")
]
ENROLLMENT_FEATURES = CATEGORICAL + BINARY + CONTINUOUS


class SchemaError(ValueError):
    """Raised when an input frame cannot be used by the pipeline."""


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # The UCI header has stray whitespace, e.g. "Daytime/evening attendance\t".
    df.columns = df.columns.str.strip()
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def load_dataset(path: str | Path, include_curricular: bool = False) -> pd.DataFrame:
    """Read the UCI file (``;``-delimited, BOM) or the comma-delimited course variant."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    return prepare(df, include_curricular=include_curricular)


def prepare(df: pd.DataFrame, include_curricular: bool = False) -> pd.DataFrame:
    """Normalise columns, encode the target and drop rows without a valid label."""
    df = _normalise_columns(df)
    if TARGET not in df.columns:
        raise SchemaError(f"missing target column {TARGET!r}")
    if not include_curricular:
        # Enrollment-time prediction: semester results are not known yet.
        df = df.loc[:, ~df.columns.str.startswith(CURRICULAR_PREFIX)]
    labels = df[TARGET]
    if not pd.api.types.is_numeric_dtype(labels):
        labels = labels.map(TARGET_MAP)
    df = df.assign(**{TARGET: labels}).dropna(subset=[TARGET])
    df = df[df[TARGET].isin(range(len(CLASS_NAMES)))]
    if df.empty:
        raise SchemaError("no rows with a valid target label")
    return df.astype({TARGET: int}).reset_index(drop=True)


def split_columns(columns) -> tuple[list[str], list[str]]:
    """Return (categorical, numeric) feature names for the given columns."""
    columns = [c for c in columns if c != TARGET]
    categorical = [c for c in columns if c in CATEGORICAL]
    numeric = [c for c in columns if c not in categorical]
    return categorical, numeric


def make_synthetic(
    n: int = 4424, seed: int = 42, include_curricular: bool = False, missing_rate: float = 0.0
) -> pd.DataFrame:
    """Synthetic students with the UCI schema and a plausible signal.

    Outcomes are drawn from a latent "risk" score driven by the features that
    matter in the real data (tuition status, scholarship, age, admission grade,
    semester results), so models learn something meaningful but imperfect.
    """
    rng = np.random.default_rng(seed)

    def codes(values, p=None):
        return rng.choice(values, size=n, p=p)

    df = pd.DataFrame(
        {
            "Marital status": codes([1, 2, 4, 3, 5, 6], [0.88, 0.08, 0.02, 0.01, 0.005, 0.005]),
            "Application mode": codes([1, 17, 39, 43, 44, 7, 18, 42, 51, 53]),
            "Course": codes([9500, 9147, 9238, 9254, 9670, 9773, 9991, 9085, 9003, 9119, 171]),
            "Previous qualification": codes(
                [1, 39, 19, 3, 12, 40], [0.84, 0.05, 0.04, 0.03, 0.02, 0.02]
            ),
            "Nacionality": codes(
                [1, 41, 26, 6, 22, 24], [0.975, 0.008, 0.005, 0.005, 0.004, 0.003]
            ),
            "Mother's qualification": codes([1, 19, 37, 38, 3, 34, 2]),
            "Father's qualification": codes([1, 19, 37, 38, 3, 34, 2]),
            "Mother's occupation": codes([9, 4, 5, 7, 3, 0, 2, 10]),
            "Father's occupation": codes([9, 4, 5, 7, 3, 0, 2, 10]),
            "Daytime/evening attendance": rng.binomial(1, 0.89, n),
            "Displaced": rng.binomial(1, 0.55, n),
            "Educational special needs": rng.binomial(1, 0.01, n),
            "Debtor": rng.binomial(1, 0.11, n),
            "Tuition fees up to date": rng.binomial(1, 0.88, n),
            "Gender": rng.binomial(1, 0.35, n),
            "Scholarship holder": rng.binomial(1, 0.25, n),
            "International": rng.binomial(1, 0.025, n),
            "Application order": np.clip(rng.poisson(1.7, n), 0, 9),
            "Previous qualification (grade)": np.round(
                np.clip(rng.normal(132.6, 13.2, n), 95, 190), 1
            ),
            "Admission grade": np.round(np.clip(rng.normal(127.0, 14.5, n), 95, 190), 1),
            "Age at enrollment": np.clip(np.round(17 + rng.gamma(1.3, 4.5, n)), 17, 70).astype(int),
            "Unemployment rate": codes([10.8, 13.9, 9.4, 16.2, 15.5, 8.9, 12.7, 11.1, 7.6, 12.4]),
            "Inflation rate": codes([1.4, -0.3, 0.6, 2.6, 0.5, 3.7, 1.9, 0.3, -0.8, 2.8]),
            "GDP": codes([1.74, 0.79, -3.12, -1.7, 0.32, 2.02, 3.51, -4.06, 1.79, -0.92]),
        }
    )

    risk = (
        1.6 * (1 - df["Tuition fees up to date"])
        + 0.9 * df["Debtor"]
        - 0.9 * df["Scholarship holder"]
        + 0.06 * (df["Age at enrollment"] - 20)
        - 0.03 * (df["Admission grade"] - 127)
        + 0.35 * (1 - df["Daytime/evening attendance"])
        + 0.4 * df["Gender"]
        + rng.normal(0, 1.0, n)
    )
    # Semester results depend on risk; the outcome then depends on them, as in reality.
    enrolled = np.clip(rng.poisson(6, n), 0, 20)
    pass_rate = 1 / (1 + np.exp(1.2 * risk - 1.2 + rng.normal(0, 0.8, n)))
    approved1 = rng.binomial(enrolled, pass_rate)
    approved2 = rng.binomial(enrolled, np.clip(pass_rate + rng.normal(0, 0.1, n), 0, 1))
    grade1 = np.where(approved1 > 0, np.clip(rng.normal(10.5 + 2.5 * pass_rate, 1.2), 10, 18.9), 0)
    grade2 = np.where(approved2 > 0, np.clip(rng.normal(10.5 + 2.5 * pass_rate, 1.2), 10, 18.9), 0)

    progress = (approved1 + approved2) / np.maximum(2 * enrolled, 1)
    score = -1.3 * risk + 3.0 * (progress - 0.6) + rng.normal(0, 0.9, n)
    # Thresholds tuned to the real class balance (~32% dropout, 18% enrolled, 50% graduate).
    lo, hi = np.quantile(score, [0.32, 0.50])
    y = np.where(score < lo, 0, np.where(score < hi, 1, 2))

    if include_curricular:
        for sem, appr, grade in (("1st", approved1, grade1), ("2nd", approved2, grade2)):
            evals = enrolled + rng.poisson(2, n) * (enrolled > 0)
            df[f"{CURRICULAR_PREFIX} {sem} sem (credited)"] = rng.binomial(1, 0.1, n) * rng.poisson(
                3, n
            )
            df[f"{CURRICULAR_PREFIX} {sem} sem (enrolled)"] = enrolled
            df[f"{CURRICULAR_PREFIX} {sem} sem (evaluations)"] = evals
            df[f"{CURRICULAR_PREFIX} {sem} sem (approved)"] = appr
            df[f"{CURRICULAR_PREFIX} {sem} sem (grade)"] = np.round(grade, 2)
            df[f"{CURRICULAR_PREFIX} {sem} sem (without evaluations)"] = rng.binomial(1, 0.07, n)

    if missing_rate > 0:
        features = df.columns
        mask = rng.random((n, len(features))) < missing_rate
        df = df.mask(pd.DataFrame(mask, columns=features))

    df[TARGET] = y
    return df
