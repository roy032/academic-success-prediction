"""End-to-end training: split, cross-validated model selection, one test evaluation."""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from . import __version__, plots
from .data import CLASS_NAMES, TARGET, load_dataset, make_synthetic
from .evaluate import bootstrap_ci, classification_metrics, cross_validate, text_report
from .pipeline import DISPLAY_NAMES, MODELS, build_model, make_preprocessor, outlier_mask

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    data: str | None = None  # CSV path; None means synthetic data
    out: str = "artifacts"
    include_curricular: bool = False
    models: list[str] = field(default_factory=lambda: list(MODELS))
    test_size: float = 0.2
    cv_folds: int = 5
    z_limit: float | None = 4.0
    seed: int = 42
    n_boot: int = 1000
    synthetic_rows: int = 4424


def load(cfg: TrainConfig) -> pd.DataFrame:
    if cfg.data:
        return load_dataset(cfg.data, include_curricular=cfg.include_curricular)
    log.warning("no --data given: using synthetic data with the UCI schema")
    return make_synthetic(cfg.synthetic_rows, cfg.seed, include_curricular=cfg.include_curricular)


def cluster_analysis(
    X: pd.DataFrame, y: pd.Series, seed: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    Z = make_preprocessor(X.columns).fit_transform(X)
    inertia, sil = {}, {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z)
        inertia[k] = float(km.inertia_)
        sil[k] = float(
            silhouette_score(Z, km.labels_, sample_size=min(2000, len(Z)), random_state=seed)
        )
    km = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(Z)
    crosstab = pd.crosstab(km.labels_, y.to_numpy())
    crosstab.columns = [CLASS_NAMES[c] for c in crosstab.columns]
    points = PCA(n_components=2, random_state=seed).fit_transform(Z)
    summary = {
        "inertia": inertia,
        "silhouette": sil,
        "cluster_vs_target": {str(k): v for k, v in crosstab.to_dict("index").items()},
    }
    return summary, points, km.labels_


def run(cfg: TrainConfig) -> dict:
    t0 = time.perf_counter()
    out = Path(cfg.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load(cfg)
    X, y = df.drop(columns=TARGET), df[TARGET]
    # The test split is carved off first and touched exactly once, at the end.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    log.info("rows: dev=%d test=%d features=%d", len(X_dev), len(X_test), X.shape[1])

    cv = {}
    for name in cfg.models:
        cv[name] = cross_validate(
            build_model(name, X.columns, cfg.seed),
            X_dev,
            y_dev,
            cfg.cv_folds,
            cfg.seed,
            cfg.z_limit,
        )
        log.info(
            "cv %-18s macro-F1 %.3f ± %.3f",
            name,
            cv[name]["macro_f1_mean"],
            cv[name]["macro_f1_std"],
        )
    best = max(cv, key=lambda m: cv[m]["macro_f1_mean"])

    keep = outlier_mask(X_dev, cfg.z_limit)
    test = {}
    fitted = {}
    for name in cfg.models:
        model = build_model(name, X.columns, cfg.seed).fit(X_dev[keep], y_dev[keep])
        fitted[name] = model
        proba = model.predict_proba(X_test)
        test[name] = classification_metrics(y_test, proba.argmax(axis=1), proba)

    model = fitted[best]
    proba = model.predict_proba(X_test)
    pred = proba.argmax(axis=1)
    ci = bootstrap_ci(y_test, pred, cfg.n_boot, cfg.seed)
    log.info(
        "selected %s; test macro-F1 %.3f (95%% CI %s)", best, test[best]["macro_f1"], ci["macro_f1"]
    )

    perm = permutation_importance(
        model, X_test, y_test, scoring="f1_macro", n_repeats=10, random_state=cfg.seed, n_jobs=-1
    )
    importance = sorted(
        zip(X.columns, perm.importances_mean, perm.importances_std, strict=True),
        key=lambda r: -r[1],
    )
    clusters, pca_points, cluster_labels = cluster_analysis(X_dev, y_dev, cfg.seed)

    title = DISPLAY_NAMES[best]
    figures = [
        plots.model_comparison(cv, DISPLAY_NAMES, best, fig_dir / "model_comparison.png"),
        plots.confusion(
            test[best]["confusion_matrix"], f"{title}: test set", fig_dir / "confusion_matrix.png"
        ),
        plots.roc_curves(y_test, proba, f"{title}: one-vs-rest ROC", fig_dir / "roc_curves.png"),
        plots.importance(
            [r[0] for r in importance],
            [r[1] for r in importance],
            [r[2] for r in importance],
            fig_dir / "permutation_importance.png",
        ),
        plots.clustering(
            clusters["inertia"],
            clusters["silhouette"],
            pca_points,
            cluster_labels,
            fig_dir / "clustering.png",
        ),
    ]

    metadata = {
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": cfg.data or "synthetic",
        "rows": {
            "dev": int(len(X_dev)),
            "dev_after_outliers": int(keep.sum()),
            "test": int(len(X_test)),
        },
        "features": list(X.columns),
        "class_names": CLASS_NAMES,
        "selected_model": best,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "config": asdict(cfg),
    }
    joblib.dump({"pipeline": model, "metadata": metadata}, out / "model.joblib")

    report = {
        "metadata": metadata,
        "cv": cv,
        "test": test,
        "selected": {"model": best, "test_bootstrap_ci95": ci, "report": text_report(y_test, pred)},
        "permutation_importance": [
            {"feature": f, "mean": float(m), "std": float(s)} for f, m, s in importance
        ],
        "clustering": clusters,
        "figures": [str(p.relative_to(out)) for p in figures],
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    (out / "results.md").write_text(results_table(report))
    log.info("artifacts written to %s in %.1fs", out, report["runtime_seconds"])
    return report


def results_table(report: dict) -> str:
    rows = [
        "| Model | CV macro-F1 (5-fold) | Test acc | Test macro-F1 | Test ROC-AUC |",
        "|---|---|---|---|---|",
    ]
    best = report["selected"]["model"]
    for name, c in sorted(report["cv"].items(), key=lambda kv: -kv[1]["macro_f1_mean"]):
        t = report["test"][name]
        label = f"**{DISPLAY_NAMES[name]}** (selected)" if name == best else DISPLAY_NAMES[name]
        rows.append(
            f"| {label} | {c['macro_f1_mean']:.3f} ± {c['macro_f1_std']:.3f} | "
            f"{t['accuracy']:.3f} | {t['macro_f1']:.3f} | {t['roc_auc_ovr']:.3f} |"
        )
    lo, hi = report["selected"]["test_bootstrap_ci95"]["macro_f1"]
    rows.append(f"\nSelected model test macro-F1 95% bootstrap CI: [{lo:.3f}, {hi:.3f}]")
    return "\n".join(rows) + "\n"
