"""Leakage-free training pipeline for the academic success dataset.

Every data-dependent step (imputation, outlier removal, scaling) is fit on the
training split only, then applied unchanged to validation and test data.

Usage:
    python train.py --data academic_success_dataset.csv
    python train.py --synthetic            # smoke test without the real dataset
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET_MAP = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
CLASS_NAMES = list(TARGET_MAP)
Z_LIMIT = 4.0


def load_real(path, include_curricular=False):
    # The UCI file is semicolon-delimited with a BOM; the course variant is comma-delimited.
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if not include_curricular:
        # The course variant has no semester-grade columns; dropping them keeps results
        # comparable to the original notebook and models prediction at enrollment time.
        df = df.loc[:, ~df.columns.str.startswith("Curricular units")]
    df = df.dropna(subset=["Target"])
    df["Target"] = df["Target"].map(TARGET_MAP)
    return df.dropna(subset=["Target"]).astype({"Target": int})


def load_synthetic(n=1500):
    """Random data with the real schema's shape and ~10% missing values."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(n, 24))
    y = rng.integers(0, 3, size=n)
    X[:, :4] += y[:, None] * 0.8
    X[rng.random(X.shape) < 0.1] = np.nan
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(24)])
    df["Target"] = y
    return df


def split(df):
    """64 / 16 / 20 stratified train / val / test split, done before any fitting."""
    X, y = df.drop(columns="Target"), df["Target"]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.2, random_state=SEED, stratify=y_tmp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def drop_train_outliers(X_train, y_train):
    """Remove training rows with |z| > Z_LIMIT, using training statistics only.

    Validation and test rows are never dropped, so evaluation covers every
    sample the model would see in practice.
    """
    imputed = SimpleImputer(strategy="median").fit_transform(X_train)
    std = imputed.std(axis=0)
    std[std == 0] = 1.0
    z = np.abs((imputed - imputed.mean(axis=0)) / std)
    keep = (z <= Z_LIMIT).all(axis=1)
    return X_train[keep], y_train[keep], int((~keep).sum())


def models():
    prep = lambda: [SimpleImputer(strategy="median"), StandardScaler()]
    return {
        "Logistic Regression": make_pipeline(
            *prep(), LogisticRegression(max_iter=1000, random_state=SEED)),
        "Logistic Regression (balanced)": make_pipeline(
            *prep(), LogisticRegression(max_iter=1000, class_weight="balanced",
                                        random_state=SEED)),
        "Naive Bayes": make_pipeline(*prep(), GaussianNB()),
        "Neural Network (MLP)": make_pipeline(
            *prep(), MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                                   max_iter=200, random_state=SEED)),
    }


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "per_class_f1": dict(zip(CLASS_NAMES, f1_score(y, pred, average=None, labels=[0, 1, 2]))),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "report": classification_report(y, pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                                        zero_division=0),
    }


def cluster_analysis(X_train, y_train, out_dir):
    """KMeans on train features (imputed + scaled with train statistics)."""
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Z = prep.fit_transform(X_train)
    inertia, sil = {}, {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Z)
        inertia[k] = float(km.inertia_)
        sil[k] = float(silhouette_score(Z, km.labels_, sample_size=2000, random_state=SEED))
    km = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit(Z)
    purity = pd.crosstab(km.labels_, y_train.values)
    purity.columns = CLASS_NAMES
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(list(inertia), list(inertia.values()), marker="o")
        ax[0].set(title="Elbow (inertia)", xlabel="k")
        ax[1].plot(list(sil), list(sil.values()), marker="o")
        ax[1].set(title="Silhouette", xlabel="k")
        P = PCA(n_components=2, random_state=SEED).fit_transform(Z)
        ax[2].scatter(P[:, 0], P[:, 1], c=km.labels_, s=6, cmap="viridis")
        ax[2].set(title="KMeans (k=3) in PCA space")
        fig.tight_layout()
        fig.savefig(out_dir / "clustering.png", dpi=120)
    except ImportError:
        print("matplotlib not installed; skipping clustering.png")
    return {"inertia": inertia, "silhouette": sil, "cluster_vs_target": purity.to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="academic_success_dataset.csv")
    ap.add_argument("--include-curricular", action="store_true",
                    help="keep 'Curricular units ...' semester columns (UCI file only)")
    ap.add_argument("--synthetic", action="store_true", help="use random data (smoke test)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    df = load_synthetic() if args.synthetic else load_real(args.data, args.include_curricular)
    X_train, X_val, X_test, y_train, y_val, y_test = split(df)
    X_train, y_train, n_dropped = drop_train_outliers(X_train, y_train)
    print(f"train={len(X_train)} (dropped {n_dropped} outliers) val={len(X_val)} test={len(X_test)}")

    results = {}
    for name, model in models().items():
        model.fit(X_train, y_train)
        results[name] = {"val": evaluate(model, X_val, y_val),
                         "test": evaluate(model, X_test, y_test)}
        print(f"\n=== {name} ===")
        print(f"val  acc={results[name]['val']['accuracy']:.4f} "
              f"macro-F1={results[name]['val']['macro_f1']:.4f}")
        print(f"test acc={results[name]['test']['accuracy']:.4f} "
              f"macro-F1={results[name]['test']['macro_f1']:.4f}")
        print(results[name]["test"]["report"])

    best = max(results, key=lambda m: results[m]["val"]["macro_f1"])
    print(f"Best model by validation macro-F1: {best}")

    clusters = cluster_analysis(X_train, y_train, out_dir)
    for r in results.values():
        for split_name in ("val", "test"):
            r[split_name].pop("report")
    payload = {"outliers_dropped_from_train": n_dropped, "selected_by_val_macro_f1": best,
               "models": results, "clustering": clusters}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=float))

    rows = ["| Model | Val acc | Val macro-F1 | Test acc | Test macro-F1 |", "|---|---|---|---|---|"]
    for name, r in results.items():
        rows.append(f"| {name} | {r['val']['accuracy']:.3f} | {r['val']['macro_f1']:.3f} | "
                    f"{r['test']['accuracy']:.3f} | {r['test']['macro_f1']:.3f} |")
    (out_dir / "results.md").write_text("\n".join(rows) + "\n")
    print("\n" + "\n".join(rows))


if __name__ == "__main__":
    main()
