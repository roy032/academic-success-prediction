"""Figures written next to the metrics. Matplotlib only, headless."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import auc, roc_curve  # noqa: E402

from .data import CLASS_NAMES  # noqa: E402

COLORS = ["#c2410c", "#a16207", "#15803d"]  # dropout, enrolled, graduate
ACCENT = "#1d4ed8"


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def model_comparison(cv: dict[str, dict], names: dict[str, str], best: str, path: Path) -> Path:
    order = sorted(cv, key=lambda m: cv[m]["macro_f1_mean"])
    means = [cv[m]["macro_f1_mean"] for m in order]
    stds = [cv[m]["macro_f1_std"] for m in order]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [ACCENT if m == best else "#94a3b8" for m in order]
    ax.barh([names[m] for m in order], means, xerr=stds, color=colors, capsize=4)
    for i, (m, s) in enumerate(zip(means, stds, strict=True)):
        ax.text(m + s + 0.005, i, f"{m:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Cross-validated macro-F1 (mean ± 1 sd)")
    ax.set_xlim(0, min(1.0, max(means) + max(stds) + 0.1))
    ax.set_title("Model selection on the development set")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def confusion(cm, title: str, path: Path) -> Path:
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}\n({int(cm[i, j])})",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.55 else "black",
                fontsize=9,
            )
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return _save(fig, path)


def roc_curves(y_true, proba, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    y_true = np.asarray(y_true)
    for k, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true == k, proba[:, k])
        ax.plot(fpr, tpr, color=COLORS[k], label=f"{name} (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#94a3b8", lw=1)
    ax.set(title=title, xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


def importance(names, means, stds, path: Path, top: int = 15) -> Path:
    order = np.argsort(means)[-top:]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(order) + 1))
    ax.barh(
        np.asarray(names)[order],
        np.asarray(means)[order],
        xerr=np.asarray(stds)[order],
        color=ACCENT,
        capsize=3,
    )
    ax.set_xlabel("Drop in test macro-F1 when shuffled")
    ax.set_title("Permutation importance (best model)")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def clustering(inertia: dict, silhouette: dict, pca_points, labels, path: Path) -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(list(inertia), list(inertia.values()), marker="o", color=ACCENT)
    ax[0].set(title="Elbow (inertia)", xlabel="k")
    ax[1].plot(list(silhouette), list(silhouette.values()), marker="o", color=ACCENT)
    ax[1].set(title="Silhouette", xlabel="k")
    ax[2].scatter(pca_points[:, 0], pca_points[:, 1], c=labels, s=5, cmap="viridis")
    ax[2].set(title="KMeans (k=3) in PCA space", xlabel="PC1", ylabel="PC2")
    return _save(fig, path)
