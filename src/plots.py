"""Matplotlib plots required as evidence (headless backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_residuals(y_true, y_pred, path: Path, code: str, model_name: str) -> None:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    resid = y_pred - y_true
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].scatter(y_true, y_pred, s=18, alpha=0.7)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax[0].plot(lim, lim, "r--", lw=1)
    ax[0].set(xlabel="Actual yield (t)", ylabel="Predicted yield (t)", title="Actual vs predicted")
    ax[1].scatter(y_pred, resid, s=18, alpha=0.7)
    ax[1].axhline(0, color="r", ls="--", lw=1)
    ax[1].set(xlabel="Predicted yield (t)", ylabel="Residual (pred - actual, t)", title="Residuals vs predicted")
    ax[2].hist(resid, bins=15, edgecolor="white")
    ax[2].axvline(0, color="r", ls="--", lw=1)
    ax[2].set(xlabel="Residual (t)", ylabel="Farms", title="Residual distribution")
    fig.suptitle(f"Regression residuals - {model_name} - {code}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(matrix, path: Path, code: str, model_name: str) -> None:
    m = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black", fontsize=16)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["not high", "high"], yticklabels=["not high", "high"],
           xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix - {model_name}\n{code}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_clusters(Z_train, labels_train, Z_test, labels_test, centers, path: Path, code: str, k: int) -> None:
    pca = PCA(n_components=2, random_state=0).fit(Z_train)
    P_tr, P_te, P_c = pca.transform(Z_train), pca.transform(Z_test), pca.transform(centers)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    cmap = plt.get_cmap("tab10")
    for c in range(k):
        ax.scatter(*P_tr[labels_train == c].T, s=22, color=cmap(c), alpha=0.75, label=f"Cluster {c} (train)")
        if (labels_test == c).any():
            ax.scatter(*P_te[labels_test == c].T, s=40, color=cmap(c), edgecolor="black", marker="s")
    ax.scatter(*P_c.T, marker="X", s=180, color="black", label="Centres")
    var = pca.explained_variance_ratio_.sum() * 100
    ax.set(xlabel="PC1", ylabel="PC2", title=f"K-means clusters (K={k}) - {code}\nsquares = test farms; PCA keeps {var:.0f}% of variance")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
