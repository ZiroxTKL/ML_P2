"""
Reducción de dimensionalidad (Sec 3.2) — PCA vs UMAP vs t-SNE.

Compara un método lineal (PCA) frente a dos de variedades múltiples (t-SNE, UMAP)
proyectando X ∈ ℝ⁶⁴ a 2D y 3D. Reporta:
  - tiempos de ejecución de cada método (rúbrica),
  - varianza retenida (PCA) y preservación de estructura local
    (trustworthiness) para los tres métodos,
  - separabilidad de clases vía kNN con validación cruzada en el espacio proyectado.

Salidas: figuras en figures/, métricas en results/, embeddings en results/embeddings.npz.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra proyección 3d)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import umap

import config
import data as D

config.apply_style()

# Color por especie, marcador por tipo de fauna (codifica taxonomía).
SPECIES_COLOR = {10: "#1f77b4", 12: "#ff7f0e", 17: "#2ca02c", 18: "#d62728", 23: "#9467bd"}
KIND_MARKER = {"Anfibio": "^", "Ave": "o"}


def _scatter2d(ax, emb, y_ids):
    for sid in config.SPECIES_IDS:
        m = y_ids == sid
        ax.scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.7,
                   c=SPECIES_COLOR[sid], marker=KIND_MARKER[config.SPECIES_KIND[sid]],
                   linewidths=0, label=config.short_label(sid))


def _scatter3d(ax, emb, y_ids):
    for sid in config.SPECIES_IDS:
        m = y_ids == sid
        ax.scatter(emb[m, 0], emb[m, 1], emb[m, 2], s=12, alpha=0.7,
                   c=SPECIES_COLOR[sid], marker=KIND_MARKER[config.SPECIES_KIND[sid]],
                   linewidths=0, label=config.short_label(sid))


def _knn_separability(emb, y_idx):
    """Accuracy de kNN (k=15) con 5-fold estratificado en el espacio proyectado."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=15), emb, y_idx, cv=cv)
    return scores.mean()


def main() -> None:
    df_tr, _ = D.load_raw()
    X = D.features(df_tr)
    y_ids = D.labels_id(df_tr)
    y_idx = D.labels_idx(df_tr)

    # Estandarización (ajustada en train) antes de toda proyección.
    Xs = StandardScaler().fit_transform(X)

    rows = []
    emb_store = {}

    # ---------------- PCA ----------------
    pca_full = PCA(random_state=config.RANDOM_STATE).fit(Xs)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)

    for ndim in (2, 3):
        t0 = time.perf_counter()
        emb = PCA(n_components=ndim, random_state=config.RANDOM_STATE).fit_transform(Xs)
        dt = time.perf_counter() - t0
        var_ret = float(pca_full.explained_variance_ratio_[:ndim].sum())
        emb_store[f"pca_{ndim}d"] = emb
        rows.append(dict(metodo="PCA", dims=ndim, tiempo_s=round(dt, 3),
                         var_retenida=round(var_ret, 4),
                         trustworthiness=round(trustworthiness(Xs, emb, n_neighbors=10), 4),
                         knn_cv_acc=round(_knn_separability(emb, y_idx), 4)))

    # ---------------- t-SNE ----------------
    for ndim in (2, 3):
        t0 = time.perf_counter()
        emb = TSNE(n_components=ndim, init="pca", learning_rate="auto",
                   perplexity=30, random_state=config.RANDOM_STATE).fit_transform(Xs)
        dt = time.perf_counter() - t0
        emb_store[f"tsne_{ndim}d"] = emb
        rows.append(dict(metodo="t-SNE", dims=ndim, tiempo_s=round(dt, 3),
                         var_retenida=np.nan,
                         trustworthiness=round(trustworthiness(Xs, emb, n_neighbors=10), 4),
                         knn_cv_acc=round(_knn_separability(emb, y_idx), 4)))

    # ---------------- UMAP ----------------
    for ndim in (2, 3):
        t0 = time.perf_counter()
        emb = umap.UMAP(n_components=ndim, n_neighbors=15, min_dist=0.1,
                        random_state=config.RANDOM_STATE).fit_transform(Xs)
        dt = time.perf_counter() - t0
        emb_store[f"umap_{ndim}d"] = emb
        rows.append(dict(metodo="UMAP", dims=ndim, tiempo_s=round(dt, 3),
                         var_retenida=np.nan,
                         trustworthiness=round(trustworthiness(Xs, emb, n_neighbors=10), 4),
                         knn_cv_acc=round(_knn_separability(emb, y_idx), 4)))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.RESULTS_DIR / "dimreduction_metrics.csv", index=False)
    np.savez_compressed(config.RESULTS_DIR / "embeddings.npz",
                        y_ids=y_ids, y_idx=y_idx, **emb_store)

    print("=== PCA: varianza acumulada ===")
    print(f"  componentes para 90 %: {n90} | para 95 %: {n95} (de 64)")
    print(f"  varianza retenida 2D: {pca_full.explained_variance_ratio_[:2].sum():.3f} | "
          f"3D: {pca_full.explained_variance_ratio_[:3].sum():.3f}")
    print("\n=== Métricas de reducción de dimensionalidad ===")
    print(metrics.to_string(index=False))

    # ================== FIGURAS ==================
    _fig_scree(pca_full, n90, n95)
    _fig_2d_panels(emb_store, y_ids, metrics)
    _fig_3d_panels(emb_store, y_ids)
    _fig_umap_is_tp(emb_store["umap_2d"], D.is_tp_mask(df_tr))

    with open(config.RESULTS_DIR / "dimreduction_metrics.md", "w", encoding="utf-8") as fh:
        fh.write("# Métricas de reducción de dimensionalidad (Sec 3.2)\n\n")
        fh.write(f"- PCA: {n90} componentes para 90 % de varianza, {n95} para 95 % (de 64).\n\n")
        try:
            fh.write(metrics.to_markdown(index=False))
        except ImportError:
            fh.write("```\n" + metrics.to_string(index=False) + "\n```")
    print("\n[ok] Sec 3.2 completada — figuras en", config.FIG_DIR)


def _fig_scree(pca_full, n90, n95):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    k = min(30, len(pca_full.explained_variance_ratio_))
    idx = np.arange(1, k + 1)
    ax.bar(idx, pca_full.explained_variance_ratio_[:k] * 100,
           color="#2a9d8f", label="Varianza individual")
    ax2 = ax.twinx()
    cum = np.cumsum(pca_full.explained_variance_ratio_)[:k] * 100
    ax2.plot(idx, cum, color="#264653", marker="o", ms=4, label="Varianza acumulada")
    ax2.axhline(90, ls="--", color="#e76f51")
    ax2.axvline(n90, ls=":", color="#e76f51")
    ax2.set_ylim(0, 100)
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Varianza individual (%)")
    ax2.set_ylabel("Varianza acumulada (%)")
    ax.set_title(f"PCA — varianza explicada (90 % con {n90} comp., 95 % con {n95})")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "dimred_pca_scree.png")
    plt.close(fig)


def _fig_2d_panels(emb_store, y_ids, metrics):
    methods = [("pca_2d", "PCA"), ("tsne_2d", "t-SNE"), ("umap_2d", "UMAP")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, (key, name) in zip(axes, methods):
        _scatter2d(ax, emb_store[key], y_ids)
        row = metrics[(metrics.metodo == name) & (metrics.dims == 2)].iloc[0]
        ax.set_title(f"{name} (2D) — {row.tiempo_s:.2f}s\n"
                     f"trust={row.trustworthiness:.3f}, kNN={row.knn_cv_acc:.3f}")
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=14,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Proyecciones 2D del espacio MFCC (X ∈ ℝ⁶⁴) por especie")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "dimred_2d_panels.png", bbox_inches="tight")
    plt.close(fig)


def _fig_3d_panels(emb_store, y_ids):
    methods = [("pca_3d", "PCA"), ("tsne_3d", "t-SNE"), ("umap_3d", "UMAP")]
    fig = plt.figure(figsize=(16, 5.6))
    for i, (key, name) in enumerate(methods, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        _scatter3d(ax, emb_store[key], y_ids)
        ax.set_title(f"{name} (3D)")
        ax.set_xlabel("C1"); ax.set_ylabel("C2"); ax.set_zlabel("C3")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=14,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Proyecciones 3D del espacio MFCC por especie")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "dimred_3d_panels.png", bbox_inches="tight")
    plt.close(fig)


def _fig_umap_is_tp(emb, tp_mask):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(emb[~tp_mask, 0], emb[~tp_mask, 1], s=12, alpha=0.4,
               c="#cccccc", label="is_tp = 0 (no verificado)")
    ax.scatter(emb[tp_mask, 0], emb[tp_mask, 1], s=18, alpha=0.85,
               c="#e76f51", label="is_tp = 1 (verificado)")
    ax.set_title("UMAP 2D coloreado por is_tp")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "dimred_umap_is_tp.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
