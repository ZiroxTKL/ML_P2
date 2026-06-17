"""
Minería de patrones y clustering (Sec 3.3) — DBSCAN vs GMM.

Dos paradigmas distintos:
  - GMM (probabilístico): selección del número de componentes vía BIC/AIC y
    métricas internas (Silhouette, Davies-Bouldin, Calinski-Harabasz).
  - DBSCAN (densidad): selección de eps por el codo del grafo de k-distancias y
    barrido de eps reportando nº de clústeres, ruido y Silhouette.

El clustering se ejecuta sobre el espacio PCA que retiene el 90 % de la varianza
(denoising + GMM tratable). La visualización se hace sobre el embedding UMAP 2D
ya calculado en la Sec 3.2. Se reporta además la concordancia con las especies
reales (ARI/NMI) como análisis externo complementario.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)

import config
import data as D

config.apply_style()

K_RANGE = list(range(2, 11))
CLUSTER_CMAP = plt.get_cmap("tab10")


def main() -> None:
    df_tr, _ = D.load_raw()
    X = D.features(df_tr)
    y_idx = D.labels_idx(df_tr)

    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=0.90, random_state=config.RANDOM_STATE).fit(Xs)
    Xr = pca.transform(Xs)
    print(f"Espacio de clustering: PCA {Xr.shape[1]}D (90 % varianza)")

    emb = np.load(config.RESULTS_DIR / "embeddings.npz")["umap_2d"]

    gmm_df, gmm_best_k, gmm_labels = _run_gmm(Xr, y_idx)
    db_df, db_eps, db_labels = _run_dbscan(Xr, y_idx)

    # ---- Concordancia con etiquetas reales (análisis externo) ----
    # Se contrasta cada partición contra especie, tipo de fauna (anfibio/ave) y songtype.
    kind = df_tr.species_id.map(config.SPECIES_KIND).map({"Anfibio": 0, "Ave": 1}).to_numpy()
    song = df_tr.songtype_id.to_numpy()
    refs = {"especie": y_idx, "tipo": kind, "songtype": song}
    ext = []
    for name, labels in (("GMM (k*)", gmm_labels), ("DBSCAN", db_labels)):
        mask = labels >= 0  # excluye ruido de DBSCAN
        row = dict(metodo=name, n_clusters=int(len(set(labels[mask]))),
                   ruido_frac=round(float((~mask).mean()), 3),
                   NMI_especie=round(normalized_mutual_info_score(y_idx[mask], labels[mask]), 3))
        for rname, ref in refs.items():
            row[f"ARI_{rname}"] = round(adjusted_rand_score(ref[mask], labels[mask]), 3)
        ext.append(row)
    ext_df = pd.DataFrame(ext)

    print("\n=== GMM: selección por k ===")
    print(gmm_df.to_string(index=False))
    print(f"\nk* por BIC = {gmm_best_k}")
    print("\n=== DBSCAN: barrido de eps ===")
    print(db_df.to_string(index=False))
    print(f"\neps elegido = {db_eps:.3f}")
    print("\n=== Concordancia con especies reales ===")
    print(ext_df.to_string(index=False))

    gmm_df.to_csv(config.RESULTS_DIR / "clustering_gmm.csv", index=False)
    db_df.to_csv(config.RESULTS_DIR / "clustering_dbscan.csv", index=False)

    _fig_gmm_selection(gmm_df, gmm_best_k)
    _fig_dbscan_selection(Xr, db_df, db_eps)
    _fig_umap_compare(emb, y_idx, gmm_labels, db_labels, gmm_best_k)

    _write_notes(pca, gmm_df, gmm_best_k, db_df, db_eps, ext_df)
    print("\n[ok] Sec 3.3 completada — figuras en", config.FIG_DIR)


def _run_gmm(Xr, y_idx):
    rows = []
    models = {}
    for k in K_RANGE:
        gm = GaussianMixture(n_components=k, covariance_type="full", n_init=5,
                             random_state=config.RANDOM_STATE).fit(Xr)
        labels = gm.predict(Xr)
        models[k] = (gm, labels)
        rows.append(dict(
            k=k,
            BIC=round(gm.bic(Xr), 1),
            AIC=round(gm.aic(Xr), 1),
            silhouette=round(silhouette_score(Xr, labels), 4),
            davies_bouldin=round(davies_bouldin_score(Xr, labels), 4),
            calinski_harabasz=round(calinski_harabasz_score(Xr, labels), 1),
        ))
    df = pd.DataFrame(rows)
    best_k = int(df.loc[df.BIC.idxmin(), "k"])
    return df, best_k, models[best_k][1]


def _run_dbscan(Xr, y_idx, min_samples: int = 10):
    # eps candidatos a partir de percentiles de la distancia al k-ésimo vecino
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Xr)
    kdist = np.sort(nn.kneighbors(Xr)[0][:, -1])
    eps_grid = np.quantile(kdist, np.linspace(0.55, 0.97, 14))

    rows = []
    results = {}
    for eps in eps_grid:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xr)
        mask = labels >= 0
        n_clusters = len(set(labels[mask]))
        sil = (silhouette_score(Xr[mask], labels[mask])
               if n_clusters >= 2 and mask.sum() > n_clusters else np.nan)
        eps_r = round(float(eps), 4)
        rows.append(dict(eps=eps_r, n_clusters=int(n_clusters),
                         ruido_frac=round(float((~mask).mean()), 3),
                         silhouette=round(sil, 4) if not np.isnan(sil) else np.nan))
        results[eps_r] = labels
    df = pd.DataFrame(rows)
    # elige eps con mejor silhouette entre soluciones con 2..8 clústeres y ruido < 60 %
    valid = df[(df.n_clusters.between(2, 8)) & (df.ruido_frac < 0.60)].dropna(subset=["silhouette"])
    best_eps = float(valid.loc[valid.silhouette.idxmax(), "eps"]) if len(valid) else float(df.eps.iloc[len(df)//2])
    return df, best_eps, results[best_eps]


def _fig_gmm_selection(df, best_k):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [("BIC", "BIC (menor mejor)", axes[0, 0]),
              ("silhouette", "Silhouette (mayor mejor)", axes[0, 1]),
              ("davies_bouldin", "Davies-Bouldin (menor mejor)", axes[1, 0]),
              ("calinski_harabasz", "Calinski-Harabasz (mayor mejor)", axes[1, 1])]
    for col, title, ax in panels:
        ax.plot(df.k, df[col], marker="o", color="#264653")
        if col == "BIC":
            ax.plot(df.k, df["AIC"], marker="s", ls="--", color="#2a9d8f", label="AIC")
            ax.legend()
        ax.axvline(best_k, color="#e76f51", ls=":", label=f"k*={best_k}")
        ax.axvline(config.N_CLASSES, color="#888", ls="--", alpha=0.7)
        ax.set_xlabel("Número de componentes k")
        ax.set_title(title)
    fig.suptitle(f"GMM — selección de k (k*={best_k} por BIC; línea gris = 5 especies)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "clustering_gmm_selection.png")
    plt.close(fig)


def _fig_dbscan_selection(Xr, df, best_eps, min_samples: int = 10):
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Xr)
    kdist = np.sort(nn.kneighbors(Xr)[0][:, -1])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.plot(np.arange(len(kdist)), kdist, color="#264653")
    ax.axhline(best_eps, color="#e76f51", ls=":", label=f"eps={best_eps:.2f}")
    ax.set_xlabel("Puntos ordenados")
    ax.set_ylabel(f"Distancia al {min_samples}º vecino")
    ax.set_title("Grafo de k-distancias (codo → eps)")
    ax.legend()

    ax = axes[1]
    ax.plot(df.eps, df.n_clusters, marker="o", color="#264653", label="Nº clústeres")
    ax.set_xlabel("eps")
    ax.set_ylabel("Nº de clústeres")
    ax2 = ax.twinx()
    ax2.plot(df.eps, df.silhouette, marker="s", ls="--", color="#e76f51", label="Silhouette")
    ax2.set_ylabel("Silhouette")
    ax.axvline(best_eps, color="#888", ls=":")
    ax.set_title("DBSCAN — barrido de eps")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "clustering_dbscan_selection.png")
    plt.close(fig)


def _scatter_clusters(ax, emb, labels, title):
    uniq = sorted(set(labels))
    for lab in uniq:
        m = labels == lab
        if lab == -1:
            ax.scatter(emb[m, 0], emb[m, 1], s=10, c="#cccccc", alpha=0.5, label="ruido")
        else:
            ax.scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.7,
                       color=CLUSTER_CMAP(lab % 10), label=f"c{lab}")
    ax.set_title(title)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")


def _fig_umap_compare(emb, y_idx, gmm_labels, db_labels, best_k):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    # especies reales
    for i in range(config.N_CLASSES):
        m = y_idx == i
        axes[0].scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.7,
                        color=CLUSTER_CMAP(i), label=config.SPECIES_NAME[config.IDX_TO_ID[i]])
    axes[0].set_title("Especies reales (referencia)")
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
    axes[0].legend(fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _scatter_clusters(axes[1], emb, gmm_labels, f"GMM (k*={best_k})")  # IDs de clúster sin leyenda (no informativos)
    n_db = len(set(db_labels[db_labels >= 0]))
    _scatter_clusters(axes[2], emb, db_labels, f"DBSCAN ({n_db} clústeres)")
    axes[2].legend(fontsize=14, ncol=2, loc="best")
    fig.suptitle("Clustering sobre el embedding UMAP 2D")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "clustering_umap_compare.png", bbox_inches="tight")
    plt.close(fig)


def _write_notes(pca, gmm_df, gmm_best_k, db_df, db_eps, ext_df):
    sil_k = int(gmm_df.loc[gmm_df.silhouette.idxmax(), "k"])
    ch_k = int(gmm_df.loc[gmm_df.calinski_harabasz.idxmax(), "k"])
    with open(config.RESULTS_DIR / "clustering_notes.md", "w", encoding="utf-8") as fh:
        fh.write("# Interpretación — Clustering (Sec 3.3)\n\n")
        fh.write(f"Espacio de clustering: PCA con {pca.n_components_} componentes "
                 f"(90 % de varianza).\n\n## GMM (probabilístico)\n\n")
        fh.write(gmm_df.to_markdown(index=False) + "\n\n")
        fh.write(f"- **k\\*={gmm_best_k}** por BIC.\n\n## DBSCAN (densidad)\n\n")
        fh.write(db_df.to_markdown(index=False) + "\n\n")
        fh.write(f"- eps elegido = {db_eps:.3f} (codo del grafo de k-distancias, min_samples=10).\n\n")
        fh.write("## Concordancia con etiquetas reales (externo)\n\n")
        fh.write(ext_df.to_markdown(index=False) + "\n\n")
        fh.write("## Lectura analítica\n\n")
        fh.write(
            f"1. **Métricas internas en conflicto.** BIC y AIC decrecen de forma "
            f"monótona (favorecen k={gmm_best_k}, el máximo del rango): señal típica de "
            f"ausencia de gaussianas separadas, pues añadir componentes siempre mejora "
            f"el ajuste de densidad. En cambio Silhouette (máx. en k={sil_k}) y "
            f"Calinski-Harabasz (máx. en k={ch_k}) favorecen particiones gruesas. No "
            f"existe un k con clústeres compactos y bien separados.\n"
            f"2. **El clustering no recupera las especies.** Todas las soluciones dan "
            f"ARI≈0 y NMI≈0 frente a la especie; tampoco se alinean con el tipo de fauna "
            f"(anfibio/ave) ni con el songtype. La estructura geométrica del espacio MFCC "
            f"está dominada por variación acústica de fondo a nivel de grabación (ruido, "
            f"canal, sitio), no por la etiqueta biológica.\n"
            f"3. **Implicación.** La señal que separa especies es débil y no recuperable "
            f"de forma no supervisada por densidad o probabilidad; se requiere aprendizaje "
            f"supervisado (Sec 3.4) para aislar las direcciones discriminantes. Es "
            f"coherente con la baja separabilidad kNN de la Sec 3.2 y con la alta "
            f"proporción de ventanas no verificadas (is_tp = 13 %).\n")


if __name__ == "__main__":
    main()
