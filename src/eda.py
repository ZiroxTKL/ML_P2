"""
EDA y análisis del espacio vectorial (Sec 3.1).

Genera las figuras base y un resumen estadístico del dataset eco-acústico:
distribución de clases (desbalance), análisis del indicador is_tp, estadística
de los 64 coeficientes MFCC y estructura de correlación. Todas las figuras
respetan fuente >= 14 (config.apply_style).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
import data as D

config.apply_style()

KIND_COLOR = {"Anfibio": "#2a9d8f", "Ave": "#e76f51"}


def main() -> None:
    df_tr, df_te = D.load_raw()
    out = []

    def log(msg: str = "") -> None:
        print(msg)
        out.append(str(msg))

    # ---------------- Resumen estructural ----------------
    log("=" * 60)
    log("EDA — Dataset eco-acústico (Proyecto 2, CS3061)")
    log("=" * 60)
    log(f"train: {df_tr.shape[0]} obs x {df_tr.shape[1]} cols")
    log(f"test : {df_te.shape[0]} obs x {df_te.shape[1]} cols")
    log(f"Espacio de características: X in R^{config.N_FEATURES} "
        f"({config.FEATURE_COLS[0]} .. {config.FEATURE_COLS[-1]})")
    log(f"Valores faltantes (train): {int(df_tr.isna().sum().sum())}")
    log(f"Valores faltantes (test) : {int(df_te.isna().sum().sum())}")
    log(f"recording_id duplicados (train): {int(df_tr[config.ID_COL].duplicated().sum())}")

    # ---------------- Distribución de clases ----------------
    log("\n--- Distribución de species_id (train) ---")
    vc = df_tr[config.TARGET].value_counts().sort_index()
    for sid, n in vc.items():
        log(f"  {sid:>2} {config.SPECIES_NAME[sid]:<32} "
            f"({config.SPECIES_KIND[sid]:<8}): {n:>4}  ({100*n/len(df_tr):4.1f} %)")
    imbalance = vc.max() / vc.min()
    log(f"  Ratio de desbalance (max/min): {imbalance:.2f}")

    # ---------------- Análisis de is_tp ----------------
    log("\n--- Indicador is_tp (True Positive verificado) ---")
    tp_rate = df_tr["is_tp"].mean()
    log(f"  Tasa global de TP (train): {tp_rate:.3f}  "
        f"({int(df_tr['is_tp'].sum())} de {len(df_tr)})")
    tp_by_sp = df_tr.groupby(config.TARGET)["is_tp"].agg(["mean", "sum", "count"])
    for sid, row in tp_by_sp.iterrows():
        log(f"  {sid:>2} {config.SPECIES_NAME[sid]:<32}: "
            f"TP={int(row['sum']):>3}/{int(row['count']):>3}  ({row['mean']:.3f})")

    log("\n--- songtype_id (train) ---")
    for st, n in df_tr["songtype_id"].value_counts().sort_index().items():
        log(f"  songtype {st}: {n}")

    # ---------------- Estadística MFCC ----------------
    X = D.features(df_tr)
    log("\n--- Coeficientes MFCC (train) ---")
    log(f"  rango global: [{X.min():.3f}, {X.max():.3f}]")
    log(f"  media global: {X.mean():.3f} | std global: {X.std():.3f}")

    with open(config.RESULTS_DIR / "eda_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(f"\n[guardado] {config.RESULTS_DIR / 'eda_summary.txt'}")

    # ================== FIGURAS ==================
    _fig_class_distribution(df_tr, df_te)
    _fig_is_tp(df_tr, tp_by_sp)
    _fig_mfcc_overview(X)
    _fig_correlation(X)
    print("[ok] EDA completado — figuras en", config.FIG_DIR)


def _ordered_ids() -> list[int]:
    return config.SPECIES_IDS


def _fig_class_distribution(df_tr: pd.DataFrame, df_te: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, df, title in ((axes[0], df_tr, "Entrenamiento (n=1906)"),
                          (axes[1], df_te, "Prueba (n=477)")):
        vc = df[config.TARGET].value_counts().reindex(config.SPECIES_IDS)
        labels = [f"{config.SPECIES_NAME[s]}\n({config.SPECIES_KIND[s]}, id={s})"
                  for s in config.SPECIES_IDS]
        colors = [KIND_COLOR[config.SPECIES_KIND[s]] for s in config.SPECIES_IDS]
        ax.barh(labels, vc.values, color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("Número de registros")
        ax.set_title(title)
        for y, v in enumerate(vc.values):
            ax.text(v + max(vc.values) * 0.01, y, str(int(v)), va="center", fontsize=14)
    fig.suptitle("Distribución de clases (especies) — desbalance asimétrico")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_class_distribution.png")
    plt.close(fig)


def _fig_is_tp(df_tr: pd.DataFrame, tp_by_sp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ids = config.SPECIES_IDS
    names = [config.SPECIES_NAME[s] for s in ids]
    tp = tp_by_sp.reindex(ids)["sum"].values
    total = tp_by_sp.reindex(ids)["count"].values
    non_tp = total - tp

    ax = axes[0]
    ax.bar(range(len(ids)), tp, label="is_tp = 1 (verificado)", color="#2a9d8f")
    ax.bar(range(len(ids)), non_tp, bottom=tp, label="is_tp = 0 (no verificado)",
           color="#cccccc")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([f"{n}\n(id={s})" for n, s in zip(names, ids)],
                       rotation=30, ha="right", fontsize=14)
    ax.set_ylabel("Registros")
    ax.set_title("Composición is_tp por especie")
    ax.legend()

    ax = axes[1]
    rate = tp_by_sp.reindex(ids)["mean"].values
    bars = ax.bar(range(len(ids)), rate, color="#264653")
    ax.axhline(df_tr["is_tp"].mean(), ls="--", color="#e76f51",
               label=f"Tasa global = {df_tr['is_tp'].mean():.2f}")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([f"id={s}" for s in ids], fontsize=14)
    ax.set_ylabel("Tasa de True Positive")
    ax.set_title("Fiabilidad (tasa is_tp) por especie")
    ax.set_ylim(0, max(rate) * 1.25)
    for b, r in zip(bars, rate):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.01, f"{r:.2f}",
                ha="center", fontsize=14)
    ax.legend()
    fig.suptitle("Indicador is_tp — solo ~13 % de registros verificados (ruido ambiental)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_is_tp.png")
    plt.close(fig)


def _fig_mfcc_overview(X: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    idx = np.arange(X.shape[1])
    ax.plot(idx, mean, color="#264653", label="media")
    ax.fill_between(idx, mean - std, mean + std, alpha=0.3, color="#2a9d8f",
                    label="± 1 desv. est.")
    ax.set_xlabel("Índice del coeficiente MFCC (0–63)")
    ax.set_ylabel("Valor")
    ax.set_title("Perfil de los 64 coeficientes MFCC")
    ax.legend()

    ax = axes[1]
    ax.hist(X.ravel(), bins=80, color="#2a9d8f", edgecolor="white")
    ax.set_xlabel("Valor del coeficiente")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución global de los valores MFCC")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_mfcc_overview.png")
    plt.close(fig)


def _fig_correlation(X: np.ndarray) -> None:
    corr = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"label": "Correlación de Pearson"}, ax=ax)
    ax.set_title("Matriz de correlación de los 64 MFCC")
    ax.set_xlabel("Índice MFCC")
    ax.set_ylabel("Índice MFCC")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_correlation.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
