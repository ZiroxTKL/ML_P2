"""
Políticas de moderación por umbrales y trade-offs (Sec 3.5).

Sobre el vector de probabilidad del mejor clasificador se aplican tres zonas
operativas (confianza máxima P = max_k p_k):
  - Confianza   (P ≥ 0.85): clasificación automática.
  - Incertidumbre (0.40 ≤ P < 0.85): auditoría humana.
  - Rechazo     (P < 0.40): descarte (ruido ambiental).

Análisis:
  - cobertura/auditoría/descarte y exactitud por zona (clasificación selectiva),
  - curva exactitud-cobertura,
  - relación entre confianza del modelo e is_tp (fiabilidad de la etiqueta),
  - trade-off costo computacional: tiempos de inferencia por modelo.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import accuracy_score, f1_score

import config
import data as D

config.apply_style()

GREEN, YELLOW, RED = "#2e7d32", "#f9a825", "#c62828"


def zone_of(pmax):
    z = np.full(len(pmax), 1)  # 1 = incertidumbre
    z[pmax >= config.THRESH_CONFIDENCE] = 0  # 0 = confianza
    z[pmax < config.THRESH_REJECT] = 2        # 2 = rechazo
    return z


def main():
    d = np.load(config.RESULTS_DIR / "test_predictions.npz", allow_pickle=True)
    y = d["y_true"]
    is_tp = d["is_tp"].astype(bool)
    best = str(d["best_model"])
    proba_map = {"MLP": d["proba_mlp"], "XGBoost": d["proba_xgb"], "LightGBM": d["proba_lgbm"]}
    proba = proba_map[best]
    pred = proba.argmax(1)
    pmax = proba.max(1)
    print(f"Modelo base de la moderación: {best}")

    zones = zone_of(pmax)
    names = ["Confianza (≥85 %)", "Incertidumbre (40–85 %)", "Rechazo (<40 %)"]
    colors = [GREEN, YELLOW, RED]

    # ---- Métricas por zona ----
    rows = []
    for z, nm in enumerate(names):
        m = zones == z
        n = int(m.sum())
        acc = accuracy_score(y[m], pred[m]) if n else float("nan")
        rows.append(dict(zona=nm, n=n, porcentaje=round(100 * n / len(y), 1),
                         exactitud=round(acc, 4) if n else np.nan,
                         tasa_is_tp=round(float(is_tp[m].mean()), 4) if n else np.nan))
    zone_df = pd.DataFrame(rows)
    zone_df.to_csv(config.RESULTS_DIR / "threshold_zones.csv", index=False)

    overall_acc = accuracy_score(y, pred)
    auto = zones == 0
    print("\n=== Zonas de moderación ===")
    print(zone_df.to_string(index=False))
    print(f"\nExactitud global: {overall_acc:.4f}")
    if auto.sum():
        print(f"Exactitud en zona de confianza (auto): {accuracy_score(y[auto], pred[auto]):.4f} "
              f"sobre {auto.mean()*100:.1f} % de cobertura")

    # ---- Tiempos de inferencia (trade-off costo) ----
    timing = _inference_timing()
    print("\n=== Tiempos de inferencia (ms por 1000 muestras) ===")
    print(timing.to_string(index=False))

    # ---- Figuras ----
    _fig_zones(zone_df, colors, overall_acc)
    _fig_selective(pmax, pred, y, zones, colors, names)
    _fig_inference(timing)
    _write_notes(best, zone_df, overall_acc, auto, y, pred, is_tp, zones, timing)
    print("\n[ok] Sec 3.5 completada — figuras en", config.FIG_DIR)


def _inference_timing(reps: int = 5):
    df_te = D.load_raw()[1]
    Xraw = D.features(df_te)
    scaler = joblib.load(config.RESULTS_DIR / "scaler.joblib")
    Xs = scaler.transform(Xraw)
    n = len(Xraw)
    out = []

    import keras
    mlp = keras.models.load_model(config.RESULTS_DIR / "mlp_model.keras")
    xgb = joblib.load(config.RESULTS_DIR / "xgb_model.joblib")
    lgbm = joblib.load(config.RESULTS_DIR / "lgbm_model.joblib")
    mlp.predict(Xs, verbose=0)  # warm-up

    def timeit(fn):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return np.median(ts) / n * 1e6  # ms por 1000 muestras

    out.append(dict(modelo="MLP", ms_por_1000=round(timeit(lambda: mlp.predict(Xs, verbose=0)), 3)))
    out.append(dict(modelo="XGBoost", ms_por_1000=round(timeit(lambda: xgb.predict_proba(Xraw)), 3)))
    out.append(dict(modelo="LightGBM", ms_por_1000=round(timeit(lambda: lgbm.predict_proba(Xraw)), 3)))
    return pd.DataFrame(out)


def _fig_zones(zone_df, colors, overall_acc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    bars = ax.bar(range(3), zone_df.porcentaje, color=colors)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Confianza", "Incertidumbre", "Rechazo"])
    ax.set_ylabel("% de detecciones")
    ax.set_title("Distribución por zona de moderación")
    for b, p, n in zip(bars, zone_df.porcentaje, zone_df.n):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.5, f"{p:.0f}%\n(n={n})",
                ha="center", fontsize=14)
    ax.set_ylim(0, max(zone_df.porcentaje) * 1.25)

    ax = axes[1]
    accs = zone_df.exactitud.fillna(0).values
    bars = ax.bar(range(3), accs, color=colors)
    ax.axhline(overall_acc, ls="--", color="#264653", label=f"Exactitud global = {overall_acc:.2f}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Confianza", "Incertidumbre", "Rechazo"])
    ax.set_ylabel("Exactitud")
    ax.set_ylim(0, 1.05)
    ax.set_title("Exactitud por zona (clasificación selectiva)")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=14)
    ax.legend(loc="upper right", fontsize=14)
    fig.suptitle("Moderación por umbrales de probabilidad")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "thr_zones.png")
    plt.close(fig)


def _fig_selective(pmax, pred, y, zones, colors, names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    # Curva exactitud-cobertura
    taus = np.linspace(0.2, 0.99, 60)
    cov, acc = [], []
    for t in taus:
        m = pmax >= t
        cov.append(m.mean())
        acc.append(accuracy_score(y[m], pred[m]) if m.sum() else np.nan)
    ax = axes[0]
    ax.plot(cov, acc, color="#264653", marker="o", ms=3)
    for t, c in [(config.THRESH_CONFIDENCE, GREEN), (config.THRESH_REJECT, RED)]:
        m = pmax >= t
        if m.sum():
            ax.scatter([m.mean()], [accuracy_score(y[m], pred[m])], color=c, s=120, zorder=5,
                       label=f"τ={t:.2f}")
    ax.set_xlabel("Cobertura (fracción aceptada)")
    ax.set_ylabel("Exactitud sobre aceptados")
    ax.set_title("Curva exactitud–cobertura")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=14)

    # Panel derecho: confianza (probabilidad máxima) media por zona.
    ax = axes[1]
    conf_by_zone = [pmax[zones == z].mean() if (zones == z).any() else 0 for z in range(3)]
    bars = ax.bar(range(3), conf_by_zone, color=colors)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Confianza", "Incertidumbre", "Rechazo"])
    ax.set_ylabel("Probabilidad máxima media")
    ax.set_title("Confianza media por zona")
    for b, v in zip(bars, conf_by_zone):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=14)
    ax.set_ylim(0, 1.05)
    fig.suptitle("Clasificación selectiva: exactitud vs cobertura")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "thr_selective.png")
    plt.close(fig)


def _fig_inference(timing):
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    bars = ax.bar(timing.modelo, timing.ms_por_1000, color=["#264653", "#2a9d8f", "#e9c46a"])
    ax.set_ylabel("ms por 1000 muestras (mediana)")
    ax.set_title("Costo de inferencia por modelo")
    for b, v in zip(bars, timing.ms_por_1000):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=14)
    ax.set_ylim(0, max(timing.ms_por_1000) * 1.20)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "thr_inference.png")
    plt.close(fig)


def _write_notes(best, zone_df, overall_acc, auto, y, pred, is_tp, zones, timing):
    tp_by_zone = [round(float(is_tp[zones == z].mean()), 4) if (zones == z).any() else float("nan")
                  for z in range(3)]
    with open(config.RESULTS_DIR / "threshold_notes.md", "w", encoding="utf-8") as fh:
        fh.write("# Interpretación — Moderación por umbrales y trade-offs (Sec 3.5)\n\n")
        fh.write(f"Modelo base: **{best}**. Probabilidad de confianza P = max_k p_k.\n\n")
        fh.write("## Zonas de moderación\n\n")
        fh.write(zone_df.assign(tasa_is_tp=tp_by_zone).to_markdown(index=False) + "\n\n")
        fh.write(f"- Exactitud global = {overall_acc:.3f}. ")
        if auto.sum():
            fh.write(f"En la zona de confianza ({auto.mean()*100:.1f} % de cobertura) la "
                     f"exactitud sube a {accuracy_score(y[auto], pred[auto]):.3f}: la "
                     f"clasificación selectiva concentra la fiabilidad donde el modelo está seguro.\n\n")
        fh.write("## Relación confianza ↔ is_tp\n\n")
        fh.write(f"Tasa de is_tp por zona (confianza/incertidumbre/rechazo): {tp_by_zone}. "
                 "Si la zona de confianza muestra mayor proporción de is_tp=1, la política de "
                 "umbral actúa además como filtro de fiabilidad de la etiqueta (ruido ambiental "
                 "concentrado en baja probabilidad).\n\n")
        fh.write("## Trade-off de costo computacional\n\n")
        fh.write(timing.to_markdown(index=False) + "\n\n")
        fh.write("El MLP implica mayor costo/latencia por el grafo de TensorFlow; los modelos de "
                 "ensamble ofrecen inferencia más liviana. La elección final pondera ese costo "
                 "frente al F1-macro alcanzado (Sec 3.4).\n")


if __name__ == "__main__":
    main()
