"""
Diagrama de arquitectura del pipeline (Sec 3.1).

Ilustra el flujo completo: ingesta del CSV -> espacio de características MFCC
(X in R^64) -> clasificación supervisada -> moderación por umbrales -> salida
informativa, con la rama de análisis exploratorio no supervisado (reducción de
dimensionalidad + clustering). Se exporta como PNG para el informe.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import config

config.apply_style()

BLUE = "#264653"
TEAL = "#2a9d8f"
SAND = "#e9c46a"
ORANGE = "#e76f51"


def box(ax, x, y, w, h, title, subtitle, color, sec=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=2, edgecolor=color, facecolor=color + "22"))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
            fontsize=16, fontweight="bold", color=BLUE)
    ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
            fontsize=14, color="#333333")
    if sec:
        ax.text(x + w / 2, y - 0.028, sec, ha="center", va="center",
                fontsize=14, style="italic", color=color)


def arrow(ax, x0, y0, x1, y1, color=BLUE, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=22, linewidth=2.2, color=color))


def main():
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    yM, hM, wM = 0.66, 0.18, 0.165
    xs = [0.02, 0.215, 0.41, 0.605, 0.80]
    box(ax, xs[0], yM, wM, hM, "Ingesta de datos", "CSV train / test\n1906 / 477 obs · 68 cols", BLUE)
    box(ax, xs[1], yM, wM, hM, "Espacio MFCC", "X ∈ R^64 (mel_0..63)\n+ StandardScaler", TEAL)
    box(ax, xs[2], yM, wM, hM, "Clasificación", "MLP (Keras) vs\nXGBoost / LightGBM", BLUE, "Sec 3.4")
    box(ax, xs[3], yM, wM, hM, "Moderación", "Umbrales sobre el\nvector de probabilidad", ORANGE, "Sec 3.5")
    box(ax, xs[4], yM, wM, hM, "Salida informativa", "Especie detectada /\ncola de auditoría", BLUE)
    for i in range(4):
        arrow(ax, xs[i] + wM, yM + hM / 2, xs[i + 1], yM + hM / 2)

    # Rama exploratoria no supervisada (desde el espacio de características)
    yL, hL, wL = 0.20, 0.16, 0.22
    xL = [0.30, 0.56]
    box(ax, xL[0], yL, wL, hL, "Reducción dimensional", "PCA · UMAP · t-SNE\n(2D / 3D, tiempos)", SAND, "Sec 3.2")
    box(ax, xL[1], yL, wL, hL, "Clustering", "DBSCAN · GMM\n+ Silhouette / BIC", SAND, "Sec 3.3")
    arrow(ax, xs[1] + wM / 2, yM, xL[0] + wL / 2, yL + hL, color=SAND)
    arrow(ax, xL[0] + wL, yL + hL / 2, xL[1], yL + hL / 2, color=SAND)
    ax.text(0.43, 0.075, "Análisis exploratorio no supervisado (informa el diseño)",
            ha="center", fontsize=14, style="italic", color="#9a7d0a")

    # Zonas de umbral (desde Moderación)
    zy, zh, zw = 0.40, 0.07, 0.30
    zx = 0.80
    zones = [("P ≥ 85 %  →  automático", "#2e7d32", "#e8f5e9"),
             ("40 % ≤ P < 85 %  →  auditoría", "#f9a825", "#fff8e1"),
             ("P < 40 %  →  descarte", "#c62828", "#ffebee")]
    arrow(ax, xs[3] + wM / 2, yM, zx + zw / 2, zy + zh, color=ORANGE)
    for i, (txt, ec, fc) in enumerate(zones):
        yy = zy - i * (zh + 0.005)
        ax.add_patch(FancyBboxPatch((zx, yy), zw, zh, boxstyle="round,pad=0.004",
                                    linewidth=2, edgecolor=ec, facecolor=fc))
        ax.text(zx + zw / 2, yy + zh / 2, txt, ha="center", va="center",
                fontsize=14, color=ec, fontweight="bold")

    ax.set_title("Arquitectura del pipeline — clasificación de señales eco-acústicas",
                 fontsize=18, fontweight="bold", pad=14)
    fig.savefig(config.FIG_DIR / "architecture.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] Diagrama guardado en", config.FIG_DIR / "architecture.png")


if __name__ == "__main__":
    main()
