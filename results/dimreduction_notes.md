# Interpretación — Reducción de dimensionalidad (Sec 3.2)

## Resultados cuantitativos
| Método | Dims | Tiempo (s) | Var. retenida | Trustworthiness | kNN CV acc |
|---|---|---|---|---|---|
| PCA | 2 | 0.002 | 0.671 | 0.867 | 0.311 |
| PCA | 3 | 0.002 | 0.767 | 0.940 | 0.327 |
| t-SNE | 2 | 9.11 | — | 0.987 | 0.369 |
| t-SNE | 3 | 12.00 | — | 0.992 | 0.388 |
| UMAP | 2 | 15.78 | — | 0.976 | 0.374 |
| UMAP | 3 | 5.90 | — | 0.986 | 0.372 |

- PCA necesita **7 componentes para 90 %** y **11 para 95 %** de la varianza (de 64).

## Lectura analítica (para el informe)
1. **Lineal vs no lineal.** PCA es prácticamente instantáneo (0.002 s) y ofrece una
   base interpretable por varianza global, pero en 2D retiene solo el **67 %** y su
   trustworthiness (0.867) es la más baja: la proyección lineal distorsiona los
   vecindarios locales. t-SNE y UMAP elevan la trustworthiness a ~0.98–0.99 a costa
   de 6–16 s y de perder la interpretación por varianza.
2. **Trade-off tiempo/estructura.** Orden de preservación local: t-SNE ≳ UMAP ≫ PCA.
   Orden de costo: PCA ≪ UMAP ≈ t-SNE. UMAP 3D fue más rápido que UMAP 2D porque el
   primer ajuste 2D incluyó la compilación JIT (numba).
3. **Clases poco separables.** El kNN en el espacio proyectado alcanza solo
   **0.31 (PCA) → 0.37–0.39 (t-SNE/UMAP)**, apenas por encima del baseline de clase
   mayoritaria (~0.29). Las 5 especies se **solapan fuertemente** en el espacio MFCC:
   la tarea de clasificación es no trivial y motiva modelos potentes + moderación por
   umbrales (Sec 3.5).
4. **`is_tp` no es un grupo geométrico.** En UMAP 2D los registros verificados
   (is_tp=1) aparecen dispersos entre los no verificados, sin formar región propia →
   confirma que `is_tp` es una **bandera de calidad de etiqueta**, no una clase latente
   separable. Refuerza la decisión de tratarlo como criterio de fiabilidad/umbral y no
   como variable de segmentación.

## Figuras
- `dimred_pca_scree.png` — varianza explicada/acumulada (justifica retención).
- `dimred_2d_panels.png` — PCA/t-SNE/UMAP en 2D por especie (tiempos + métricas).
- `dimred_3d_panels.png` — idem en 3D.
- `dimred_umap_is_tp.png` — UMAP 2D coloreado por is_tp.
