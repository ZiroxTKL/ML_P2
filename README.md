# Proyecto 2 — Clasificación de Señales Eco-Acústicas

**Curso:** CS3061 Machine Learning · UTEC
**Objetivo:** Pipeline integral de ML para identificar especies faunísticas a partir
de descriptores eco-acústicos tabulares (coeficientes MFCC, X ∈ ℝ⁶⁴).

## Dataset

| Archivo | Observaciones | Columnas |
|---|---|---|
| `data/eco_acoustic_train.csv` | 1906 | 68 |
| `data/eco_acoustic_test.csv` | 477 | 68 |

- **Metadatos:** `recording_id`, `species_id` (target), `songtype_id`, `is_tp`.
- **Características:** `mel_0 … mel_63` (MFCC, X ∈ ℝ⁶⁴).
- **5 clases** (asimétricas): 10, 12 (anfibios) · 17, 18, 23 (aves).
- `is_tp`: validación binaria de *true positive* (solo ~13 % verificados).

## Estructura

```
P2_ML/
├── data/        CSVs + documentación del dataset
├── src/         pipeline (config, carga, EDA, reducción, clustering, clasificación)
├── figures/     figuras generadas para el informe (fuente ≥ 14)
├── results/     métricas y tablas
└── report/      informe LaTeX (máx. 10 págs.)
```

## Pipeline (secciones del informe)

1. **3.1** Resumen ejecutivo + diagrama de arquitectura + espacio vectorial
2. **3.2** Reducción de dimensionalidad — PCA vs UMAP / t-SNE (2D/3D, tiempos)
3. **3.3** Clustering — DBSCAN vs GMM + Silhouette
4. **3.4** Clasificación — MLP (Keras) vs XGBoost/LightGBM (F1, matrices de confusión)
5. **3.5** Políticas de umbral y trade-offs

## Entorno

```bash
conda create -n p2ml python=3.11
conda activate p2ml
pip install -r requirements.txt
```

## Ejecución

```bash
conda run -n p2ml python src/eda.py            # EDA + figuras base
# (los demás scripts del pipeline se ejecutan en orden)
```
