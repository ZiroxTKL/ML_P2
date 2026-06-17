# Interpretación — Moderación por umbrales y trade-offs (Sec 3.5)

Modelo base: **LightGBM**. Probabilidad de confianza P = max_k p_k.

## Zonas de moderación

| zona                    |   n |   porcentaje |   exactitud |   tasa_is_tp |
|:------------------------|----:|-------------:|------------:|-------------:|
| Confianza (≥85 %)       |   1 |          0.2 |      1      |       0      |
| Incertidumbre (40–85 %) | 329 |         69   |      0.4742 |       0.1277 |
| Rechazo (<40 %)         | 147 |         30.8 |      0.4014 |       0.1565 |

- Exactitud global = 0.453. En la zona de confianza (0.2 % de cobertura) la exactitud sube a 1.000: la clasificación selectiva concentra la fiabilidad donde el modelo está seguro.

## Relación confianza ↔ is_tp

Tasa de is_tp por zona (confianza/incertidumbre/rechazo): [0.0, 0.1277, 0.1565]. Si la zona de confianza muestra mayor proporción de is_tp=1, la política de umbral actúa además como filtro de fiabilidad de la etiqueta (ruido ambiental concentrado en baja probabilidad).

## Trade-off de costo computacional

| modelo   |   ms_por_1000 |
|:---------|--------------:|
| MLP      |       140.188 |
| XGBoost  |         3.421 |
| LightGBM |         4.791 |

El MLP implica mayor costo/latencia por el grafo de TensorFlow; los modelos de ensamble ofrecen inferencia más liviana. La elección final pondera ese costo frente al F1-macro alcanzado (Sec 3.4).
