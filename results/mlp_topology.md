# Topología del MLP final (Sec 3.4)

- Variante de regularización: **Act → Dropout (solo Drop)**
- Entrenamiento: early stopping sobre val_loss (restore_best_weights)
- F1-macro en validación agrupada: 0.4020
- Pérdida: entropía cruzada categórica (sparse categorical cross-entropy)
- Optimizador: Adam (lr=1e-3) · batch=32

| Capa | Salida | Parámetros | Activación |
|---|---|---|---|
| InputLayer | (None, 64) | 0 | — |
| Dense | (None, 128) | 8320 | linear |
| Activation | (None, 128) | 0 | relu |
| Dropout | (None, 128) | 0 | — |
| Dense | (None, 64) | 8256 | linear |
| Activation | (None, 64) | 0 | relu |
| Dropout | (None, 64) | 0 | — |
| Dense | (None, 5) | 325 | softmax |

**Parámetros totales:** 16901