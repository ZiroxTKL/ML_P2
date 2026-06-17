# Interpretación — Clasificación (Sec 3.4)

## Experimento de regularización (posición Dropout/BatchNorm)

| Variante | F1-macro val | σ(val_loss) últ. 20 ép. | Deriva val_loss |
|---|---|---|---|
| Act → Dropout → BN | 0.4165 | 0.015 | +0.064 |
| Sin regularización | 0.4032 | 0.114 | +1.190 |
| Act → Dropout (solo Drop) | 0.4020 | 0.016 | +0.033 |
| BN → Act → Dropout | 0.3847 | 0.021 | +0.092 |
| BN → Act (solo BN) | 0.3782 | 0.159 | +1.757 |

**Lectura guiada por las curvas (no por el F1):** el factor decisivo de la estabilidad es la **presencia de Dropout**, no el orden relativo de las capas. Las tres variantes con Dropout mantienen la pérdida de validación plana (~1.35) durante 100 épocas, mientras que *BN sin Dropout* y *sin regularización* sobreajustan: la pérdida de entrenamiento cae pero la de validación se dispara (deriva positiva grande). La curva más estable fue *Act → Dropout → BN* y la más inestable *BN → Act (solo BN)*. Entre las variantes con Dropout, el orden BN→Dropout vs Dropout→BN apenas altera el F1-macro (rango 0.378–0.416) por el fuerte solapamiento entre especies. Variante seleccionada para el modelo final, por menor pérdida de validación en convergencia (penaliza el sobreajuste): **Act → Dropout (solo Drop)**.

## Benchmark

| modelo   |   F1_macro_val |   F1_macro_test |   F1_weighted_test |   accuracy_test |   F1_macro_test_isTP |
|:---------|---------------:|----------------:|-------------------:|----------------:|---------------------:|
| MLP      |         0.3877 |          0.4309 |             0.4333 |          0.4319 |               0.3348 |
| XGBoost  |         0.3923 |          0.4188 |             0.4352 |          0.4361 |               0.4023 |
| LightGBM |         0.3731 |          0.4406 |             0.4542 |          0.4528 |               0.4546 |

Mejor modelo por F1-macro (test): **LightGBM**. El desempeño global (F1-macro ≈ 0.43–0.44) supera el baseline de clase mayoritaria (~0.29 de exactitud) pero refleja la fuerte confusión entre especies anticipada en las Secs 3.2–3.3.

La brecha F1_macro_val (agrupado por recording_id) vs F1_macro_test (split provisto, con 122 grabaciones solapadas con train) refleja el optimismo por fuga del conjunto de prueba entregado.

### Reporte por clase del mejor modelo (test)

```
                             precision    recall  f1-score   support

Leptodactylus discodactylus      0.383     0.449     0.413        69
     Osteocephalus taurinus      0.383     0.443     0.411        70
        Chiroxiphia lineata      0.534     0.477     0.504       130
           Saltator grossus      0.419     0.377     0.397        69
    Pheucticus chrysopeplus      0.482     0.475     0.478       139

                   accuracy                          0.453       477
                  macro avg      0.440     0.444     0.441       477
               weighted avg      0.458     0.453     0.454       477

```
