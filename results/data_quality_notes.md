# Notas de calidad de datos y decisiones analíticas

Hallazgos del EDA que alimentan el informe (Sec 3.1 y 3.5). Estos puntos son las
"dificultades del caso" que la rúbrica premia abordar explícitamente.

## 1. Espacio de características
- X ∈ ℝ⁶⁴ (mel_0..mel_63), coeficientes MFCC. Sin valores faltantes.
- Rango global ≈ [-1.59, 5.85], media 0.34, std 0.93 → casi estandarizado pero
  conviene aplicar `StandardScaler` (ajustado solo en train) antes de PCA/UMAP/MLP.

## 2. Desbalance de clases (moderado)
- 5 clases: 17 (27.3 %) y 23 (29.0 %) dominan; 10, 12, 18 ≈ 14.5 % c/u.
- Ratio max/min = 2.0. → usar F1 macro + `class_weight`/estratificación, NO accuracy
  (la rúbrica penaliza evaluar solo con accuracy).

## 3. Indicador `is_tp` — la decisión central del proyecto
- Solo 252/1906 (13.2 %) están verificados como True Positive; tasa uniforme entre
  especies (0.10–0.16), así que NO es un sesgo por clase.
- Interpretación (según documentación): fiabilidad/certeza de presencia de la señal.
  Los `is_tp=0` representan ventanas no confiables ≈ ruido ambiental.
- **Decisión adoptada (a confirmar):** entrenar la clasificación sobre TODAS las filas
  (entrenar solo con 252 TP deja muy pocos datos para 5 clases + MLP), y usar `is_tp`
  para (a) evaluar también sobre el subconjunto verificado y (b) fundamentar las
  políticas de umbral de la Sec 3.5 (baja probabilidad ≈ ruido a descartar/auditar).
  Se evaluará además ponderar por `is_tp` como experimento.

## 4. Fuga de datos por `recording_id` (importante)
- train: 1906 filas pero solo 1597 `recording_id` únicos (257 grabaciones con
  múltiples ventanas, hasta 5 filas por grabación).
- 187 grabaciones contienen >1 especie distinta (varias especies por audio).
- **122 `recording_id` aparecen tanto en train como en test** → el split provisto
  tiene fuga: métricas sobre el test dado son algo optimistas.
- **Mitigación:** la validación interna usará `GroupKFold` agrupando por `recording_id`
  para estimaciones honestas; se reportará también el test provisto, señalando el caveat.
