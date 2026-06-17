# Contenido del informe (borrador, tono académico impersonal)

> Texto base reutilizable para volcar en la plantilla LaTeX. Redactado en tercera
> persona y voz pasiva impersonal, según los lineamientos. Las figuras se citan por
> su nombre de archivo en `figures/`. Sección 3.4–3.5 se completan tras la corrida final.

## 3.1 Resumen ejecutivo e introducción al espacio vectorial

El presente trabajo aborda la clasificación automatizada de señales eco-acústicas para
la identificación autónoma de especies faunísticas. El problema posee impacto ecológico
directo: el monitoreo acústico pasivo permite censar biodiversidad en zonas extensas sin
intervención humana, pero genera volúmenes de audio cuya anotación manual es inviable. Se
dispone de un conjunto tabular preprocesado de 1906 observaciones de entrenamiento y 477
de prueba, donde cada registro queda descrito por un vector de características
X ∈ ℝ⁶⁴ correspondiente a coeficientes cepstrales en escala de Mel (MFCC, `mel_0`–`mel_63`).
La variable objetivo `species_id` codifica cinco especies —dos anfibios (*Leptodactylus
discodactylus*, *Osteocephalus taurinus*) y tres aves (*Chiroxiphia lineata*, *Saltator
grossus*, *Pheucticus chrysopeplus*)— distribuidas de forma asimétrica (razón de desbalance
máximo/mínimo = 2.0). Los metadatos `songtype_id` (tipo de canto) e `is_tp` (validación
binaria de *true positive*) acompañan a cada vector. Únicamente el 13.2 % de los registros
se encuentra verificado (is_tp = 1), de modo que la mayor parte constituye señal de
fiabilidad incierta asimilable a ruido ambiental. La arquitectura completa del pipeline,
desde la ingesta del CSV hasta la moderación por umbrales, se ilustra en la
Figura `architecture.png`.

Se documenta además una particularidad estructural del conjunto: las 1906 filas de
entrenamiento provienen de solo 1597 grabaciones (`recording_id`), con hasta cinco ventanas
temporales por grabación, y 122 grabaciones aparecen simultáneamente en train y test. Esta
superposición implica fuga de información en el split provisto, por lo que toda validación
interna se efectúa con partición agrupada por `recording_id`.

## 3.2 Exploración geométrica y reducción de dimensionalidad

Se contrastó un método lineal (PCA) frente a dos técnicas de variedades múltiples (t-SNE y
UMAP), proyectando el espacio MFCC estandarizado a 2D y 3D (Figuras `dimred_2d_panels.png`,
`dimred_3d_panels.png`). El análisis de varianza (Figura `dimred_pca_scree.png`) reveló que
PCA requiere 7 componentes para retener el 90 % de la varianza y 11 para el 95 %; en dos
dimensiones retiene apenas el 67.1 % y en tres el 76.7 %.

La capacidad de preservación de la estructura local se cuantificó mediante el índice de
*trustworthiness* y la separabilidad de clases vía kNN con validación cruzada estratificada
en el espacio proyectado:

| Método | Dim | Tiempo (s) | Var. ret. | Trustworthiness | kNN CV acc |
|---|---|---|---|---|---|
| PCA | 2 | 0.002 | 0.671 | 0.867 | 0.311 |
| PCA | 3 | 0.002 | 0.767 | 0.940 | 0.327 |
| t-SNE | 2 | 9.11 | — | 0.987 | 0.369 |
| t-SNE | 3 | 12.00 | — | 0.992 | 0.388 |
| UMAP | 2 | 15.78 | — | 0.976 | 0.374 |
| UMAP | 3 | 5.90 | — | 0.986 | 0.372 |

Se concluye que PCA, aun siendo prácticamente instantáneo (0.002 s) y ofreciendo una base
interpretable por varianza global, distorsiona los vecindarios locales (trustworthiness
0.867 en 2D). Las técnicas no lineales elevan dicho índice a ~0.98–0.99 a un costo de
6–16 s, sin preservar interpretabilidad por varianza. Resulta crítico que la separabilidad
kNN no supera 0.39 en ningún espacio, apenas por encima del baseline de clase mayoritaria
(~0.29): las cinco especies se solapan intensamente en el dominio MFCC, anticipando la
dificultad de la tarea supervisada. Adicionalmente, los registros verificados (is_tp = 1)
aparecen dispersos entre los no verificados sin formar región propia
(Figura `dimred_umap_is_tp.png`), lo que confirma que `is_tp` es una bandera de calidad de
etiqueta y no una clase latente geométricamente separable.

## 3.3 Minería de patrones y estructuras de clustering

Se implementaron dos paradigmas distintos sobre el espacio PCA que retiene el 90 % de la
varianza (7D): un modelo probabilístico (Gaussian Mixture Model) y uno basado en densidad
(DBSCAN). Para el GMM se barrió k ∈ [2, 10] evaluando BIC, AIC, Silhouette, Davies-Bouldin
y Calinski-Harabasz (Figura `clustering_gmm_selection.png`); para DBSCAN se seleccionó eps
mediante el codo del grafo de k-distancias y un barrido que reporta número de clústeres,
fracción de ruido y Silhouette (Figura `clustering_dbscan_selection.png`).

Las métricas internas resultaron contradictorias: BIC y AIC decrecen monótonamente
—favoreciendo el máximo k = 10, síntoma típico de ausencia de componentes gaussianas
separadas— mientras que Silhouette (máximo en k = 2) y Calinski-Harabasz (máximo en k = 3)
favorecen particiones gruesas. No se identificó un número de grupos con clústeres compactos
y bien separados. El contraste externo con las etiquetas reales (Figura
`clustering_umap_compare.png`) fue concluyente: ninguna partición recupera las especies
(ARI ≈ 0.0–0.05, NMI ≈ 0.0), ni se alinea con el tipo de fauna (anfibio/ave) ni con el
`songtype` (ARI ≈ 0). Se infiere que la estructura geométrica dominante del espacio MFCC
obedece a variación acústica de fondo a nivel de grabación, y no a la etiqueta biológica.
Este hallazgo justifica el recurso al aprendizaje supervisado (Sec 3.4) para aislar las
direcciones discriminantes que los métodos no supervisados no logran exponer.

## 3.4 Arquitectura de clasificación: MLP vs. modelos de ensamble

**Función de pérdida.** Para la clasificación multiclase se empleó la entropía cruzada
categórica, definida sobre N muestras y C = 5 clases como
L = −(1/N) Σ_i Σ_c y_{ic} log(p_{ic}), donde p_{ic} = softmax(z_i)_c es la probabilidad
predicha y y_{ic} la codificación one-hot de la clase verdadera. El desbalance se compensó
con ponderación de clases inversamente proporcional a su frecuencia (`class_weight`
balanceado). La topología detallada del MLP final (capas, unidades, parámetros y
activaciones) se reporta en la Tabla de topología (`results/mlp_topology.md`).

**Estrategia de regularización.** Se estudió el efecto de la posición relativa de las capas
de Dropout y Batch Normalization sobre la estabilidad de las curvas de aprendizaje
(Figura `clf_learning_curves.png`), comparando cinco configuraciones del bloque oculto. El
análisis evidenció que el factor determinante de la estabilidad es la **presencia de
Dropout**, no su orden relativo: las tres variantes con Dropout mantienen la pérdida de
validación plana (~1.35) durante 100 épocas, mientras que las configuraciones *BN sin
Dropout* y *sin regularización* sobreajustan —la pérdida de entrenamiento decae pero la de
validación diverge—. Entre las variantes con Dropout, el orden BN→Dropout frente a
Dropout→BN apenas altera el F1-macro. La arquitectura final se seleccionó por mínima pérdida
de validación en convergencia, criterio que descarta las variantes sobreajustadas.

**Benchmark.** El MLP se comparó contra dos modelos de árboles potenciados por gradiente
(XGBoost y LightGBM). La evaluación se sostuvo sobre el F1-Score macro y las matrices de
confusión (Figura `clf_confusion_matrices.png`), con validación agrupada por `recording_id`:

| Modelo | F1-macro (val agr.) | F1-macro (test) | F1-pond. (test) | Exactitud (test) | F1-macro is_tp |
|---|---|---|---|---|---|
| MLP (Keras) | 0.388 | 0.431 | 0.433 | 0.432 | 0.335 |
| XGBoost | 0.392 | 0.419 | 0.435 | 0.436 | 0.402 |
| LightGBM | 0.373 | **0.441** | **0.454** | **0.453** | **0.455** |

El mejor desempeño correspondió a **LightGBM** (F1-macro test = 0.441). El nivel global
(F1-macro ≈ 0.43–0.44) supera el baseline de clase mayoritaria (~0.29 de exactitud) pero
confirma la fuerte confusión inter-especies anticipada en las Secciones 3.2–3.3: las
matrices de confusión muestran dispersión sistemática, en particular entre las tres aves. La
brecha entre F1-macro de validación agrupada y de test refleja el optimismo inducido por las
122 grabaciones solapadas en el split provisto.

## 3.5 Decisiones de ingeniería y mitigación de riesgos

**Trade-off costo/rendimiento.** Se midió el tiempo de inferencia mediano por modelo
(Figura `thr_inference.png`): el MLP requiere ≈140 ms por cada 1000 muestras frente a
≈3.4 ms (XGBoost) y ≈4.8 ms (LightGBM). Los ensambles resultan, por tanto, ~30–40× más
rápidos y simultáneamente más precisos, por lo que LightGBM constituye la elección dominante
en la frontera costo–rendimiento para el despliegue.

**Políticas de moderación por umbrales.** Sobre el vector de probabilidad de LightGBM se
definió la confianza P = max_k p_k y se aplicaron las tres zonas operativas prescritas
(Figuras `thr_zones.png`, `thr_selective.png`). El análisis arrojó un hallazgo de ingeniería
relevante: bajo el umbral prescrito de 85 %, la zona de clasificación automática captura
apenas el 0.2 % de las detecciones (1 de 477), pues el solapamiento entre especies impide
que el modelo alcance alta confianza. La gran mayoría (69 %) cae en la zona de auditoría
(40–85 %) y el 30.8 % en rechazo (<40 %). En consecuencia, se recomienda **recalibrar los
umbrales** a partir de la curva exactitud–cobertura en lugar de adoptar el 85 % nominal, de
modo que la cola de auditoría humana sea operativamente sostenible. Asimismo, la tasa de
is_tp no aumenta en la zona de mayor confianza, lo que indica que la confianza del modelo no
constituye por sí sola un sustituto de la verificación de la etiqueta; el umbral debe
entenderse como mecanismo de control de carga y de descarte de ruido, no como garantía de
veracidad biológica.

## 3.6 Contribution statement

_(tabla de coevaluación — pendiente de nombres y % del equipo)_
