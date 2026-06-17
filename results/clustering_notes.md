# Interpretación — Clustering (Sec 3.3)

Espacio de clustering: PCA con 7 componentes (90 % de varianza).

## GMM (probabilístico)

|   k |     BIC |     AIC |   silhouette |   davies_bouldin |   calinski_harabasz |
|----:|--------:|--------:|-------------:|-----------------:|--------------------:|
|   2 | 54048   | 53653.7 |       0.202  |           1.8026 |               447.4 |
|   3 | 53338   | 52743.8 |       0.1648 |           1.9375 |               514.3 |
|   4 | 52598.8 | 51804.8 |       0.1182 |           2.3642 |               256.9 |
|   5 | 52281.6 | 51287.6 |       0.0613 |           2.573  |               251.4 |
|   6 | 51959.5 | 50765.6 |       0.0684 |           2.1228 |               242.7 |
|   7 | 51941.3 | 50547.5 |       0.0508 |           2.2434 |               219.2 |
|   8 | 51836.1 | 50242.5 |       0.0625 |           2.0044 |               217.2 |
|   9 | 51679.7 | 49886.2 |       0.0221 |           2.4608 |               176.8 |
|  10 | 51507.5 | 49514.1 |       0.0198 |           2.2439 |               185.7 |

- **k\*=10** por BIC.

## DBSCAN (densidad)

|    eps |   n_clusters |   ruido_frac |   silhouette |
|-------:|-------------:|-------------:|-------------:|
| 2.4358 |            7 |        0.316 |       0.1801 |
| 2.5528 |            6 |        0.29  |       0.2218 |
| 2.6845 |            6 |        0.251 |       0.1829 |
| 2.7963 |            4 |        0.226 |       0.2243 |
| 2.9133 |            5 |        0.188 |       0.2329 |
| 3.0594 |            4 |        0.159 |       0.2329 |
| 3.1784 |            3 |        0.134 |       0.2772 |
| 3.2936 |            2 |        0.121 |       0.2897 |
| 3.4636 |            2 |        0.094 |       0.2862 |
| 3.6826 |            1 |        0.078 |     nan      |
| 3.922  |            2 |        0.058 |       0.5409 |
| 4.1916 |            3 |        0.036 |       0.4312 |
| 4.7466 |            1 |        0.022 |     nan      |
| 5.4111 |            1 |        0.01  |     nan      |

- eps elegido = 3.922 (codo del grafo de k-distancias, min_samples=10).

## Concordancia con etiquetas reales (externo)

| metodo   |   n_clusters |   ruido_frac |   NMI_especie |   ARI_especie |   ARI_tipo |   ARI_songtype |
|:---------|-------------:|-------------:|--------------:|--------------:|-----------:|---------------:|
| GMM (k*) |           10 |        0     |         0.065 |         0.052 |     -0.003 |          0.008 |
| DBSCAN   |            2 |        0.058 |         0.002 |        -0     |     -0.001 |         -0.004 |

## Lectura analítica

1. **Métricas internas en conflicto.** BIC y AIC decrecen de forma monótona (favorecen k=10, el máximo del rango): señal típica de ausencia de gaussianas separadas, pues añadir componentes siempre mejora el ajuste de densidad. En cambio Silhouette (máx. en k=2) y Calinski-Harabasz (máx. en k=3) favorecen particiones gruesas. No existe un k con clústeres compactos y bien separados.
2. **El clustering no recupera las especies.** Todas las soluciones dan ARI≈0 y NMI≈0 frente a la especie; tampoco se alinean con el tipo de fauna (anfibio/ave) ni con el songtype. La estructura geométrica del espacio MFCC está dominada por variación acústica de fondo a nivel de grabación (ruido, canal, sitio), no por la etiqueta biológica.
3. **Implicación.** La señal que separa especies es débil y no recuperable de forma no supervisada por densidad o probabilidad; se requiere aprendizaje supervisado (Sec 3.4) para aislar las direcciones discriminantes. Es coherente con la baja separabilidad kNN de la Sec 3.2 y con la alta proporción de ventanas no verificadas (is_tp = 13 %).
