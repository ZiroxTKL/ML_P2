# Metricas de reduccion de dimensionalidad (Sec 3.2)

| metodo   |   dims |   tiempo_s |   var_retenida |   trustworthiness |   knn_cv_acc |
|:---------|-------:|-----------:|---------------:|------------------:|-------------:|
| PCA      |      2 |      0.002 |         0.671  |            0.867  |       0.3106 |
| PCA      |      3 |      0.002 |         0.7669 |            0.9395 |       0.3274 |
| t-SNE    |      2 |      9.111 |       nan      |            0.9871 |       0.3688 |
| t-SNE    |      3 |     11.997 |       nan      |            0.992  |       0.3877 |
| UMAP     |      2 |     15.779 |       nan      |            0.9759 |       0.3741 |
| UMAP     |      3 |      5.903 |       nan      |            0.9856 |       0.372  |
