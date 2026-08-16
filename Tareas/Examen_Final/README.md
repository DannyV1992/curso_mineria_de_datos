# Examen Final — Aprendizaje no supervisado

**Lead University · Ciencia de Datos · Minería de Datos I**

## Objetivo

Examen final del curso, compuesto por dos partes:

- **Parte I (teórica):** 15 preguntas de respuesta corta sobre PCA, ACM, t-SNE, UMAP y clustering (K-Means, DBSCAN, jerárquico) y sus criterios de evaluación.
- **Parte II (práctica):** dos casos de aplicación de reducción de dimensionalidad y clustering:
  - **Caso 1 — Wine Quality:** determinar si las características fisicoquímicas de los vinos permiten identificar grupos naturales, siguiendo la ruta exploración → PCA → K-Means → validación con múltiples criterios → interpretación con las variables originales.
  - **Caso 2 — Iris:** clusterizar las flores usando únicamente las cuatro variables predictoras y comparar después los grupos encontrados con las especies reales.

## Estructura de archivos

```
Tareas/Examen_Final/
├── datos/
│   ├── winequality.csv
│   └── iris.csv
├── scripts/
│   ├── __init__.py
│   ├── graficos_cuantitativos.py
│   ├── pca_analysis.py
│   ├── kmeans_utils.py
│   └── clustering_jerarquico_utils.py
├── Examen_final.ipynb
├── Examen_Minería_de_datos_I.pdf
└── README.md
```

## Descripción de los notebooks

- **`Examen_final.ipynb`:** contiene ambas partes del examen. La Parte I responde las 15 preguntas teóricas en celdas Markdown. La Parte II desarrolla los dos casos prácticos (Wine Quality e Iris), cada uno con exploración inicial, preparación de datos, aplicación de PCA, selección de componentes, clustering con K-Means (validado con codo de Jambú, índice de silueta y, en el Caso 1, dendrograma de Ward), visualización de los clusters sobre el plano principal e interpretación de los grupos con las variables originales.

## Scripts

Los scripts en `scripts/` reutilizan las clases y funciones vistas en el material de clase, ya usadas en tareas anteriores:

- `graficos_cuantitativos.py` → clase `GraficosCuantitativos`.
- `pca_analysis.py` → clase `PCAAnalysis`.
- `kmeans_utils.py` → funciones `codo_jambu`, `silhouette_kmeans`, `bar_plot`, `bar_plot_detail`, `radar_plot`, `biplot_pca` de los notebooks de Kmeans y Evaluacion Clustering.
- `clustering_jerarquico_utils.py` → funciones `agregaciones`, `altura_corte`, `plot_dendrograma`, `plot_dendrograma_cortes`, `etiquetas_jerarquicas` del notebook de Clustering Jerarquico.

## Ejecución

### Dependencias

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
prince
```

### Cómo ejecutar

Desde la carpeta `Tareas/Examen_Final/`:

```bash
jupyter notebook Examen_final.ipynb
```

Los datos se cargan con rutas relativas (`datos/winequality.csv`, `datos/iris.csv`), por lo que el notebook debe ejecutarse desde esta carpeta.
