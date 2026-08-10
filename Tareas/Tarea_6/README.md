# Tarea 6 — Aprendizaje No Supervisado: K-Means y Clustering Jerárquico

**Lead University · Minería de Datos · Métodos NO Supervisados**

---

## Objetivo

Aplicar los algoritmos de agrupamiento K-Means y Clustering Jerárquico Aglomerativo sobre dos datasets con contextos distintos (negocio y científico), comparando los resultados mediante métricas de calidad y visualizaciones interpretativas.

---

## Estructura de archivos

```
Tarea_6/
├── datos/
│   ├── Customers.csv          # Dataset de negocio: segmentación de clientes (mall)
│   └── wine-clustering.csv    # Dataset científico: propiedades fisicoquímicas de vinos
├── scripts/
│   ├── __init__.py
│   ├── clustering_jerarquico_utils.py  # Visualizaciones de clustering jerárquico (Clase 7)
│   └── kmeans_utils.py                 # Utilidades de K-Means: codo, silueta, biplot (Clase 8)
├── 1_Customers.ipynb                # Análisis del dataset de clientes
├── 2_Wine.ipynb                     # Análisis del dataset de vinos
├── tarea_clustering.pdf             # Enunciado original de la tarea
└── README.md
```

---

## Descripción de notebooks

### 1_Customers.ipynb — Segmentación de Clientes
Analiza el dataset `Customers.csv` (188 clientes de un centro comercial) con las variables: Age, Annual Income ($), Spending Score (1-100), Work Experience y Family Size.

Secciones:
1. Carga y exploración inicial
2. Preprocesamiento (imputación, outliers, estandarización)
3. Método del Codo + Silueta para K óptimo (K-Means)
4. K-Means con K óptimo + proyección PCA 2D
5. Clustering Jerárquico: dendrogramas, cophenética, silueta vs k, corte Ward
6. Comparación visual K-Means vs Jerárquico (PCA)
7. Perfiles de clusters: heatmap, barras z, radar, líneas
8. Distribución por variable: boxplots y violines
9. Diagnóstico de silueta por cliente
10. Correlación entre variables
11. Tabla resumen y conclusiones de negocio

### 2_Wine.ipynb — Clustering de Vinos
Analiza el dataset `wine-clustering.csv` (178 muestras, 13 propiedades fisicoquímicas) basado en el dataset UCI Wine de tres cultivares italianos.

Secciones:
1. Carga y exploración inicial
2. Preprocesamiento (imputación, outliers, estandarización)
3. Método del Codo + Silueta para K óptimo (K-Means)
4. K-Means con K óptimo + proyecciones PCA 2D, PC1-PC3 y PC2-PC3
5. Clustering Jerárquico: dendrogramas, cophenética, silueta vs k, corte Ward
6. Comparación visual K-Means vs Jerárquico (PCA)
7. Perfiles de clusters: heatmap, barras z, radar, líneas
8. Distribución por variable: boxplots y violines (variables clave)
9. Diagnóstico de silueta por muestra
10. Correlación entre variables fisicoquímicas
11. Tabla resumen y conclusiones enológicas

---

## Ejecución

### Dependencias

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
```

### Instalación

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

### Ejecutar los notebooks

Desde la carpeta `Tarea_6/`:

```bash
jupyter notebook Notebook_1_Customers.ipynb
jupyter notebook Notebook_2_Wine.ipynb
```

> **Importante:** los notebooks deben ejecutarse desde la carpeta `Tarea_6/` para que las rutas relativas (`datos/`, `scripts/`) funcionen correctamente.
