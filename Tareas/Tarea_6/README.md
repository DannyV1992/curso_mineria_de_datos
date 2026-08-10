# Tarea 6 — Aprendizaje No Supervisado: K-Means y Clustering Jerárquico

**Lead University · Minería de Datos · Métodos NO Supervisados**

---

## Objetivo

Aplicar los algoritmos de agrupamiento **K-Means** y **Clustering Jerárquico Aglomerativo** sobre dos datasets de contextos distintos (negocio y científico), determinando el número óptimo de clusters con el Método del Codo de Jambú, validando el corte con el dendrograma y comparando visualmente las asignaciones de cada algoritmo mediante componentes principales.

---

## Estructura de archivos

```
Tarea_6/
├── datos/
│   ├── Customers.csv          # Dataset de negocio: segmentación de clientes (mall)
│   └── wine-clustering.csv    # Dataset científico: propiedades fisicoquímicas de vinos
├── scripts/
│   ├── __init__.py
│   ├── clustering_jerarquico_utils.py  # Clustering jerárquico (Clase 7)
│   └── kmeans_utils.py                 # K-Means: codo, silueta, bar_plot, radar, biplot (Clase 8)
├── 1_Customers.ipynb                # Análisis del dataset de clientes
├── 2_Wine.ipynb                     # Análisis del dataset de vinos
├── tarea_clustering.pdf             # Enunciado original de la tarea
└── README.md
```

---

## Descripción de notebooks

Ambos notebooks siguen la misma estructura:

| Sección | Punto del enunciado |
|---|---|
| 2. Carga y exploración inicial | 1 — Selección de datos |
| 3. Preprocesamiento (nulos, atípicos, estandarización) | 2 — Preprocesamiento |
| 4. Codo de Jambú y silueta | 3 — Número óptimo de clusters (K) |
| 5. K-Means con el K óptimo | 4 — Aplicación de algoritmos |
| 6. Clustering jerárquico (dendrogramas y corte) | 4 — Aplicación de algoritmos |
| 7. Dispersión con PCA por algoritmo | 5 — Análisis y conclusiones visuales |
| 8. Conclusiones | 5 — Análisis y conclusiones visuales |

### 1_Customers.ipynb — Segmentación de clientes

`Customers.csv`: 2000 clientes de un centro comercial. Variables usadas: `Age`, `Annual Income ($)`, `Spending Score (1-100)`, `Work Experience` y `Family Size`.

Resultado: el codo desciende suavemente sin quiebre nítido y la silueta maximiza en el extremo del rango, señales de que **no hay grupos naturalmente separados**. Se adopta k = 4 por el codo visual y por interpretabilidad comercial. La concordancia entre K-Means y Ward es baja (ARI ≈ 0.25) y las dos primeras componentes explican solo el 43.7 % de la varianza: las fronteras entre segmentos son difusas.

### 2_Wine.ipynb — Clustering de vinos

`wine-clustering.csv`: 178 muestras con 13 propiedades fisicoquímicas (dataset UCI Wine, tres cultivares italianos).

Resultado: el codo marca un quiebre claro en **k = 3** y la silueta lo confirma de forma independiente, coincidiendo con los tres cultivares reales. Ambos algoritmos convergen a los mismos perfiles químicos (ARI ≈ 0.85) y las dos primeras componentes explican el 55.4 % de la varianza, suficiente para ver tres nubes bien separadas.

El contraste entre ambos datasets es en sí mismo un resultado: con estructura real, el algoritmo elegido casi no altera el resultado; con estructura difusa, sí.

---

## Ejecución

### Dependencias

Determinadas a partir de los imports reales de los notebooks y scripts:

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
jupyter notebook 1_Customers.ipynb
jupyter notebook 2_Wine.ipynb
```

> **Importante:** los notebooks deben ejecutarse desde la carpeta `Tarea_6/` para que las rutas relativas (`datos/`, `scripts/`) funcionen correctamente.
