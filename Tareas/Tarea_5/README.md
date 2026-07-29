# Tarea 5 — Reducción de Dimensionalidad: ACP, t-SNE y UMAP

**Lead University · Minería de Datos · Métodos NO Supervisados**

## Objetivo

Comparar ACP (lineal), t-SNE y UMAP (no lineales) sobre tres datasets distintos,
evaluando su capacidad de visualizar estructura interna y separar grupos en 2D y 3D.

---

## Estructura de archivos

```
Tarea_5/
├── datos/
│   ├── ropa.csv              # Fashion MNIST (1 000 × 785): imágenes de prendas
│   ├── winequality.csv       # Vinho Verde (1 599 × 12): vino tinto
│   └── Players1.csv          # Copas Mundiales (595 × 8): estadísticas de jugadores
├── scripts/
│   ├── __init__.py
│   └── dim_reducer.py        # Clase DimReducer (ACP + t-SNE + UMAP + Plotly)
├── 1_Fashion_MNIST.ipynb     # Ejercicio 1
├── 2_Wine_Quality.ipynb      # Ejercicio 2
├── 3_Players_Mundiales.ipynb # Ejercicio 3
└── README.md
```

---

## Descripción de notebooks

| Notebook | Dataset | Ejercicios |
|---|---|---|
| `1_Fashion_MNIST.ipynb` | `ropa.csv` | a–e: 2D y 3D sobre 10 tipos de prendas |
| `2_Wine_Quality.ipynb` | `winequality.csv` | a–d: 3 componentes + proyección 2D, análisis por calidad |
| `3_Players_Mundiales.ipynb` | `Players1.csv` | a–d: separación por posición táctica |

---

## Uso de la clase DimReducer

```python
from scripts import DimReducer

dr = DimReducer(df, label_col='etiqueta', seed=42)

# Explorar n_neighbors antes de elegir el óptimo
dr.explore_umap_neighbors(neighbors_list=[5, 15, 30, 50])

# Ajustar
dr.fit_2d(tsne_perplexity=30, umap_n_neighbors=30)
dr.fit_3d(tsne_perplexity=30, umap_n_neighbors=30)

# Gráficos interactivos (Plotly)
dr.plot_2d(title='Comparación 2D')
dr.plot_3d(title='Comparación 3D')
```

---

## Ejecución

### Dependencias

```
pandas
scikit-learn
umap-learn
plotly
```

```bash
pip install pandas scikit-learn umap-learn plotly
```

```bash
jupyter notebook
```

> **Nota:** t-SNE sobre `ropa.csv` (784 variables, 1 000 obs.) puede tardar varios minutos.
> UMAP es considerablemente más rápido.
