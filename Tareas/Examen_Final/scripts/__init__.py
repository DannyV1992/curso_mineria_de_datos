"""Scripts reutilizados del material de clase para el Examen Final 2.

- GraficosCuantitativos → Clases/Repo/Clase 2 - EDA/scripts/ (vía Tarea_2)
- PCAAnalysis           → Clases/Repo/Clase 3 - PCA/scripts/ (vía Tarea_3)
- kmeans_utils          → Clases/Notebooks/Clase 8 - Kmeans.ipynb y
                          Clase 10 - Evaluacion clustering.ipynb (vía Tarea_6)
- clustering_jerarquico_utils → Clases/Notebooks/Clase 7 - Clustering jerarquico.ipynb
                          (vía Tarea_6)
"""

from .graficos_cuantitativos import GraficosCuantitativos
from .pca_analysis import PCAAnalysis
from .kmeans_utils import (
    codo_jambu,
    silhouette_kmeans,
    bar_plot,
    bar_plot_detail,
    radar_plot,
    biplot_pca,
)
from .clustering_jerarquico_utils import (
    agregaciones,
    altura_corte,
    plot_dendrograma,
    plot_dendrograma_cortes,
    etiquetas_jerarquicas,
)

__all__ = [
    "GraficosCuantitativos",
    "PCAAnalysis",
    "codo_jambu",
    "silhouette_kmeans",
    "bar_plot",
    "bar_plot_detail",
    "radar_plot",
    "biplot_pca",
    "agregaciones",
    "altura_corte",
    "plot_dendrograma",
    "plot_dendrograma_cortes",
    "etiquetas_jerarquicas",
]
