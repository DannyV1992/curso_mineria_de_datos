from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import average, complete, dendrogram, fcluster, linkage, single, ward
from scipy.spatial.distance import pdist


def centroide(num_cluster, datos, clusters):
    ind = clusters == num_cluster
    return pd.DataFrame(datos[ind].mean()).T


def matriz_distancias(datos, metric: str = "euclidean"):
    return pdist(datos, metric=metric)


def agregaciones(datos):
    ward_res = ward(datos)          # Ward
    average_res = average(datos)    # Promedio
    single_res = single(datos)      # Salto mínimo
    complete_res = complete(datos)  # Salto máximo
    return ward_res, average_res, single_res, complete_res


def plot_dendrograma(
    Z,
    labels=None,
    titulo: str = "Dendrograma",
    *,
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 100,
    truncate_mode: str | None = None,
    p: int = 30,
):
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    if truncate_mode is not None:
        dendrogram(Z, ax=ax, truncate_mode=truncate_mode, p=p, show_leaf_counts=True)
        ax.set_xlabel("Observaciones (grupos truncados, entre paréntesis su tamaño)")
    else:
        dendrogram(Z, labels=list(labels) if labels is not None else None, ax=ax)
        ax.set_xlabel("Orden en el eje X")
    ax.set_ylabel("Distancia o Agregación")
    ax.set_title(titulo)
    return fig


def plot_dendrograma_cortes(
    ward_res,
    labels=None,
    cortes: list[tuple[float, str]] | None = None,
    *,
    figsize: tuple[float, float] = (15, 8),
    dpi: int = 120,
    titulo: str = "Dendrograma Ward con líneas de corte",
    truncate_mode: str | None = None,
    p: int = 30,
):
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    if truncate_mode is not None:
        dendrogram(ward_res, ax=ax, truncate_mode=truncate_mode, p=p, show_leaf_counts=True)
    else:
        dendrogram(ward_res, labels=list(labels) if labels is not None else None, ax=ax)

    limites = ax.get_xbound()
    for altura, texto in cortes or []:
        ax.plot(limites, [altura, altura], "--", c="k")
        ax.text(limites[1], altura, f" {texto}", va="center", fontdict={"size": 13})

    ax.set_xlabel("Orden en el eje X")
    ax.set_ylabel("Distancia o Agregación")
    ax.set_title(titulo)
    return fig


def altura_corte(Z, k: int) -> float:
    """Extensión propia — no existe en el notebook de clase.
    El notebook escribe a mano las alturas de corte (7.25 para dos clústeres y
    4 para tres) porque el dataset es pequeño y se leen del gráfico. Aquí se
    calculan desde la matriz de enlaces para poder anotar el corte de cada k."""
    alturas = np.sort(Z[:, 2])
    n = Z.shape[0] + 1
    if k <= 1:
        return float(alturas[-1]) * 1.05
    if k >= n:
        return 0.0
    return float((alturas[n - k - 1] + alturas[n - k]) / 2)


def etiquetas_jerarquicas(datos, k: int, method: str = "ward", metric: str = "euclidean"):
    grupos = fcluster(linkage(datos, method=method, metric=metric), k, criterion="maxclust")
    grupos = grupos - 1  # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
    return grupos


def centros_clusters(datos, grupos, k: int) -> np.ndarray:
    return np.array(pd.concat([centroide(c, datos, grupos) for c in range(k)]))
