from __future__ import annotations

from math import ceil, floor, pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def codo_jambu(X: np.ndarray, max_k: int = 10, *, figsize: tuple[float, float] = (8, 5)) -> plt.Figure:
    inercias = []
    for k in range(1, max_k + 1):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo.fit(X)
        inercias.append(modelo.inertia_)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(range(1, max_k + 1), inercias, marker="o", color="steelblue", linewidth=2)
    ax.set_title("Método del Codo de Jambú")
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inercia (WCSS)")
    ax.grid(alpha=0.3)
    return fig


def silhouette_kmeans(X: np.ndarray, max_k: int = 10, *, figsize: tuple[float, float] = (8, 5)) -> plt.Figure:
    max_k = min(max_k, len(X) - 1)
    scores = []
    for k in range(2, max_k + 1):
        etiquetas = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        scores.append({"k": k, "silhouette": silhouette_score(X, etiquetas)})

    sil_df = pd.DataFrame(scores)
    k_opt = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(sil_df["k"], sil_df["silhouette"], marker="o", color="steelblue", linewidth=2)
    ax.axvline(k_opt, color="crimson", ls="--", label=f"k óptimo = {k_opt}")
    ax.set_title("Método de la Silueta")
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Silueta media")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig, k_opt


def bar_plot(
    centros: np.ndarray,
    labels: list[str],
    *,
    scale: bool = False,
    cluster: list[int] | None = None,
    var: list[str] | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    centros = np.copy(centros).astype(float)
    if scale:
        for col in range(centros.shape[1]):
            col_max = centros[:, col].max()
            if col_max != 0:
                centros[:, col] /= col_max

    colores = sns.color_palette()
    minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0
    labels = np.array(labels)

    if var is not None:
        mask = np.array([l in var for l in labels])
        centros = centros[:, mask]
        colores = [colores[i % len(colores)] for i, flag in enumerate(mask) if flag]
        labels = labels[mask]

    clusters_to_plot = cluster if cluster is not None else list(range(centros.shape[0]))
    n = len(clusters_to_plot)

    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for pos, idx in enumerate(clusters_to_plot):
        axes[pos].barh(range(len(labels)), centros[idx].tolist(), 1 / 1.5, color=colores)
        axes[pos].set_xlim(minimo, ceil(centros.max()))
        axes[pos].set_title(f"Cluster {idx}")
        if pos == 0:
            axes[pos].set_yticks(range(len(labels)))
            axes[pos].set_yticklabels(labels)
        else:
            axes[pos].set_yticks([])

    fig.suptitle("Perfiles de centroides por cluster", fontsize=12)
    return fig


def bar_plot_detail(
    centros: np.ndarray,
    columns_names: list[str],
    columns_to_plot: list | None = None,
    *,
    figsize: tuple[float, float] = (10, 7),
) -> plt.Figure:
    n_clusters = centros.shape[0]
    cluster_labels = [f"Cluster {i}" for i in range(n_clusters)]
    df_centros = pd.DataFrame(centros, columns=columns_names, index=cluster_labels)

    columns = columns_names
    if columns_to_plot:
        if isinstance(columns_to_plot[0], str):
            columns = columns_to_plot
        else:
            columns = [columns_names[i] for i in columns_to_plot]

    plots = len(columns)
    rows = ceil(plots / 2)
    cols = 2

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    for var_idx, col_name in enumerate(columns):
        ax = fig.add_subplot(rows, cols, var_idx + 1)
        sns.barplot(y=cluster_labels, x=col_name, data=df_centros, ax=ax, palette="Set2")
        ax.set_ylabel("")

    fig.suptitle("Detalle de centroides por variable", fontsize=12)
    return fig


def radar_plot(centros: np.ndarray, labels: list[str], *, figsize: tuple[float, float] = (8, 8)) -> plt.Figure:
    centros = np.array(
        [
            ((n - n.min()) / (n.max() - n.min()) * 100) if n.max() != n.min() else (n / n * 50)
            for n in centros.T
        ]
    )
    n_feat = len(labels)
    angulos = [i / float(n_feat) * 2 * pi for i in range(n_feat)]
    angulos += angulos[:1]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(labels)
    ax.set_rlabel_position(0)
    ax.set_ylim(-10, 100)
    plt.yticks(
        [20, 40, 60, 80, 100],
        ["20%", "40%", "60%", "80%", "100%"],
        color="grey",
        size=8,
    )

    cmap = plt.get_cmap("tab10")
    for i in range(centros.shape[1]):
        valores = centros[:, i].tolist() + centros[:, i].tolist()[:1]
        ax.plot(angulos, valores, linewidth=1.5, linestyle="solid", label=f"Cluster {i}", color=cmap(i))
        ax.fill(angulos, valores, alpha=0.2, color=cmap(i))

    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    fig.suptitle("Radar de perfiles por cluster (escala 0–100%)", fontsize=12)
    fig.tight_layout()
    return fig


def biplot_pca(
    datos: pd.DataFrame,
    labels: np.ndarray,
    *,
    random_state: int = 42,
    figsize: tuple[float, float] = (11, 9),
    titulo: str = "Biplot PCA: observaciones (clusters) + variables",
    nombre_obs: str | None = None,
) -> tuple[plt.Figure, PCA]:
    """Scatter PCA coloreado por cluster con flechas de loadings (biplot)."""
    pca = PCA(n_components=2, random_state=random_state)
    scores = pca.fit_transform(datos)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    scale_pts = 1.0 / (scores.max(axis=0) - scores.min(axis=0))
    xs = scores[:, 0] * scale_pts[0]
    ys = scores[:, 1] * scale_pts[1]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    scatter = ax.scatter(xs, ys, c=labels, cmap="tab10", s=120, edgecolor="k", zorder=3)

    # Etiquetas de observaciones
    obs_names = nombre_obs if nombre_obs is not None else (
        list(datos.index) if isinstance(datos, pd.DataFrame) else [str(i) for i in range(len(xs))]
    )
    for i, nombre in enumerate(obs_names):
        ax.annotate(nombre, (xs[i], ys[i]), textcoords="offset points", xytext=(7, 5),
                    fontsize=8, fontweight="bold", zorder=4)

    # Flechas de loadings
    scale_arrow = 1.0 / np.abs(loadings).max() * 0.9
    feature_names = datos.columns if isinstance(datos, pd.DataFrame) else [f"V{i}" for i in range(loadings.shape[0])]
    for i, variable in enumerate(feature_names):
        ax.arrow(0, 0,
                 loadings[i, 0] * scale_arrow, loadings[i, 1] * scale_arrow,
                 color="red", alpha=0.85, head_width=0.025, length_includes_head=True,
                 linewidth=1.8, zorder=2)
        ax.text(loadings[i, 0] * scale_arrow * 1.18, loadings[i, 1] * scale_arrow * 1.18,
                variable, color="darkred", fontsize=9, fontweight="bold",
                ha="center", va="center", zorder=5)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    var_exp = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% varianza)")
    ax.set_title(titulo)
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.grid(alpha=0.25)
    return fig, pca
