"""K-Means + PCA.

Clase POO construida a partir del notebook
``Clases/Notebooks/Clase 8 - Kmeans.ipynb``.

Cada método indica de qué celda o función del notebook proviene. Los métodos
marcados como "Extensión propia" no existen en el notebook de clase y se
agregaron porque el enunciado de la Tarea 8 los requiere.
"""

from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class KMeansAnalysis:
    """K-Means sobre un DataFrame numérico, con interpretación y PCA.

    Parámetros
    ----------
    datos : pd.DataFrame
        Datos numéricos ya estandarizados. El índice se usa para etiquetar las
        observaciones en el biplot (equivale a ``datos.set_index('Nombre')``
        del notebook de clase).
    """

    def __init__(self, datos):
        # Bloques "6. Cargar los datos" y "7. Preparar el índice" del notebook Clase 8
        self.datos = datos.copy()
        self.kmedias = None
        self.etiquetas = None
        self.centros = None
        self.pca = None
        self.componentes = None
        print(self.datos.shape)

    # ------------------------------------------------------------------
    # Funciones auxiliares definidas al inicio del notebook
    # ------------------------------------------------------------------
    @staticmethod
    def centroide(num_cluster, datos, clusters):
        # Función centroide() del notebook Clase 8 - Kmeans.ipynb
        ind = clusters == num_cluster
        return pd.DataFrame(datos[ind].mean()).T

    @staticmethod
    def bar_plot(centros, labels, scale=False, cluster=None, var=None):
        # Función bar_plot() del notebook Clase 8 - Kmeans.ipynb
        from math import ceil, floor

        from seaborn import color_palette

        centros = np.copy(centros)

        if scale:
            for col in range(centros.shape[1]):
                centros[:, col] /= max(centros[:, col])

        colores = color_palette()
        minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0

        def inside_plot(valores, labels, titulo):
            plt.barh(range(len(valores)), valores, 1 / 1.5, color=colores)
            plt.xlim(minimo, ceil(centros.max()))
            plt.title(titulo)

        if var is not None:
            centros = np.array([n[[x in var for x in labels]] for n in centros])
            colores = [colores[x % len(colores)] for x, i in enumerate(labels) if i in var]
            labels = labels[[x in var for x in labels]]
        if cluster is None:
            for i in range(centros.shape[0]):
                plt.subplot(1, centros.shape[0], i + 1)
                inside_plot(centros[i].tolist(), labels, ("Cluster " + str(i)))
                plt.yticks(range(len(labels)), labels) if i == 0 else plt.yticks([])
        else:
            pos = 1
            for i in cluster:
                plt.subplot(1, len(cluster), pos)
                inside_plot(centros[i].tolist(), labels, ("Cluster " + str(i)))
                plt.yticks(range(len(labels)), labels) if pos == 1 else plt.yticks([])
                pos += 1

    @staticmethod
    def bar_plot_detail(centros, columns_names=[], columns_to_plot=[], figsize=(10, 7), dpi=150):
        # Función bar_plot_detail() del notebook Clase 8 - Kmeans.ipynb
        from math import ceil

        import seaborn as sb

        numClusters = centros.shape[0]
        labels = ["Cluster " + str(i) for i in range(numClusters)]
        centros = pd.DataFrame(centros, columns=columns_names, index=labels)

        plots = len(columns_to_plot) if len(columns_to_plot) != 0 else len(columns_names)
        rows, cols = ceil(plots / 2), 2

        plt.figure(1, figsize=figsize, dpi=dpi)
        plt.subplots_adjust(hspace=1, wspace=0.5)
        columns = columns_names
        if len(columns_to_plot) > 0:
            if type(columns_to_plot[0]) is str:
                columns = columns_to_plot
            else:
                columns = [columns_names[i] for i in columns_to_plot]
        var = 0
        for numRow in range(rows):
            for numCol in range(cols):
                if var < plots:
                    ax = plt.subplot2grid((rows, cols), (numRow, numCol), colspan=1, rowspan=1)
                    sb.barplot(y=labels, x=columns[var], data=centros, ax=ax)
                    var += 1

    @staticmethod
    def radar_plot(centros, labels):
        # Función radar_plot() del notebook Clase 8 - Kmeans.ipynb
        from math import pi

        centros = np.array(
            [
                ((n - min(n)) / (max(n) - min(n)) * 100) if max(n) != min(n) else (n / n * 50)
                for n in centros.T
            ]
        )
        angulos = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angulos += angulos[:1]
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angulos[:-1], labels)
        ax.set_rlabel_position(0)
        plt.yticks(
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"],
            color="grey",
            size=8,
        )
        plt.ylim(-10, 100)
        for i in range(centros.shape[1]):
            valores = centros[:, i].tolist()
            valores += valores[:1]
            ax.plot(angulos, valores, linewidth=1, linestyle="solid", label="Cluster " + str(i))
            ax.fill(angulos, valores, alpha=0.3)
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    # ------------------------------------------------------------------
    # 8. Determinación del número de clusters
    # ------------------------------------------------------------------
    def codo_jambu(self, max_k=10):
        # Función codo_jambu() del notebook Clase 8 - Kmeans.ipynb
        inercias = []

        for k in range(1, max_k + 1):
            modelo = KMeans(n_clusters=k, random_state=42)
            modelo.fit(self.datos)
            inercias.append(modelo.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_k + 1), inercias, marker="o")
        plt.title("Método del Codo de Jambú")
        plt.xlabel("Número de Clusters (k)")
        plt.ylabel("Inercia")
        plt.grid(True)
        plt.show()
        # Extensión propia — se devuelven las inercias para citarlas en la interpretación
        return inercias

    def silhouette_kmeans(self, max_k=10):
        # Función silhouette_kmeans() del notebook Clase 8 - Kmeans.ipynb
        max_k = min(max_k, len(self.datos) - 1)

        scores = []

        for k in range(2, max_k + 1):
            modelo = KMeans(n_clusters=k, random_state=42)
            etiquetas = modelo.fit_predict(self.datos)

            score = silhouette_score(self.datos, etiquetas)
            scores.append(score)

        plt.figure(figsize=(8, 5))
        plt.plot(range(2, max_k + 1), scores, marker="o")
        plt.title("Método Silhouette")
        plt.xlabel("Número de Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.grid(True)
        plt.show()
        # Extensión propia — se devuelven los scores para citarlos en la interpretación
        return pd.DataFrame({"k": range(2, max_k + 1), "silhouette": scores})

    # ------------------------------------------------------------------
    # 8-10. Ajustar K-Means y obtener etiquetas y centros
    # ------------------------------------------------------------------
    def ajustar(self, n_clusters=3, random_state=42):
        # Bloques "8. Aplicar K-Means", "9. Obtener las etiquetas de cluster" y
        # "10. Obtener los centros de cada cluster" del notebook Clase 8
        self.kmedias = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.kmedias.fit(self.datos)
        self.etiquetas = self.kmedias.predict(self.datos)
        self.centros = np.array(self.kmedias.cluster_centers_)
        print("Grupos\n", self.etiquetas)
        print("Centros\n", self.centros)
        return self.etiquetas, self.centros

    # ------------------------------------------------------------------
    # 11-12. Interpretación de los centros
    # ------------------------------------------------------------------
    def plot_barras_centros(self, figsize=(14, 8), dpi=100):
        # Bloque "11. Visualizar los centros con bar_plot" del notebook Clase 8
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        self.bar_plot(self.centros, self.datos.columns)
        return fig

    def plot_detalle_centros(self, columns_to_plot=[], figsize=(10, 9), dpi=110):
        # Bloque de la función bar_plot_detail() del notebook Clase 8
        self.bar_plot_detail(
            self.centros,
            columns_names=list(self.datos.columns),
            columns_to_plot=columns_to_plot,
            figsize=figsize,
            dpi=dpi,
        )
        return plt.gcf()

    def plot_radar_centros(self, figsize=(10, 8), dpi=100):
        # Bloque "12. Visualizar los centros con gráfico de radar" del notebook Clase 8
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        self.radar_plot(self.centros, self.datos.columns)
        return fig

    # ------------------------------------------------------------------
    # 13-15. PCA y biplot
    # ------------------------------------------------------------------
    def aplicar_pca(self, n_components=2):
        # Bloques "13. Reducción de dimensiones con PCA" y "14. Mostrar las
        # componentes" del notebook Clase 8
        self.pca = PCA(n_components=n_components)
        self.componentes = self.pca.fit_transform(self.datos)
        print("Componentes\n", self.componentes)
        return self.componentes

    def plot_biplot(self, figsize=(11, 9), dpi=110, anotar=False, titulo=None):
        # Bloque "15. Visualizar los clusters en 2D (PCA + nombres)" del notebook Clase 8
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        etiquetas = self.kmedias.predict(self.datos)

        scores = self.componentes
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)

        scale_pts = 1.0 / (scores.max(axis=0) - scores.min(axis=0))
        xs = scores[:, 0] * scale_pts[0]
        ys = scores[:, 1] * scale_pts[1]

        scatter = ax.scatter(
            xs, ys, c=etiquetas, cmap="viridis", s=45, edgecolor="k", linewidth=0.3, zorder=3
        )

        # Extensión propia — las anotaciones del notebook se vuelven opcionales
        # porque con cientos de observaciones el gráfico queda ilegible.
        if anotar:
            for i, nombre in enumerate(self.datos.index):
                ax.annotate(
                    nombre,
                    (xs[i], ys[i]),
                    textcoords="offset points",
                    xytext=(8, 6),
                    fontsize=10,
                    fontweight="bold",
                    zorder=4,
                )

        scale_arrow = 1.0 / np.abs(loadings).max() * 0.9
        for i, variable in enumerate(self.datos.columns):
            ax.arrow(
                0,
                0,
                loadings[i, 0] * scale_arrow,
                loadings[i, 1] * scale_arrow,
                color="red",
                alpha=0.85,
                head_width=0.025,
                length_includes_head=True,
                linewidth=2,
                zorder=2,
            )
            ax.text(
                loadings[i, 0] * scale_arrow * 1.15,
                loadings[i, 1] * scale_arrow * 1.15,
                variable,
                color="darkred",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=5,
            )

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

        var_exp = self.pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"Componente 1 ({var_exp[0]:.1f}% varianza)")
        ax.set_ylabel(f"Componente 2 ({var_exp[1]:.1f}% varianza)")
        ax.set_title(titulo or "Biplot del PCA: observaciones (clusters) + variables")

        legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
        ax.add_artist(legend1)

        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Extensiones propias requeridas por la Tarea 8
    # ------------------------------------------------------------------
    def tabla_perfiles(self, datos_originales=None):
        # Extensión propia — no existe en el notebook de clase.
        # Perfil medio y tamaño de cada cluster en la escala original.
        base = self.datos if datos_originales is None else datos_originales
        tabla = base.groupby(self.etiquetas).mean()
        tabla.insert(0, "n", pd.Series(self.etiquetas).value_counts().sort_index().values)
        tabla.index.name = "cluster"
        return tabla.round(2)

    def plot_pca_clusters(self, figsize=(9, 7), titulo="K-Means proyectado en PCA"):
        # Extensión propia — no existe en el notebook de clase.
        # Dispersión PCA sin flechas, para comparar lado a lado con el jerárquico.
        var_exp = self.pca.explained_variance_ratio_ * 100
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(
            self.componentes[:, 0],
            self.componentes[:, 1],
            c=self.etiquetas,
            cmap="viridis",
            s=45,
            edgecolor="k",
            linewidth=0.3,
            alpha=0.9,
        )
        ax.set_xlabel(f"Componente 1 ({var_exp[0]:.1f}% varianza)")
        ax.set_ylabel(f"Componente 2 ({var_exp[1]:.1f}% varianza)")
        ax.set_title(titulo)
        ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig
