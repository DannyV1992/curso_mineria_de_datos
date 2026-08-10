"""Clustering Jerárquico Aglomerativo.

Clase POO construida a partir del notebook
``Clases/Notebooks/Clase 7 - Clustering jerarquico.ipynb``.

Cada método indica de qué celda o función del notebook proviene. Los métodos
marcados como "Extensión propia" no existen en el notebook de clase y se
agregaron porque el enunciado de la Tarea 8 los requiere.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import (
    average,
    complete,
    dendrogram,
    fcluster,
    linkage,
    single,
    ward,
)
from scipy.spatial.distance import pdist


class ClusteringJerarquico:
    """Clustering jerárquico aglomerativo sobre un DataFrame numérico.

    Parámetros
    ----------
    datos : pd.DataFrame
        Datos numéricos ya estandarizados. El índice se usa como etiqueta de
        las hojas del dendrograma (equivale a ``datos.set_index('Nombre')``
        del notebook de clase).
    """

    def __init__(self, datos):
        # Celdas 10-13 del notebook: datos con el índice ya definido y shape
        self.datos = datos.copy()
        self.dist = None
        self.ward_res = None
        self.average_res = None
        self.single_res = None
        self.complete_res = None
        self.grupos = None
        self.centros = None
        print(self.datos.shape)

    # ------------------------------------------------------------------
    # Funciones auxiliares definidas al inicio del notebook
    # ------------------------------------------------------------------
    @staticmethod
    def centroide(num_cluster, datos, clusters):
        # Función centroide() del notebook Clase 7 - Clustering jerarquico.ipynb
        ind = clusters == num_cluster
        return pd.DataFrame(datos[ind].mean()).T

    @staticmethod
    def bar_plot(centros, labels, scale=False, cluster=None, var=None):
        # Función bar_plot() del notebook Clase 7 - Clustering jerarquico.ipynb
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
    def radar_plot(centros, labels):
        # Función radar_plot() del notebook Clase 7 - Clustering jerarquico.ipynb
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
    # 7. Matriz de distancias
    # ------------------------------------------------------------------
    def matriz_distancias(self, metric="euclidean"):
        # Bloque "7. Calcular la matriz de distancias" del notebook Clase 7
        self.dist = pdist(self.datos, metric=metric)
        return self.dist

    # ------------------------------------------------------------------
    # 10. Probar los 4 métodos de agregación
    # ------------------------------------------------------------------
    def calcular_agregaciones(self):
        # Bloque "10. Probar los 4 métodos de agregación" del notebook Clase 7
        self.ward_res = ward(self.datos)  # Ward
        self.average_res = average(self.datos)  # Promedio
        self.single_res = single(self.datos)  # Salto mínimo
        self.complete_res = complete(self.datos)  # Salto máximo
        return self.ward_res, self.average_res, self.single_res, self.complete_res

    # ------------------------------------------------------------------
    # 11-14. Dendrogramas por método
    # ------------------------------------------------------------------
    def plot_dendrograma(self, metodo="ward", figsize=(12, 8), dpi=100, truncate=None):
        # Bloques "11-14. Dendrograma con <método>" del notebook Clase 7
        resultados = {
            "ward": self.ward_res,
            "average": self.average_res,
            "single": self.single_res,
            "complete": self.complete_res,
        }
        Z = resultados[metodo]
        if Z is None:
            raise ValueError("Debe llamar antes a calcular_agregaciones().")

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        # Extensión propia — truncate_mode se agrega porque los datasets de la
        # tarea tienen cientos de observaciones y las etiquetas no caben.
        if truncate is not None:
            dendrogram(Z, ax=ax, truncate_mode="lastp", p=truncate, show_leaf_counts=True)
            ax.set_xlabel("Observaciones agrupadas (tamaño del nodo)")
        else:
            dendrogram(Z, labels=self.datos.index.tolist(), ax=ax)
            ax.set_xlabel("Orden en el eje X")
        ax.set_ylabel("Distancia o Agregación")
        ax.set_title(f"Dendrograma — método {metodo}")
        return fig

    def plot_dendrogramas_comparacion(self, figsize=(16, 10), dpi=100, truncate=30):
        # Extensión propia — reúne en una cuadrícula los 4 dendrogramas que el
        # notebook Clase 7 dibuja en celdas separadas (bloques 11 a 14).
        metodos = [
            ("single", "Single (salto mínimo)"),
            ("complete", "Complete (salto máximo)"),
            ("average", "Average (promedio)"),
            ("ward", "Ward (mínima varianza)"),
        ]
        resultados = {
            "ward": self.ward_res,
            "average": self.average_res,
            "single": self.single_res,
            "complete": self.complete_res,
        }
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        for ax, (met, titulo) in zip(axes.ravel(), metodos):
            dendrogram(
                resultados[met],
                ax=ax,
                truncate_mode="lastp",
                p=truncate,
                show_leaf_counts=True,
            )
            ax.set_title(titulo)
            ax.set_xlabel("Observaciones agrupadas")
            ax.set_ylabel("Distancia o Agregación")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 15. Elegir el número de clusters (cortar el dendrograma)
    # ------------------------------------------------------------------
    def plot_dendrograma_corte(self, alturas, etiquetas, figsize=(15, 8), dpi=150, truncate=30):
        # Bloque "15. Elegir el número de clusters (cortar el dendrograma)"
        # del notebook Clase 7 - Clustering jerarquico.ipynb
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        dendrogram(self.ward_res, ax=ax, truncate_mode="lastp", p=truncate, show_leaf_counts=True)

        limites = ax.get_xbound()
        for altura, texto in zip(alturas, etiquetas):
            ax.plot(limites, [altura, altura], "--", c="k")
            ax.text(limites[1], altura, " " + texto, va="center", fontdict={"size": 12})
        ax.set_xlabel("Orden en el eje X")
        ax.set_ylabel("Distancia o Agregación")
        ax.set_title("Dendrograma Ward con cortes candidatos")
        return fig

    # ------------------------------------------------------------------
    # 16. Obtener las etiquetas de cluster
    # ------------------------------------------------------------------
    def obtener_grupos(self, k, method="ward", metric="euclidean"):
        # Bloque "16. Obtener las etiquetas de cluster" del notebook Clase 7
        grupos = fcluster(
            linkage(self.datos, method=method, metric=metric), k, criterion="maxclust"
        )
        # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
        grupos = grupos - 1
        self.grupos = grupos
        return grupos

    # ------------------------------------------------------------------
    # 17. Calcular los centros de cada cluster
    # ------------------------------------------------------------------
    def calcular_centros(self, k=None):
        # Bloque "17. Calcular los centros de cada cluster" del notebook Clase 7
        if self.grupos is None:
            raise ValueError("Debe llamar antes a obtener_grupos().")
        k = k if k is not None else len(np.unique(self.grupos))
        self.centros = np.array(
            pd.concat([self.centroide(i, self.datos, self.grupos) for i in range(k)])
        )
        return self.centros

    # ------------------------------------------------------------------
    # 18 y 21. Interpretación de los clusters
    # ------------------------------------------------------------------
    def plot_barras_centros(self, figsize=(14, 8)):
        # Bloque "18. Interpretar los clusters con gráfico de barras" del notebook Clase 7
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.bar_plot(self.centros, self.datos.columns)
        return fig

    def plot_radar_centros(self, figsize=(10, 8)):
        # Bloque "21. Interpretar los clusters con gráfico de radar" del notebook Clase 7
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.radar_plot(self.centros, self.datos.columns)
        return fig

    # ------------------------------------------------------------------
    # Extensiones propias requeridas por la Tarea 8
    # ------------------------------------------------------------------
    def tabla_perfiles(self, datos_originales=None):
        # Extensión propia — no existe en el notebook de clase.
        # Resume el perfil medio y el tamaño de cada cluster en escala original.
        if self.grupos is None:
            raise ValueError("Debe llamar antes a obtener_grupos().")
        base = self.datos if datos_originales is None else datos_originales
        tabla = base.groupby(self.grupos).mean()
        tabla.insert(0, "n", pd.Series(self.grupos).value_counts().sort_index().values)
        tabla.index.name = "cluster"
        return tabla.round(2)

    def correlacion_cophenetica(self):
        # Extensión propia — no existe en el notebook de clase.
        # Compara la calidad de los 4 métodos de vinculación: mide qué tanto el
        # dendrograma preserva las distancias originales entre observaciones.
        from scipy.cluster.hierarchy import cophenet

        if self.dist is None:
            self.matriz_distancias()

        resultados = {
            "single": self.single_res,
            "complete": self.complete_res,
            "average": self.average_res,
            "ward": self.ward_res,
        }
        filas = []
        for metodo, Z in resultados.items():
            coef, _ = cophenet(Z, self.dist)
            filas.append({"metodo": metodo, "cophenetica": round(float(coef), 3)})
        return pd.DataFrame(filas).sort_values("cophenetica", ascending=False)

    def plot_pca_clusters(self, figsize=(9, 7), titulo="Clustering jerárquico proyectado en PCA"):
        # Extensión propia — no existe en el notebook de clase.
        # El enunciado pide graficar los clusters con PCA cuando hay más de 2 variables.
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        componentes = pca.fit_transform(self.datos)
        var_exp = pca.explained_variance_ratio_ * 100

        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(
            componentes[:, 0],
            componentes[:, 1],
            c=self.grupos,
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
