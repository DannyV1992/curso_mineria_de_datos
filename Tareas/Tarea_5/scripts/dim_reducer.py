import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


class DimReducer:
    def __init__(self, df: pd.DataFrame, color_col: str = None, seed: int = 42):
        self.seed = seed
        self.color_col = color_col
        self.color_label = df[color_col].astype(str) if color_col else None

        X = df.drop(columns=[color_col]) if color_col else df.copy()
        self.X_scaled = StandardScaler().fit_transform(X.select_dtypes("number"))

        self.coords_pca = None
        self.coords_tsne = None
        self.coords_umap = None

    def fit(self, n_components: int = 2, tsne_perplexity: int = 30, umap_n_neighbors: int = 15):
        # Secciones "3. PCA", "4. t-SNE" y "5. UMAP" del notebook original
        self.coords_pca = PCA(n_components=n_components, random_state=self.seed).fit_transform(self.X_scaled)

        self.coords_tsne = TSNE(
            n_components=n_components, perplexity=tsne_perplexity,
            init="pca", learning_rate="auto", random_state=self.seed
        ).fit_transform(self.X_scaled)

        self.coords_umap = umap.UMAP(
            n_components=n_components, n_neighbors=umap_n_neighbors,
            min_dist=0.1, random_state=self.seed
        ).fit_transform(self.X_scaled)

        print(f"Ajuste completado — n_components={n_components}, perplexity={tsne_perplexity}, n_neighbors={umap_n_neighbors}")

    def plot_mapa_interactivo(self, coords, title, xlabel="Dim 1", ylabel="Dim 2"):
        # Función plot_mapa_interactivo() del notebook original
        tmp = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
        if self.color_col:
            tmp[self.color_col] = self.color_label.values

        fig = px.scatter(
            tmp, x="x", y="y", color=self.color_col,
            hover_data=[self.color_col] if self.color_col else None,
            title=title,
            labels={"x": xlabel, "y": ylabel},
            opacity=0.65, width=900, height=550,
        )
        fig.update_traces(marker=dict(size=6))
        fig.show()

    def plot_comparacion(self):
        # Bloque "6. Comparación" del notebook original
        # Cada panel tiene ejes independientes — los rangos de ACP, t-SNE y UMAP son incomparables
        metodos = [("ACP", self.coords_pca), ("t-SNE", self.coords_tsne), ("UMAP", self.coords_umap)]
        categorias = self.color_label.unique() if self.color_label is not None else []
        colores = px.colors.qualitative.Plotly
        color_map = {cat: colores[i % len(colores)] for i, cat in enumerate(sorted(categorias))}

        fig = make_subplots(rows=1, cols=3, subplot_titles=["ACP", "t-SNE", "UMAP"])

        for col_idx, (nombre, coords) in enumerate(metodos, start=1):
            for cat in sorted(categorias):
                mask = self.color_label.values == cat
                fig.add_trace(
                    go.Scatter(
                        x=coords[mask, 0], y=coords[mask, 1],
                        mode="markers",
                        marker=dict(size=4, color=color_map[cat], opacity=0.65),
                        name=str(cat),
                        legendgroup=str(cat),
                        showlegend=(col_idx == 1),
                        hovertemplate=f"{self.color_col}: {cat}<extra></extra>",
                    ),
                    row=1, col=col_idx,
                )

        fig.update_layout(title_text="Comparación ACP vs t-SNE vs UMAP", width=1200, height=450)
        fig.show()

    def plot_3d(self, coords, title):
        # Extensión propia — no existe en el notebook; agregado para los incisos d) y e) de la tarea
        tmp = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2]})
        if self.color_col:
            tmp[self.color_col] = self.color_label.values

        fig = px.scatter_3d(
            tmp, x="x", y="y", z="z", color=self.color_col,
            opacity=0.6, title=title, width=900, height=600,
        )
        fig.update_traces(marker=dict(size=3))
        fig.show()

    def explore_umap_neighbors(self, neighbors_list: list):
        # Extensión propia — no existe en el notebook; agregado para seleccionar n_neighbors óptimo según la tarea
        for nn in neighbors_list:
            coords = umap.UMAP(
                n_components=2, n_neighbors=nn, min_dist=0.1, random_state=self.seed
            ).fit_transform(self.X_scaled)
            self.plot_mapa_interactivo(coords, title=f"UMAP — n_neighbors={nn}")
