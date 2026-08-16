"""Clase PCA reutilizable — Tarea 3: Reducción de Dimensionalidad.

Usa prince.PCA para alinearse con la metodología de clase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import prince


class PCAAnalysis:
    """Encapsula estandarización, ajuste PCA (prince) y visualizaciones obligatorias."""

    def __init__(self, n_components: int = 5, **pca_kwargs) -> None:
        self.n_components = n_components
        self._pca_kwargs = pca_kwargs
        self.pca: prince.PCA | None = None
        self.datos_: pd.DataFrame | None = None
        self._cols_pca: list[str] | None = None

    def ajustar(self, datos: pd.DataFrame, columnas_pca: list[str] | None = None) -> "PCAAnalysis":
        self.datos_ = datos.copy()
        if columnas_pca is None:
            cols = self.datos_.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = list(columnas_pca)
        self._cols_pca = cols
        self.pca = prince.PCA(
            n_components=min(self.n_components, len(cols)),
            **self._pca_kwargs,
        )
        self.pca.fit(self.datos_[self._cols_pca])
        return self

    def _require(self) -> prince.PCA:
        if self.pca is None:
            raise RuntimeError("Llama primero a ajustar(datos).")
        return self.pca

    # --- Propiedades ---

    @property
    def varianza_explicada(self) -> np.ndarray:
        return np.array(self._require().percentage_of_variance_)

    @property
    def varianza_acumulada(self) -> np.ndarray:
        return np.array(self._require().cumulative_percentage_of_variance_)

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.array(self._require().eigenvalues_)

    @property
    def row_coords(self) -> pd.DataFrame:
        pca = self._require()
        coords = pca.row_coordinates(self.datos_)
        coords.columns = [f"PC{i+1}" for i in range(coords.shape[1])]
        return coords

    @property
    def col_correlations(self) -> pd.DataFrame:
        pca = self._require()
        col_corr_obj = pca.column_correlations
        corr = col_corr_obj(self.datos_) if callable(col_corr_obj) else col_corr_obj
        corr.columns = [f"PC{i+1}" for i in range(corr.shape[1])]
        return corr.loc[self._cols_pca]

    def n_componentes_80(self) -> int:
        return int(np.argmax(self.varianza_acumulada >= 80)) + 1

    # --- 1. Scree Plot ---

    def scree_plot(self, ax=None) -> plt.Axes:
        ax = ax or plt.gca()
        explained = self.varianza_explicada
        cumulative = self.varianza_acumulada
        comps = [f"PC{i+1}" for i in range(len(explained))]

        ax.bar(comps, explained, color="slateblue", alpha=0.85)
        ax.set_ylabel("% Varianza explicada")
        ax.set_xlabel("Componente principal")

        ax2 = ax.twinx()
        ax2.plot(comps, cumulative, color="darkorange", marker="o", linewidth=2)
        ax2.set_ylabel("% Varianza acumulada")
        ax2.set_ylim(0, 105)
        ax2.axhline(80, ls="--", color="gray", lw=1, label="80 %")

        for i, v in enumerate(explained):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

        n80 = self.n_componentes_80()
        ax.axvline(n80 - 0.5, ls=":", color="green", lw=1.2)
        ax.set_title("Scree Plot — Varianza explicada por componente")
        return ax

    # --- 2. Círculo de correlación ---

    def circulo_correlacion(self, ejes=(1, 2), ax=None) -> plt.Axes:
        ax = ax or plt.gca()
        i, j = ejes[0] - 1, ejes[1] - 1
        corr = self.col_correlations
        explained = self.varianza_explicada

        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=1)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)

        pc_i = f"PC{ejes[0]}"
        pc_j = f"PC{ejes[1]}"
        for var in corr.index:
            x_val = corr.loc[var, pc_i]
            y_val = corr.loc[var, pc_j]
            ax.arrow(0, 0, x_val, y_val, color="darkred", alpha=0.75,
                     head_width=0.03, head_length=0.04, length_includes_head=True)
            ax.text(x_val * 1.1, y_val * 1.1, var, fontsize=9, ha="center", va="center")

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_xlabel(f"PC{ejes[0]} ({explained[i]:.1f}% var.)")
        ax.set_ylabel(f"PC{ejes[1]} ({explained[j]:.1f}% var.)")
        ax.set_title("Círculo de correlación")
        ax.grid(True, alpha=0.3)
        return ax

    # --- 3. Plano principal ---

    def plano_principal(self, hue: pd.Series | None = None, ejes=(1, 2), ax=None, **kwargs) -> plt.Axes:
        ax = ax or plt.gca()
        coords = self.row_coords
        pc_i = f"PC{ejes[0]}"
        pc_j = f"PC{ejes[1]}"
        explained = self.varianza_explicada
        i, j = ejes[0] - 1, ejes[1] - 1

        plot_df = coords[[pc_i, pc_j]].copy()
        if hue is not None:
            plot_df["_hue"] = hue.values
            sns.scatterplot(data=plot_df, x=pc_i, y=pc_j, hue="_hue",
                            s=55, alpha=0.75, ax=ax, **kwargs)
            ax.legend(title=hue.name or "Clase", bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            ax.scatter(plot_df[pc_i], plot_df[pc_j], alpha=0.6, s=40, c="steelblue")

        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel(f"PC{ejes[0]} ({explained[i]:.1f}% var.)")
        ax.set_ylabel(f"PC{ejes[1]} ({explained[j]:.1f}% var.)")
        ax.set_title("Plano principal (individuos)")
        ax.grid(True, alpha=0.3)
        return ax

    # --- 4. Biplot ---

    def biplot(self, hue: pd.Series | None = None, ejes=(1, 2), ax=None, **kwargs) -> plt.Axes:
        ax = ax or plt.gca()
        coords = self.row_coords
        corr = self.col_correlations
        pc_i = f"PC{ejes[0]}"
        pc_j = f"PC{ejes[1]}"
        explained = self.varianza_explicada
        i, j = ejes[0] - 1, ejes[1] - 1

        plot_df = coords[[pc_i, pc_j]].copy()
        if hue is not None:
            plot_df["_hue"] = hue.values
            sns.scatterplot(data=plot_df, x=pc_i, y=pc_j, hue="_hue",
                            s=40, alpha=0.6, ax=ax, **kwargs)
            ax.legend(title=hue.name or "Clase", bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            ax.scatter(plot_df[pc_i], plot_df[pc_j], alpha=0.5, s=30, c="steelblue")

        scale = min(np.nanmax(np.abs(coords[pc_i])), np.nanmax(np.abs(coords[pc_j]))) * 0.8
        for var in corr.index:
            vx = corr.loc[var, pc_i] * scale
            vy = corr.loc[var, pc_j] * scale
            ax.arrow(0, 0, vx, vy, color="darkred", alpha=0.8,
                     head_width=0.08, head_length=0.12, length_includes_head=True)
            ax.text(vx * 1.08, vy * 1.08, var, color="darkred", fontsize=9)

        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel(f"PC{ejes[0]} ({explained[i]:.1f}% var.)")
        ax.set_ylabel(f"PC{ejes[1]} ({explained[j]:.1f}% var.)")
        ax.set_title("Biplot (individuos + variables)")
        ax.grid(True, alpha=0.3)
        return ax

    # --- 5. Contribuciones ---

    def contribuciones(self) -> pd.DataFrame:
        corr = self.col_correlations
        eigenvals = self.eigenvalues
        contrib = (corr ** 2).copy()
        for i, pc in enumerate(contrib.columns):
            contrib[pc] = (contrib[pc] / eigenvals[i]) * 100
        return contrib.round(2)

    # --- 6. Cos² variables ---

    def cos2_variables(self) -> pd.DataFrame:
        return (self.col_correlations ** 2).round(4)
