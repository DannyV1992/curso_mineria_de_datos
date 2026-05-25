"""Graficos para explorar relaciones entre variables cuantitativas."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class GraficosCuantitativos:
    """Graficos utiles para variables numericas y sus relaciones."""

    def __init__(self, df: pd.DataFrame, cols: Iterable[str] | None = None):
        self.df = df
        self.cols = list(cols) if cols else df.select_dtypes(include=np.number).columns.tolist()

    def pairplot(self, hue: str | None = None, **kwargs) -> sns.axisgrid.PairGrid:
        """Matriz de dispersiones entre variables numericas."""
        g = sns.pairplot(self.df, vars=self.cols, hue=hue, corner=True, **kwargs)
        g.fig.suptitle("Pairplot", y=1.02)
        plt.tight_layout()
        return g

    def violin(self, x: str, y: str, hue: str | None = None, ax=None, **kwargs):
        """Distribucion de y segun categorias en x (violin plot)."""
        ax = ax or plt.gca()
        sns.violinplot(data=self.df, x=x, y=y, hue=hue, ax=ax, **kwargs)
        ax.set_title(f"Violin: {y} por {x}")
        return ax

    def barplot(self, x: str, y: str, hue: str | None = None, ax=None, **kwargs):
        """Barras de una media o conteo agregado de y por x."""
        ax = ax or plt.gca()
        sns.barplot(data=self.df, x=x, y=y, hue=hue, ax=ax, errorbar="ci", **kwargs)
        ax.set_title(f"Barplot: {y} por {x}")
        return ax

    def boxplot(self, x: str, y: str, hue: str | None = None, ax=None, **kwargs):
        """Caja y bigotes de y segun x."""
        ax = ax or plt.gca()
        sns.boxplot(data=self.df, x=x, y=y, hue=hue, ax=ax, **kwargs)
        ax.set_title(f"Boxplot: {y} por {x}")
        return ax

    def lineplot(self, x: str, y: str, hue: str | None = None, ax=None, **kwargs):
        """Tendencia de y frente a x."""
        ax = ax or plt.gca()
        sns.lineplot(data=self.df, x=x, y=y, hue=hue, ax=ax, marker="o", **kwargs)
        ax.set_title(f"Lineplot: {y} vs {x}")
        return ax

    def scatter(self, x: str, y: str, hue: str | None = None, ax=None, **kwargs):
        """Dispersion entre dos variables numericas."""
        ax = ax or plt.gca()
        sns.scatterplot(data=self.df, x=x, y=y, hue=hue, ax=ax, **kwargs)
        ax.set_title(f"Scatter: {y} vs {x}")
        return ax

    def regplot(self, x: str, y: str, ax=None, **kwargs):
        """Dispersion con recta de regresion simple."""
        ax = ax or plt.gca()
        sns.regplot(data=self.df, x=x, y=y, ax=ax, **kwargs)
        ax.set_title(f"Regresion simple: {y} ~ {x}")
        return ax

    def histplot(self, col: str, kde: bool = True, ax=None, **kwargs):
        """Histograma de una variable numerica."""
        ax = ax or plt.gca()
        sns.histplot(data=self.df, x=col, kde=kde, ax=ax, **kwargs)
        ax.set_title(f"Histograma: {col}")
        return ax

    def heatmap_correlacion(self, method: str = "pearson", annot: bool = True, ax=None, **kwargs):
        """Mapa de calor de la matriz de correlacion."""
        corr = self.df[self.cols].corr(method=method)
        ax = ax or plt.gca()
        sns.heatmap(corr, annot=annot, fmt=".2f", cmap="coolwarm", center=0, ax=ax, **kwargs)
        ax.set_title(f"Correlacion ({method})")
        return ax

    def jointplot(self, x: str, y: str, kind: str = "scatter", **kwargs) -> sns.JointGrid:
        """Grafico conjunto (marginales + relacion central)."""
        g = sns.jointplot(data=self.df, x=x, y=y, kind=kind, **kwargs)
        g.fig.suptitle(f"Jointplot: {y} vs {x}", y=1.02)
        return g
