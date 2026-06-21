"""Graficos para variables categoricas o cualitativas."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class GraficosCualitativos:
    """Graficos para frecuencias, proporciones y tablas de contingencia."""

    def __init__(self, df: pd.DataFrame, cols: Iterable[str] | None = None):
        self.df = df
        self.cols = list(cols) if cols else df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    def countplot(self, col: str, hue: str | None = None, ax=None, **kwargs):
        """Barras con el conteo de cada categoria."""
        ax = ax or plt.gca()
        sns.countplot(data=self.df, x=col, hue=hue, ax=ax, **kwargs)
        ax.set_title(f"Conteo por {col}")
        plt.xticks(rotation=30, ha="right")
        return ax

    def bar_frecuencias(self, col: str, ordenar: bool = True, ax=None, **kwargs):
        """Barras horizontales de frecuencias absolutas."""
        freq = self.df[col].value_counts()
        if ordenar:
            freq = freq.sort_values(ascending=True)
        ax = ax or plt.gca()
        freq.plot(kind="barh", ax=ax, **kwargs)
        ax.set_xlabel("Frecuencia")
        ax.set_title(f"Frecuencias de {col}")
        return ax

    def pie(self, col: str, autopct: str = "%1.1f%%", ax=None, **kwargs):
        """Grafico circular de proporciones."""
        freq = self.df[col].value_counts()
        ax = ax or plt.gca()
        ax.pie(freq, labels=freq.index, autopct=autopct, startangle=90, **kwargs)
        ax.set_title(f"Proporciones de {col}")
        ax.axis("equal")
        return ax

    def barras_apiladas(self, col_x: str, col_hue: str, normalizar: bool = False, ax=None, **kwargs):
        """Barras apiladas para dos variables categoricas."""
        tabla = pd.crosstab(self.df[col_x], self.df[col_hue], normalize="index" if normalizar else False)
        ax = ax or plt.gca()
        tabla.plot(kind="bar", stacked=True, ax=ax, **kwargs)
        titulo = "Proporciones" if normalizar else "Conteos"
        ax.set_title(f"{titulo}: {col_x} por {col_hue}")
        ax.legend(title=col_hue, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=30, ha="right")
        return ax

    def heatmap_contingencia(self, col1: str, col2: str, normalizar: bool = False, annot: bool = True, ax=None, **kwargs):
        """Mapa de calor de la tabla de contingencia."""
        tabla = pd.crosstab(self.df[col1], self.df[col2], normalize="all" if normalizar else None)
        ax = ax or plt.gca()
        sns.heatmap(tabla, annot=annot, fmt=".2f" if normalizar else "d", cmap="Blues", ax=ax, **kwargs)
        ax.set_title(f"Contingencia: {col1} x {col2}")
        return ax

    def mosaic_simple(self, col1: str, col2: str, ax=None):
        """Barras agrupadas como alternativa simple al mosaico."""
        tabla = pd.crosstab(self.df[col1], self.df[col2])
        ax = ax or plt.gca()
        tabla.plot(kind="bar", ax=ax)
        ax.set_title(f"Relacion {col1} vs {col2}")
        ax.legend(title=col2, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=30, ha="right")
        return ax
