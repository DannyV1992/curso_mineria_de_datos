"""Pruebas estadisticas frecuentes en EDA."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import anova_lm


def _var(col: str) -> str:
    """Nombre seguro para formulas patsy (columnas con espacios o simbolos)."""
    return col if col.isidentifier() else f'Q("{col}")'


class TestEstadisticos:
    """ANOVA, MANOVA, t de Student y chi-cuadrado."""

    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df

    def anova(self, valor: str, grupo: str, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        ANOVA de un factor (comparar medias entre grupos).
        Devuelve tabla ANOVA de statsmodels.
        """
        data = data if data is not None else self.df
        modelo = ols(f"{_var(valor)} ~ C({_var(grupo)})", data=data).fit()
        return anova_lm(modelo)

    def anova_scipy(self, valor: str, grupo: str, data: pd.DataFrame | None = None) -> dict:
        """ANOVA con scipy (estadistico F y p-valor)."""
        data = data if data is not None else self.df
        grupos = [g[valor].dropna().values for _, g in data.groupby(grupo)]
        f_stat, p_val = stats.f_oneway(*grupos)
        return {"F": f_stat, "p_valor": p_val}

    def manova(self, formula: str, data: pd.DataFrame | None = None) -> object:
        """
        MANOVA multivariada.
        Ejemplo de formula: 'sepal_length + sepal_width ~ species'
        """
        data = data if data is not None else self.df
        modelo = MANOVA.from_formula(formula, data=data)
        return modelo.mv_test()

    def ttest_student(
        self,
        valor: str,
        grupo: str,
        grupo_a: str,
        grupo_b: str,
        paired: bool = False,
        data: pd.DataFrame | None = None,
    ) -> dict:
        """Prueba t de Student entre dos grupos."""
        data = data if data is not None else self.df
        a = data.loc[data[grupo] == grupo_a, valor].dropna()
        b = data.loc[data[grupo] == grupo_b, valor].dropna()
        if paired:
            n = min(len(a), len(b))
            stat, p_val = stats.ttest_rel(a.iloc[:n], b.iloc[:n])
            prueba = "t pareada"
        else:
            stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            prueba = "t independiente (Welch)"
        return {
            "prueba": prueba,
            "grupo_a": grupo_a,
            "grupo_b": grupo_b,
            "t": stat,
            "p_valor": p_val,
            "media_a": a.mean(),
            "media_b": b.mean(),
        }

    def chi_cuadrado(self, col1: str, col2: str, data: pd.DataFrame | None = None) -> dict:
        """Chi-cuadrado de independencia entre dos variables cualitativas."""
        data = data if data is not None else self.df
        tabla = pd.crosstab(data[col1], data[col2])
        chi2, p_val, gl, esperados = stats.chi2_contingency(tabla)
        return {
            "chi2": chi2,
            "p_valor": p_val,
            "gl": gl,
            "tabla_observada": tabla,
            "tabla_esperada": pd.DataFrame(esperados, index=tabla.index, columns=tabla.columns),
        }

    def ttest_muestras(self, muestra_a: Sequence[float], muestra_b: Sequence[float], paired: bool = False) -> dict:
        """t de Student directamente sobre dos vectores numericos."""
        a, b = np.asarray(muestra_a), np.asarray(muestra_b)
        if paired:
            stat, p_val = stats.ttest_rel(a, b)
            prueba = "t pareada"
        else:
            stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            prueba = "t independiente (Welch)"
        return {"prueba": prueba, "t": stat, "p_valor": p_val}
