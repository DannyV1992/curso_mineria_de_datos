"""Regresion lineal multiple y logistica orientadas a interpretacion del modelo."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import logit, ols


class RegresionLineal:
    """Regresion OLS con statsmodels: coeficientes, bondad de ajuste y diagnosticos."""

    def __init__(self, formula: str, data: pd.DataFrame):
        self.formula = formula
        self.data = data
        self.modelo = None
        self.resultados = None

    def ajustar(self):
        self.modelo = ols(self.formula, data=self.data)
        self.resultados = self.modelo.fit()
        return self

    def resumen(self) -> str:
        self._verificar_ajuste()
        return str(self.resultados.summary())

    def coeficientes(self) -> pd.DataFrame:
        self._verificar_ajuste()
        tabla = pd.DataFrame(
            {
                "coef": self.resultados.params,
                "error_std": self.resultados.bse,
                "t": self.resultados.tvalues,
                "p_valor": self.resultados.pvalues,
                "ic_inf": self.resultados.conf_int()[0],
                "ic_sup": self.resultados.conf_int()[1],
            }
        )
        return tabla

    def bondad_ajuste(self) -> pd.Series:
        self._verificar_ajuste()
        return pd.Series(
            {
                "R2": self.resultados.rsquared,
                "R2_ajustado": self.resultados.rsquared_adj,
                "AIC": self.resultados.aic,
                "BIC": self.resultados.bic,
                "RMSE": np.sqrt(self.resultados.mse_resid),
            }
        )

    def grafico_regresion(self, var_x: str, var_y: str | None = None, ax=None):
        """Recta ajustada para una variable explicativa (marginal)."""
        self._verificar_ajuste()
        var_y = var_y or self.resultados.model.endog_names
        ax = ax or plt.gca()
        sns.regplot(data=self.data, x=var_x, y=var_y, ax=ax, scatter_kws={"alpha": 0.6})
        ax.set_title(f"Ajuste marginal: {var_y} ~ {var_x}")
        return ax

    def grafico_predichos_vs_observados(self, ax=None):
        self._verificar_ajuste()
        ax = ax or plt.gca()
        y_hat = self.resultados.fittedvalues
        y_obs = self.resultados.model.endog
        ax.scatter(y_obs, y_hat, alpha=0.7)
        lims = [min(y_obs.min(), y_hat.min()), max(y_obs.max(), y_hat.max())]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel("Observado")
        ax.set_ylabel("Predicho")
        ax.set_title("Observado vs predicho")
        return ax

    def diagnostico_residuos(self):
        """Residuos, QQ-plot y escalograma para evaluar supuestos."""
        self._verificar_ajuste()
        resid = self.resultados.resid
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        sns.histplot(resid, kde=True, ax=axes[0])
        axes[0].set_title("Residuos")

        stats.probplot(resid, plot=axes[1])
        axes[1].set_title("QQ-plot")

        sns.scatterplot(x=self.resultados.fittedvalues, y=resid, ax=axes[2], alpha=0.7)
        axes[2].axhline(0, color="red", ls="--")
        axes[2].set_xlabel("Valores ajustados")
        axes[2].set_ylabel("Residuos")
        axes[2].set_title("Residuos vs ajustados")

        plt.tight_layout()
        return fig

    def _verificar_ajuste(self):
        if self.resultados is None:
            raise RuntimeError("Primero ejecuta ajustar().")


class RegresionLogistica:
    """Regresion logistica con statsmodels: odds ratios y clasificacion basica."""

    def __init__(self, formula: str, data: pd.DataFrame):
        self.formula = formula
        self.data = data
        self.modelo = None
        self.resultados = None

    def ajustar(self):
        self.modelo = logit(self.formula, data=self.data)
        self.resultados = self.modelo.fit(disp=0)
        return self

    def resumen(self) -> str:
        self._verificar_ajuste()
        return str(self.resultados.summary())

    def coeficientes(self) -> pd.DataFrame:
        self._verificar_ajuste()
        tabla = pd.DataFrame(
            {
                "coef": self.resultados.params,
                "error_std": self.resultados.bse,
                "z": self.resultados.tvalues,
                "p_valor": self.resultados.pvalues,
                "odds_ratio": np.exp(self.resultados.params),
                "ic_inf_or": np.exp(self.resultados.conf_int()[0]),
                "ic_sup_or": np.exp(self.resultados.conf_int()[1]),
            }
        )
        return tabla

    def bondad_ajuste(self) -> pd.Series:
        self._verificar_ajuste()
        return pd.Series(
            {
                "pseudo_R2": self.resultados.prsquared,
                "AIC": self.resultados.aic,
                "BIC": self.resultados.bic,
                "log_likelihood": self.resultados.llf,
            }
        )

    def curva_roc_aprox(self, var_x: str, ax=None):
        """Relacion entre probabilidad predicha y variable explicativa (exploratorio)."""
        self._verificar_ajuste()
        prob = self.resultados.predict()
        ax = ax or plt.gca()
        sns.scatterplot(x=self.data[var_x], y=prob, ax=ax, alpha=0.6)
        ax.set_xlabel(var_x)
        ax.set_ylabel("Probabilidad predicha")
        ax.set_title(f"Probabilidad vs {var_x}")
        return ax

    def matriz_confusion_basica(self, umbral: float = 0.5) -> pd.DataFrame:
        self._verificar_ajuste()
        y_true = self.resultados.model.endog
        y_pred = (self.resultados.predict() >= umbral).astype(int)
        tabla = pd.crosstab(y_true, y_pred, rownames=["Real"], colnames=["Predicho"])
        return tabla

    def _verificar_ajuste(self):
        if self.resultados is None:
            raise RuntimeError("Primero ejecuta ajustar().")
