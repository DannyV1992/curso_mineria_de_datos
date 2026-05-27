"""Herramientas de EDA: graficos, pruebas estadisticas y regresion."""

from .graficos_cuantitativos import GraficosCuantitativos
from .graficos_cualitativos import GraficosCualitativos
from .test_estadisticos import TestEstadisticos
from .regresion import RegresionLineal, RegresionLogistica

__all__ = [
    "GraficosCuantitativos",
    "GraficosCualitativos",
    "TestEstadisticos",
    "RegresionLineal",
    "RegresionLogistica",
]
