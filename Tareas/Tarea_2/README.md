# Tarea 2 — EDA y Test Estadísticos
**Lead University · Minería de Datos · Ciencia de Datos**

## Objetivo

Demostrar competencias en la exploración visual de datos (EDA), la validación de supuestos
estadísticos y la implementación de modelos de regresión lineal y logística utilizando Python,
aplicando programación orientada a objetos con las clases del módulo `scripts/`.

---

## Estructura del proyecto

```
Tarea_2/
├── datos/
│   ├── personality_synthetic_dataset.csv   # Dataset de perfiles de personalidad
│   └── daily_food_nutrition_dataset.csv    # Dataset de composición nutricional
├── scripts/
│   ├── __init__.py
│   ├── graficos_cuantitativos.py           # Clase GraficosCuantitativos
│   ├── graficos_cualitativos.py            # Clase GraficosCualitativos
│   ├── test_estadisticos.py                # Clase TestEstadisticos
│   └── regresion.py                        # Clases RegresionLineal y RegresionLogistica
├── EDA_Personalidad.ipynb                  # Análisis del dataset de personalidad
├── EDA_Nutricion.ipynb                     # Análisis del dataset nutricional
├── Tarea_EDA.pdf                           # Enunciado original de la tarea
└── README.md
```

---

## Descripción de los notebooks

### `EDA_Personalidad.ipynb`
Análisis del dataset `personality_synthetic_dataset.csv` (perfiles psicológicos sintéticos).

| Fase | Contenido |
|------|-----------|
| **Fase 1 — EDA Visual** | Pairplot de `social_energy`, `talkativeness`, `deep_reflection` por tipo de personalidad; violin plot de `risk_taking`; heatmap de correlación con identificación de pares \|r\| > 0.7 |
| **Fase 2 — Hipótesis** | Shapiro-Wilk + t de Student o Mann-Whitney U sobre `decision_speed` (Introvert vs Extrovert); MANOVA sobre `empathy`, `listening_skill`, `friendliness` |
| **Fase 3 — Clasificación** | Regresión logística para predecir `leadership > 7.5` con predictores `public_speaking_comfort`, `stress_handling`, `organization`; odds ratios e interpretación |

### `EDA_Nutricion.ipynb`
Análisis del dataset `daily_food_nutrition_dataset.csv` (composición nutricional diaria).

| Fase | Contenido |
|------|-----------|
| **Fase 1 — Perfilamiento** | Barras apiladas de macronutrientes (protein, carbs, fat) para el Top 10 de categorías; scatter plot fat vs calories con curva de tendencia LOWESS |
| **Fase 2 — ANOVA** | ANOVA de una vía para `sugar` entre categorías Fruit y Grain; gráficos post-hoc (violin + boxplot) |
| **Fase 3 — Regresión** | Regresión lineal múltiple `calories ~ protein + carbs + fat + sugar`; interpretación de coeficientes y R² ajustado; diagnóstico de residuos |

---

## Ejecución

### Dependencias

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

### Correr los notebooks

Abrir con Jupyter Lab o VS Code desde la carpeta `Tarea_2/` (importante para que las rutas relativas funcionen correctamente):

```bash
cd Tareas/Tarea_2
jupyter lab
```

Los notebooks importan los scripts locales con:
```python
from scripts import GraficosCuantitativos, TestEstadisticos, RegresionLineal, ...
```

Y cargan los datos con:
```python
df = pd.read_csv('datos/nombre_archivo.csv')
```
