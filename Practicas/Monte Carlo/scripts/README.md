# Tarea 8 — Simulación de Monte Carlo

**Curso:** Minería de Datos | Lead University  
**Carrera:** Ciencia de Datos

---

## Objetivo

Aplicar simulación de Monte Carlo para evaluar la viabilidad financiera de proyectos de inversión
bajo incertidumbre, calculando el Valor Actual Neto (VAN) y cuantificando el riesgo.

---

## Estructura de archivos

```
Tarea_8/
├── datos/                                      # Datasets (no aplica: datos embebidos en notebooks)
├── scripts/
│   ├── __init__.py
│   └── monte_carlo.py                          # Clase MonteCarloSimulator (base: Clase 11)
├── 1_Notebook_Caso1_Tokyo_Gallery.ipynb        # Caso 1: producto en plaza comercial
├── 2_Notebook_Caso2_Transamerica_Airlines.ipynb # Caso 2: compra de aeronave
├── Caso_1/
│   └── Problem Set.pdf                         # Enunciado Caso 1
├── Caso_2/
│   └── Transamerica Airlines Case Study.pdf    # Enunciado Caso 2
└── README.md
```

---

## Descripción de los notebooks

### `1_Notebook_Caso1_Tokyo_Gallery.ipynb`

Evalúa si conviene vender un nuevo producto en Tokyo Gallery durante 5 años.

- **Variable incierta:** precio por unidad ($2.75 / $3.50 / $4.25) con distribución discreta
- **Modelo:** flujos de caja descontados al 10%, VAN sobre inversión de $1.500
- **Preguntas respondidas:**
  1. VAN determinístico con precio esperado
  2. Simulación Monte Carlo ≥ 500 iteraciones
  3. Análisis de cambio en la decisión

### `2_Notebook_Caso2_Transamerica_Airlines.ipynb`

Evalúa si Transamerica Airlines debe comprar un Piper Chieftain ($600.000, 5 años).

- **Variables inciertas (6):** horas de vuelo, % vuelos programados, ocupación, tarifa charter,
  tarifa programado, costo operativo — todas con distribución triangular
- **Modelo:** flujos after-tax con depreciación e impuestos (33%), VAN al 15%
- **Preguntas respondidas:**
  1. ¿Debe comprarse el avión?
  2. Factores de mayor impacto en la rentabilidad (correlaciones y rangos)

---

## Script POO: `MonteCarloSimulator`

Basado en el notebook `Clases/Notebooks/Clase 11 - Simulacion Montecarlo.ipynb`.

| Método | Origen en el notebook de clase |
|--------|-------------------------------|
| `__init__` | `np.random.seed(42)`, `n = 10_000` |
| `agregar_variable_triangular` | `np.random.triangular(...)` |
| `resumen` | Bloque de `print()` — Sección 6 |
| `plot_histograma` | `plt.hist(margen_sim, ...)` — Sección 6 |
| `plot_distribucion_triangular` | Gráfico de distribución triangular — Sección 3 |
| `plot_dispersion_vs_resultado` | Scatter "variable incierta vs margen" — Sección 7 |
| `plot_correlacion` | Barras de correlación de Pearson — Sección 7 |
| `plot_impacto_por_rangos` | Barras y boxplot por rangos — Sección 7 |
| `plot_probabilidad_acumulada` | Curva CDF del margen — Sección 7 |
| `agregar_variable_discreta` | *Extensión propia* — variables con prob. fijas |
| `agregar_variable_uniforme` | *Extensión propia* — distribución uniforme |

---

## Ejecución

### Dependencias

```
numpy
matplotlib
```

### Cómo ejecutar

```bash
# Desde la carpeta Tarea_8/
jupyter notebook 1_Notebook_Caso1_Tokyo_Gallery.ipynb
jupyter notebook 2_Notebook_Caso2_Transamerica_Airlines.ipynb
```
