# Tarea 4 — Análisis de Correspondencia Múltiple (ACM)
**Lead University · Minería de Datos · Ciencia de Datos**

## Objetivo

Aplicar Análisis de Correspondencias Múltiples (ACM) sobre dos datasets reales para explorar relaciones entre variables categóricas, reducir dimensionalidad y generar interpretaciones visuales de los resultados.

---

## Estructura del proyecto

```
Tarea_4/
├── datos/
│   ├── datos_vuelos_comerciales.csv        # Vuelos comerciales (aerolínea, ruta, clase, precio)
│   └── Banking_Dataset.csv                 # Campañas de marketing bancario
├── scripts/
│   ├── __init__.py
│   └── acm_analyzer.py                    # Clase ACMAnalyzer (prince)
├── 1_ACM_Vuelos.ipynb                      # ACM sobre dataset de vuelos (Partes 1-3)
├── 2_ACM_Banking.ipynb                     # ACM sobre dataset bancario (Partes 1-3)
├── 3_Comparacion.ipynb                     # Comparación y discusión (Parte 4)
├── ACM.pdf                                 # Enunciado original
└── README.md
```

---

## Descripción de los notebooks

### `1_ACM_Vuelos.ipynb`
ACM sobre el dataset de vuelos comerciales (Flight Price Prediction).

| Sección | Contenido |
|---------|-----------|
| **Parte 1** | Carga, limpieza, selección de categóricas, discretización de duration/days_left/price |
| **Parte 2** | Ajuste MCA (prince), tabla de inercia, scree plot, elección de dimensiones |
| **Parte 3** | Biplot, mapa de modalidades, contribuciones, cos², perfiles por clase y aerolínea |

### `2_ACM_Banking.ipynb`
ACM sobre el dataset de marketing bancario.

| Sección | Contenido |
|---------|-----------|
| **Parte 1** | Carga, limpieza, selección de categóricas, discretización de age/balance |
| **Parte 2** | Ajuste MCA, tabla de inercia, scree plot, elección de dimensiones |
| **Parte 3** | Biplot, mapa de modalidades, contribuciones, cos², perfiles por suscripción (y) y trabajo |

### `3_Comparacion.ipynb`
Comparación y discusión (Parte 4).

| Sección | Contenido |
|---------|-----------|
| **Punto 14** | Comparación entre datasets (asociaciones, dispersión, interpretabilidad) |
| **Punto 15** | ACM vs PCA (diferencias, variables, casos de uso) |
| **Punto 16** | Perspectiva de negocio (aerolínea y marketing bancario) |

---

## Ejecución

### Dependencias

```bash
pip install pandas numpy matplotlib seaborn prince
```

### Correr los notebooks

Abrir con Jupyter Lab o VS Code desde la carpeta `Tarea_4/`:

```bash
cd Tareas/Tarea_4
jupyter lab
```

Los notebooks importan la clase local con:
```python
from scripts import ACMAnalyzer
```
