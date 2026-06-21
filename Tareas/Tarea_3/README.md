# Tarea 3 — Reducción de Dimensionalidad (PCA)
**Lead University · Minería de Datos · Ciencia de Datos**

## Objetivo

Aplicar Análisis de Componentes Principales (PCA) sobre tres dominios distintos para identificar patrones ocultos, reducir redundancias y visualizar la dispersión de los datos en espacios de baja dimensión.

---

## Estructura del proyecto

```
Tarea_3/
├── datos/
│   ├── USA_Cars_Dataset.csv              # Vehículos eléctricos (eficiencia, potencia)
│   ├── Country-data.csv                  # Indicadores de desarrollo por país
│   └── Telco_Customer_Churn.csv          # Comportamiento de clientes telecom
├── scripts/
│   ├── __init__.py
│   └── pca_analysis.py                   # Clase PCAAnalysis (prince)
├── 1_PCA_USA_Cars.ipynb                  # PCA sobre dataset automotriz
├── 2_PCA_Country_Data.ipynb              # PCA sobre indicadores de países
├── 3_PCA_Telco_Churn.ipynb               # PCA sobre churn de clientes
├── Tarea_PCA-2.pdf                       # Enunciado original
└── README.md
```

---

## Descripción de los notebooks

### `1_PCA_USA_Cars.ipynb`
PCA sobre el dataset de vehículos eléctricos.

| Visualización | Contenido |
|---------------|-----------|
| **Scree Plot** | Varianza explicada; 3 PCs capturan ≥80 % |
| **Círculo de correlación** | Relación year/autonomía vs consumo energético |
| **Plano principal** | Vehículos coloreados por marca |
| **Biplot** | Superposición individuos + variables |

### `2_PCA_Country_Data.ipynb`
PCA sobre indicadores de desarrollo (Help International).

| Visualización | Contenido |
|---------------|-----------|
| **Scree Plot** | Varianza explicada; 4 PCs capturan ≥80 % |
| **Círculo de correlación** | Oposición child_mort vs gdpp/income |
| **Plano principal** | Países etiquetados en extremos |
| **Biplot** | Segmentación para priorización humanitaria |

### `3_PCA_Telco_Churn.ipynb`
PCA sobre comportamiento de clientes de telecomunicaciones.

| Visualización | Contenido |
|---------------|-----------|
| **Scree Plot** | Varianza explicada; 5 PCs capturan ≥80 % |
| **Círculo de correlación** | Ortogonalidad MonthlyCharges vs tenure |
| **Plano principal** | Clientes coloreados por Churn (Yes/No) |
| **Biplot** | Variables que empujan hacia zona de abandono |

---

## Ejecución

### Dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn prince
```

### Correr los notebooks

Abrir con Jupyter Lab o VS Code desde la carpeta `Tarea_3/`:

```bash
cd Tareas/Tarea_3
jupyter lab
```

Los notebooks importan la clase local con:
```python
from scripts import PCAAnalysis
```
