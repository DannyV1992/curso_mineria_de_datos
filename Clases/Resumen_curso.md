# Resumen — Clases 2 y 3: Minería de Datos

---

## CLASE 2: Aprendizaje No Supervisado + Pruebas Estadísticas

### Mapa general de ML
- IA > Machine Learning > Deep Learning (subconjuntos anidados)
- ML tiene 3 ramas: **Supervisado** (Classification, Regression), **No Supervisado** (Clustering, Dimensionality Reduction), **Reinforcement Learning**

### Aprendizaje No Supervisado
- Los datos NO tienen etiquetas. El algoritmo busca patrones ocultos.
- Formalmente: no existe una función objetivo y(i). Se busca f: Rn → Z que revele estructura.
- Ejemplos de tareas: clustering, reducción de dimensionalidad, detección de anomalías, estimación de densidad.

### Preprocesamiento (pasos clave)
1. **Valores faltantes**: numéricos → media/mediana; categóricos → moda; avanzado → KNN Imputer. Imputar mal crea "clusters falsos".
2. **Outliers**: eliminar (IQR, Z-score), transformar (log), métodos robustos. K-Means y PCA son especialmente sensibles a outliers.
3. **Codificación**: One-Hot Encoding (más común), Ordinal Encoding (si tiene orden).
4. **Escalado (CRÍTICO)**: sin esto K-Means falla y PCA se distorsiona por escala.
   - Standardization (Z-score): z = (x - media) / desv
   - Min-Max: escala a [0,1]
   - Robust Scaling: usa mediana e IQR (si hay outliers)
5. **Transformaciones de distribución** (para datos sesgados): log transform, Box-Cox, Yeo-Johnson.

### Pruebas estadísticas

| Prueba | Para qué sirve | Tipo | Tamaño muestra |
|--------|----------------|------|----------------|
| **T-Student** | Comparar medias de 2 grupos | Paramétrico | Pequeñas |
| **ANOVA** | Comparar 3+ medias (con estadístico F) | Paramétrico | Moderada/grande |
| **MANOVA** | ANOVA con varias variables dependientes al mismo tiempo | Paramétrico | Moderada/grande |
| **Chi-Cuadrado** | Independencia entre 2 variables categóricas | No paramétrico | — |
| **A/B Testing** | Comparar proporciones (conversión, clics) | Prueba Z | Grande |

#### T-Student
- Compara medias de 2 grupos; permite saber si la diferencia es real o producto del azar.
- Formula: t = (X1 - X2) / (sp * sqrt(1/n1 + 1/n2))
- p < 0.05 → se rechaza H0 (hay diferencia real)
- p >= 0.05 → no se rechaza H0 (diferencia podría ser azar)
- "Rechazar H0" NO significa que H1 sea 100% verdadera.
- "No rechazar H0" NO significa que las medias sean exactamente iguales.
- Es un test **paramétrico**, trabaja con muestras **pequeñas**.

#### ANOVA
- El t-Student compara medias directamente; el ANOVA compara **varianzas entre medias**.
- Usa estadístico F = MS_entre / MS_dentro.
- Si se rechaza H0, usar pruebas post-hoc (Tukey, Bonferroni) para saber cuáles grupos difieren.
- Es un test **paramétrico**.
- Ejemplo en Python: `from scipy.stats import f_oneway`

#### MANOVA
- Extensión del ANOVA: compara **varias variables dependientes al mismo tiempo**.
- Si es significativo, se realizan ANOVAs univariadas o pruebas de Tukey para ver en qué variable están las diferencias.
- Sensible a violaciones de normalidad o varianzas desiguales; requiere muestras **moderadas o grandes**.
- Es un test **paramétrico**.
- Ejemplo en Python: `from statsmodels.multivariate.manova import MANOVA`

#### Chi-Cuadrado
- H0 = las variables categóricas son independientes.
- Si p < 0.05, existe asociación entre las variables.
- Estadístico: chi² = suma((O_ij - E_ij)² / E_ij)
- E_ij = (Total fila_i * Total columna_j) / Total general
- gl = (r - 1)(c - 1)

---

## CLASE 3: Reducción de Dimensionalidad (PCA)

### El problema de la alta dimensionalidad
- A mayor dimensión, los puntos están más dispersos ("maldición de la dimensionalidad").
- Problemas: complejidad computacional, riesgo de sobreajuste, dificultad para visualizar.
- Con muchas variables, los modelos pueden aprender el **ruido** en lugar del patrón real.

### ¿Qué es PCA?
Sintetiza la tabla de datos X en un conjunto más pequeño de variables llamadas **componentes principales** (C1, C2, ...), que son combinaciones lineales de las variables originales, manteniendo la mayor información posible.

- C1 = a11*X1 + ... + a1m*Xm (capta la mayor varianza)
- C2 es ortogonal a C1 y capta la siguiente mayor varianza, etc.
- Analogía visual: proyectar una taza 3D en planos 2D — cada plano es un componente principal.

### Eigenvectores y Eigenvalores
- **Eigenvectores** (v): al aplicar la matriz de covarianza solo cambian de escala, no de dirección (Σv = λv). Forman ejes ortogonales que eliminan correlaciones. Cada componente captura información única, sin redundancia.
- **Eigenvalor** (λ): mide cuánta varianza hay en la dirección de su eigenvector. Mayor λ = más información. PCA conserva los eigenvectores con mayores λ para minimizar pérdida de información.

### Algoritmo PCA (pasos)

| Paso | Acción |
|------|--------|
| 1 | Centrar y estandarizar X (restar la media a cada variable) |
| 2 | Calcular matriz de correlaciones R = (1/n) * X^T * X |
| 3 | Calcular eigenvectores (v_j) y eigenvalores (λ_j) |
| 4 | Ordenar eigenvalores de mayor a menor |
| 5 | Construir matriz V = [v1 \| v2 \| ... \| vm] |
| 6 | Calcular componentes principales: C = X · V |
| 7 | Calidad de individuos (cos²): Q_ir = (C_ir)² / suma(X_ij)² |
| 8 | Coordenadas de variables: T_jr = sqrt(λ_r) * v_jr |
| 9 | Calidad de variables (cos²): S_jr = λ_r * (v_jr)² |
| 10 | Inercias de ejes: I_k = 100 * λ_k / m |

### Por qué estandarizar ANTES de PCA
Sin estandarizar, PCA queda dominado por variables con mayor escala. Con datos estandarizados, todas las variables tienen varianza = 1 y PCA no está sesgado.

Ejemplo: Var(Edad) ≈ 66.67 vs Var(Ingreso) = 600,000,000 → sin estandarizar, Ingreso domina.

### Outputs que interpreta PCA
- **Scree plot**: varianza explicada por componente. Ejemplo: PC1 del decathlon explica 41.2%, PC2 el 18.4%.
- **Plano principal**: individuos proyectados. El cos² indica calidad de representación (rojo = bien representado).
- **Círculo de correlación**: variables como vectores. Misma dirección = correlacionadas positivamente; opuestas = correlación negativa.
- **Biplot**: individuos + variables superpuestos.
- **Contribución (ctr)**: ctr_jk = cos²_jk / λ_k. La suma de todas las contribuciones a un componente = 1 (100%).
- **cos²**: calidad de representación. cos² ≈ 1 = bien representada; cos² ≈ 0 = mal representada.

### Ventajas y Desventajas de PCA

| Ventajas | Desventajas |
|----------|-------------|
| Reduce dimensiones conservando variabilidad | Componentes difíciles de interpretar (son combinaciones lineales) |
| Elimina multicolinealidad | Sensible a escala (hay que estandarizar siempre) |
| Comprime datos | Solo captura relaciones lineales |
| Minimiza pérdida de información | Sensible a outliers |
| Útil en exploración, reducción de ruido, preprocesamiento | No considera variable respuesta (método no supervisado) |

### Usos de PCA
1. Reducción de dimensionalidad
2. Visualización de datos en 2D/3D
3. Reducción de ruido
4. Extracción de características no correlacionadas
5. Manejo de multicolinealidad
6. Compresión de datos
7. Preprocesamiento en aprendizaje automático
8. Detección de anomalías
9. Análisis exploratorio (EDA)
10. Modelado predictivo con menor riesgo de sobreajuste
