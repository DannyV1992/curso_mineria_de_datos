# Clase 2 — Aprendizaje no supervisado

**Curso:** Minería de Datos · Lead University
**Docente:** Rachit Vargas (MBA | Engineer)
**PDF fuente:** `Clases/PDFs/Clase 2 - Intro data mining.pdf`

> Primera clase de contenido técnico. Ubica el aprendizaje no supervisado dentro del ML, define el flujo de trabajo de un proyecto, repasa el preprocesamiento y cubre los cinco tests estadísticos que sirven para decidir **qué variables valen la pena** antes de modelar.

---

## 1. Ubicación disciplinar

Diagrama de conjuntos anidados:

- **Artificial Intelligence** (círculo externo) contiene todo. Es un campo enorme.
- **Machine Learning** es un subconjunto de la IA.
- **Deep Learning** es un subconjunto del ML. Su diferencia respecto a las demás ramas: además de los algoritmos tradicionales, usa **redes neuronales**.
- **Natural Language Processing** vive dentro de la IA y se **solapa parcialmente** con ML y Deep Learning. Es la rama de los LLM (lo que hoy usa ChatGPT).

Minería de datos y machine learning son prácticamente lo mismo: la minería es un poco más estadística, pero **muchos de los algoritmos son idénticos** y se comparten entre ambas.

### Las tres ramas del Machine Learning

| Rama | Tareas | Aplicaciones |
|------|--------|--------------|
| **Unsupervised Learning** | *Dimensionality Reduction* | Meaningful Compression, Structure Discovery, Big Data Visualisation, Feature Elicitation |
| | *Clustering* | Recommender Systems, Targetted Marketing, Customer Segmentation |
| **Supervised Learning** | *Classification* | Image Classification, Customer Retention, Diagnostics, Identity Fraud Detection |
| | *Regression* | Advertising Popularity Prediction, Weather Forecasting, Market Forecasting, Estimating life expectancy, Population Growth Prediction |
| **Reinforcement Learning** | — | Real-time decisions, Game AI, Robot Navigation, Skill Acquisition, Learning Tasks |

**Supervisado — dos enfoques según el tipo de variable objetivo:**
- **Regresión:** problemas cuantitativos. Precio de una acción, tipo de cambio, precio de una casa, salario de un nuevo empleado.
- **Clasificación:** problemas cualitativos, binarios o multiclase. Si una transacción es fraudulenta o no, si el próximo cliente hace clic o no, clasificación de imágenes, diagnósticos, *customer churn*.

**Reforzado:** aprende **por prueba y error**. A diferencia del supervisado —que recibe un dataset etiquetado y estructurado—, aquí el modelo predice, y si falla se lo **castiga**; a partir de ese castigo recalibra sus resultados. Es la rama detrás de los modelos conversacionales actuales.

> El curso se centra en la rama **no supervisada**: reducción de dimensionalidad y clustering. El aprendizaje supervisado (regresión y clasificación) corresponde a **Minería de Datos 2**.

### Hoja de ruta del curso

| Bloque | Algoritmos |
|--------|-----------|
| **Reducción de dimensionalidad** | PCA (primero, la siguiente clase), Análisis de Correspondencias Simple y Múltiple, t-SNE, UMAP |
| **Clustering** | K-means, clustering jerárquico, DBSCAN (clustering por **densidad**, no por distancias) |

---

## 2. ¿Qué es el aprendizaje no supervisado?

### Analogía
En aprendizaje **supervisado** se proporcionan al algoritmo ejemplos con etiquetas (p. ej., "manzana", "naranja"): el dataset ya trae la respuesta explicada para que el algoritmo aprenda de ella. En aprendizaje **no supervisado** solo se observan los datos y el objetivo es **descubrir su estructura subyacente** — no existe esa etiqueta.

### Definición técnica

Es una rama del aprendizaje automático en la cual el conjunto de datos

$$X = \{x^{(1)}, x^{(2)}, \dots, x^{(m)}\} \subset \mathbb{R}^n$$

**no tiene etiquetas asociadas**. El objetivo es encontrar patrones, estructuras o relaciones ocultas en los datos.

Formalmente:
- No existe una función objetivo con "respuesta correcta" $y^{(i)}$.
- El algoritmo busca una función $f$ tal que

$$f : \mathbb{R}^n \rightarrow \mathcal{Z}$$

que revele alguna estructura en los datos, donde $\mathcal{Z}$ puede representar **grupos, componentes, representaciones latentes**, entre otros.

### Ejemplos de tareas
Clustering, reducción de dimensionalidad, detección de anomalías, aprendizaje de representaciones, estimación de densidad. De todas ellas, **clustering y reducción de dimensionalidad son los dos enfoques principales**.

### Lectura de la Figura 1.1 — agrupamiento en $\mathbb{R}^2$

Puntos en un espacio de dos dimensiones ($x_1$, $x_2$) que el algoritmo particiona en 3 grupos según **similitud**, sin conocer etiquetas.

Aterrizado a un caso concreto: si los ejes son *grasas* y *calorías*, cada punto es un alimento (pescado, manzana, etc.). No existe la etiqueta "ultraprocesado / no ultraprocesado" ni "comida dañina / no dañina" — solo se agrupan los puntos. Un clúster puede caracterizarse por ser alto en grasas y otro por tener calorías altas sin tanta grasa. Lo que se busca es **qué caracteriza a cada clúster y en qué se diferencia de los demás**.

Cada punto es la unidad de análisis: países, personas, alimentos. En un clustering real **todos los puntos terminan asignados a algún clúster** — ninguno se descarta por quedar fuera de la elipse del dibujo. La figura es solo ilustrativa.

### Por qué se reduce la dimensionalidad

El ojo humano llega como máximo a $\mathbb{R}^3$, y un gráfico tridimensional ya es incómodo de leer: la mayoría de las personas prefieren $\mathbb{R}^2$. Cuando el dataset tiene decenas o cientos de variables, visualizarlo es imposible.

**Ejemplo canónico:** los dígitos escritos a mano. Cada píxel de la imagen es una columna y su valor va de 0 a 255 → una tabla con cientos de columnas.

Lo que hace un algoritmo como PCA es **comprimir esa información a $\mathbb{R}^2$ perdiendo la menor cantidad de información posible**, para poder visualizar e interpretar datos que antes no se podían mirar.

---

## 3. Estructura de un proyecto de ML no supervisado (nivel senior)

Flujo de trabajo recomendado, con retroalimentación desde la última etapa hacia la primera. Es muy similar a **CRISP-DM**.

### 1. Definición del problema y objetivos
- Entender el contexto de negocio.
- Formular hipótesis sobre la estructura subyacente.
- Definir métricas de éxito (intrínsecas o extrínsecas).

### 2. Recolección y entendimiento de datos
- Recolectar datos relevantes.
- Análisis exploratorio (EDA).
- Identificar tipos de variables, distribuciones, outliers y valores faltantes.

### 3. Preprocesamiento de datos
- Limpieza de datos.
- Imputación de valores faltantes.
- Escalado / Normalización / Estandarización.
- Transformaciones (log, Box-Cox).
- Reducción de dimensionalidad preliminar (opcional).

### 4. Modelado no supervisado (descubrimiento)
- Selección de técnicas apropiadas: K-Means, GMM, DBSCAN, HDBSCAN, PCA, t-SNE, Autoencoders.
- Entrenamiento / ajuste.
- Selección de hiperparámetros: $k$, $\varepsilon$, `min_samples`.

El objetivo de fondo: partir de un dataset del que **no se tiene ninguna idea de cómo se comporta** y descubrir patrones estructurales que se puedan entregar como *insight* a la Junta Directiva o a los *stakeholders*.

> Ejemplo sencillo: notas de estudiantes en matemáticas, ciencias, español, historia y educación física. Sin saber nada de antemano, se busca quiénes son los mejores y los peores en cada asignatura y qué grupos se forman.

### 5. Evaluación e interpretación
- Métricas intrínsecas: Silhouette, Calinski-Harabasz, Davies-Bouldin, Inercia.
- Validación con conocimiento del dominio (si aplica).
- Interpretación de clusters / componentes / patrones.

**Esta etapa es la que marca la diferencia profesional.** La mayoría del mercado aplica los algoritmos sin saber si el resultado es bueno o malo: reportan "esto encontré en los datos" sin verificar que el hallazgo sea concluyente, que la data se prestara para el análisis o que el algoritmo elegido sirviera para el problema.

Todo modelo debe evaluarse: si la cantidad de clusters elegida fue la correcta, si la convergencia tuvo sentido. Una conclusión válida también puede ser que **los datos no tienen potencial para llegar a nada concluyente**.

### 6. Comunicación y entrega de resultados
- Visualización de resultados: UMAP, t-SNE, heatmaps — lo más amigables posible.
- Storytelling basado en datos.
- Recomendaciones accionables para el negocio.
- Documentación de hallazgos, decisiones y **limitaciones** del modelo.

---

## 4. Estructura de proyecto (vista de software)

Organización de código y datos para proyectos reproducibles y escalables:

```
nombre-del-proyecto/
├── data/
│   ├── raw/                      # Datos originales sin modificar (nunca se tocan)
│   │   └── .gitkeep
│   └── processed/                # Datos limpios, transformados, escalados o normalizados
│       └── .gitkeep
├── notebooks/
│   ├── 01_exploracion.ipynb      # Análisis exploratorio, modelado
│   ├── 02_modelado.ipynb         # e interpretación en cuadernos de Jupyter
│   └── 03_interpretacion.ipynb
├── src/
│   ├── __init__.py               # Código fuente para producción
│   ├── preprocess.py             # y reutilización
│   ├── model.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── utils.py
├── models/
│   ├── kmeans_model.pkl          # Modelos entrenados y objetos serializados
│   └── scaler.pkl
├── reports/
│   └── figures/                  # Resultados, visualizaciones y reportes generados
│       ├── elbow_method.png
│       ├── clusters_pca.png
│       └── perfil_clusters.png
│   └── informe.pdf
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Descripción, instrucciones y resumen de hallazgos
```

**Recorrido de los datos por la estructura:**

1. `data/raw/` guarda la extracción tal cual, **sin limpiar**. Es inmutable: toda transformación produce archivos nuevos.
2. `data/processed/` recibe el resultado de la curación: detección y relleno de nulos, normalización, estandarización, transformaciones (Box-Cox).
3. `notebooks/` consume esos datos ya procesados.
4. `models/` almacena los modelos entrenados en formato `.pkl`, listos para **producción**: un K-means ya entrenado que el área de operaciones o marketing puede reutilizar sin volver a ajustarlo.
5. `reports/figures/` guarda las salidas: formación de clústeres, perfiles de clúster, método del codo.

Esta estructura es análoga a la que se usa en ingeniería de datos (arquitectura *medallion*, ETL/ELT en *data lakes* y nube): **un archivo, una misión específica**. Un archivo extrae, otro procesa, otro carga. Entregar un solo archivo que hace todo el trabajo no es aceptable en un equipo real.

### Notebooks vs. scripts: por qué la POO

Los **notebooks son cuartos de trabajo autorreproducibles**: se ejecutan celda por celda y devuelven resultados inmediatos, ideales para experimentación rápida y recolección de resultados. Un archivo `.py` hay que ejecutarlo completo para ver algo, lo que no es amigable para compartir con un *stakeholder*.

El problema aparece cuando se escribe todo el código hacia abajo dentro del notebook: se vuelve ilegible e inmanejable. La alternativa es dejar en `src/` las clases y métodos (por ejemplo, un `barplot` dentro de `visualization.py`) e **invocarlos desde el notebook**. El resultado es un notebook bien estructurado, con pocas líneas, que sí se puede entregar a la Junta Directiva.

Python es un lenguaje **orientado a objetos** — ese es su paradigma y su ventaja frente a R, donde la codificación es lineal. Programar sin aprovecharlo es desperdiciar la herramienta. Los beneficios: código mejor estructurado, más legible y proyectos mejor organizados.

Escribir un módulo propio equivale a **crear una librería desde cero**: al importar `metodos_cuantitativos` se dispone de todos los gráficos ya programados, igual que al importar scikit-learn.

**Sobre el entorno:** funciona igual en Colab que en VS Code. En Colab se puede subir un archivo `.py` (no `.ipynb`) al entorno y hacer `from metodos_cuantitativos import GraficosCuantitativos`. La diferencia práctica es que Colab ya trae las librerías instaladas y en local hay que provisionar el ambiente.

---

## 5. Preprocesamiento de datos

### 5.1 Manejo de valores faltantes (imputación)

| Tipo de variable | Técnica |
|------------------|---------|
| **Numéricos** | media / mediana |
| **Categóricos** | moda |
| **Más avanzado** | KNN Imputer (usa la vecindad del registro para estimar el valor) |

Ejemplo categórico: en una columna con `rojo, azul, azul, —, rojo, azul`, el nulo se reemplaza por **azul**, el valor que más se repite.

También hay quienes usan modelos de IA para rellenar valores perdidos.

**¿Por qué no simplemente eliminar las filas con nulos?** Es una decisión de criterio, y depende de cuánta información se pierde:

- Un dataset de un millón de filas con dos líneas incompletas → se eliminan sin problema, no altera nada.
- Un dataset donde eliminar los nulos deja solo el **50 % de la información** → crítico, no se puede.
- **Criterio práctico:** hasta un **10 % de valores nulos** se puede quitar. Por encima de ese umbral hay que aplicar técnicas de imputación para no perder los datos.

**Cuidado con imputar por promedio:** el promedio es lo más instintivo, pero es sensible a outliers. Con la columna `3.5, 2.7, 3.1` el promedio es ≈ 3; si entra un valor de 55, el promedio se dispara a 20 y el valor imputado deja de ser representativo. Por eso la **mediana** o **KNN** suelen ser más realistas y confiables.

> ⚠️ **En no supervisado, imputar mal puede crear "clusters falsos".**

### 5.2 Tratamiento de outliers

Muy importante porque **afectan las distancias**.

- Eliminación (IQR, Z-score).
- Transformación (log).
- Métodos robustos (vía distribuciones normales).

Un solo punto muy alejado **distorsiona el clúster**: al tratarlo o eliminarlo, la estructura que emerge es más robusta y tiene más sentido.

> ⚠️ **K-Means y PCA son especialmente sensibles a esto.**

La razón: son algoritmos que **dependen de distancias**. Con un punto muy lejano, los parámetros e hiperparámetros del modelo se desestabilizan y los resultados se sesgan. Lo mismo aplica a KNN.

### 5.3 Codificación de variables categóricas

Los algoritmos de minería de datos son **algoritmos matemáticos**: solo procesan números, nunca letras.

| Método | Cuándo usarlo | Ejemplo |
|--------|---------------|---------|
| **One-Hot Encoding** (más común) | Sin orden intrínseco | Rojo → `[1,0,0]`, Azul → `[0,1,0]`, Verde → `[0,0,1]` |
| **Ordinal Encoding** | Si tiene orden | Pequeño → 1, Mediano → 2, Grande → 3 |

Es un paso obligatorio tanto en modelos supervisados como no supervisados.

### 5.4 Escalado de variables (CRÍTICO)

Uno de los pasos más importantes. Los modelos se confunden cuando las variables viven en escalas distintas:

> **Edad** varía entre 0 y 100. **Salario** (en colones) varía entre 400 000 y 5 000 000. No son valores comparables.

**Standardization (Z-score):**
$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

**Min-Max Scaling:**
$$\tilde{x}_{ij} = \frac{x_{ij} - \min_j}{\max_j - \min_j} \in [0, 1]$$

**Robust Scaling** (si hay outliers):
$$x_{ij}^{rob} = \frac{x_{ij} - \text{mediana}_j}{IQR_j}$$

> ⚠️ **Sin escalado:** K-Means **falla**; PCA **se distorsiona** (la varianza queda dominada por la escala de las variables, no por su información).

Visualmente: la nube de puntos sin escalar aparece estirada y sesgada; escalada queda **compacta**.

### 5.5 Transformaciones de distribución

Cuando los datos están sesgados:

**Log transform:**
$$x' = \log(x + c), \quad c > 0$$

**Box-Cox:**
$$x'(\lambda) = \begin{cases} \dfrac{x^\lambda - 1}{\lambda}, & \lambda \neq 0 \\ \log(x), & \lambda = 0 \end{cases}$$

**Yeo-Johnson** (admite valores negativos):
$$x'(\lambda) = \begin{cases}
\dfrac{(x+1)^\lambda - 1}{\lambda}, & x \geq 0,\ \lambda \neq 0 \\
\log(x+1), & x \geq 0,\ \lambda = 0 \\
-\dfrac{(-x+1)^{2-\lambda} - 1}{2 - \lambda}, & x < 0,\ \lambda \neq 2 \\
-\log(-x+1), & x < 0,\ \lambda = 2
\end{cases}$$

Secuencia visual: distribución sesgada → después de log transform → más simétrica (ideal).

**Ayudan a que:**
- Las distancias sean más representativas.
- PCA capture mejor la estructura.

### 5.6 Por qué importa conocer la distribución

Saber qué distribución siguen los datos permite cuantificar probabilidades.

> **Ejemplo — tipo de cambio.** Distribución normal centrada en 500 colones, con extremos en 400 y 600. Los dos cuadrantes centrales concentran alrededor del 64 % de la probabilidad; las **colas largas son los outliers**, los valores más atípicos.
>
> - ¿Probabilidad de que el tipo de cambio esté entre 500 y 550 los próximos meses? ≈ 30 %.
> - ¿Probabilidad de que supere los 600? ≈ 2 %.

Sin esa lectura no se puede interpretar correctamente el resultado de un modelo.

---

## 6. Tests estadísticos

### Para qué sirven en este curso

Todo esto se hace para **descubrir patrones accionables**: algo cuantificable que se le pueda decir al negocio y que nadie más estaba viendo.

Ejemplos de hallazgos contraintuitivos:
- Los clientes que caen en mora se caracterizan por tener tres hijos.
- En un dataset de precios de vivienda en Illinois, el precio dependía **más de la cantidad de baños que de las habitaciones** — lo contrario de lo que responde unánimemente cualquier persona a la que se le pregunte.

Con 500 variables que hay que analizar simultáneamente, esto **no se puede hacer en Excel**, ni con tablas dinámicas ni con gráficos de barras. De ahí la necesidad de algoritmos capaces de procesar Big Data y rescatar los patrones estructurales.

Estos tests funcionan mejor para aprendizaje **supervisado**, pero son parte de la exploración de datos y hay que dominarlos: sirven para saber **qué variables tienen impacto y cuáles no antes de meterlas a un modelo**.

> Si al modelo se le mete basura, devuelve basura.

**El problema que evitan:** un *stakeholder* pide un modelo predictivo, se le meten 10 variables a ojos cerrados y el modelo devuelve una métrica de 0.04 — no predice nada. Al analizar las distribuciones se descubre que **todas las variables se comportan igual entre las clases**: ninguna separa. El modelo nunca tuvo los insumos para clasificar. Muchas veces se pide un modelo predictivo sin tener los datos que lo hagan posible (como pedir una serie de tiempo de un producto que todavía no salió al mercado).

### Tabla resumen: qué test usar

| Test | Compara | Tipo de variable | Nº de grupos | Nº de variables dependientes |
|------|---------|------------------|--------------|------------------------------|
| **T-Student** | Medias | Numérica | 2 | 1 |
| **ANOVA** | Varianzas entre medias | Numérica | 3+ | 1 |
| **A/B Testing (Z)** | Proporciones | Categórica (binaria) | 2 | 1 |
| **MANOVA** | Vectores de medias | Numérica | 2+ | 2+ |
| **Chi Cuadrado** | Frecuencias | Categórica | 2+ | — (asociación) |

El **A/B testing** depende de la prueba t o del ANOVA para que sus datos sean cuantificables; en sí mismo es una prueba estadística, pero se usa sobre todo para **prototipado**.

---

### 6.1 T-Student (comparación de medias)

- Es una prueba estadística que compara las **medias de dos grupos**.
- Permite saber si la diferencia observada es **real o producto del azar**.
- Trabaja con muestras **pequeñas** — una desventaja hoy, que se trabaja con Big Data.
- Es un test **paramétrico**.

**Supuestos:** datos ~ Normal, varianzas iguales (o usar corrección de Welch).

**Cómo funciona:** toma la media de la primera distribución, la media de la segunda, las desviaciones combinadas y el tamaño de muestra, y determina si esas dos medias son **significativamente diferentes** o no.

#### Paramétrico vs. no paramétrico

Distinción que reaparece en todos los algoritmos de ML:

| Tipo | Datos |
|------|-------|
| **Paramétrico** | Siguen una distribución normal |
| **No paramétrico** | No siguen ninguna distribución normal (p. ej. sesgada, exponencial) |

La regresión lineal es un modelo **paramétrico**: rinde bien cuando los datos son aproximadamente normales y se degrada cuando no lo son. Con distribuciones fuertemente asimétricas corresponde usar **Mann-Whitney** en lugar del t-test.

#### El caso de uso típico

Dos grupos de clientes y la variable *salario*. La pregunta: ¿el salario determina si la persona paga o no paga el préstamo?

El test devuelve dos distribuciones. Si el Grupo 1 (los que no pagan) tiene salarios de 1 000 a 2 000 dólares y el Grupo 2 (los que pagan) de 1 500 a 5 000, el test **confirma si esa brecha entre grupos es significativa** — es decir, si el salario es una variable crítica para predecir el pago.

#### Lo que aporta frente a una tabla dinámica

Graficar o pivotear puede llevar al mismo resultado aparente, pero **lo estadístico controla la variabilidad**. Puede ocurrir que el grupo que no pagó lo hiciera por otra variable $x_2$ no contemplada, y que el salario no explicara nada: el resultado no sería concluyente.

> **Correlación espuria.** Un empresario vende helados y en ese mismo periodo se registra la mayor cantidad de ataques de tiburón. Las dos variables no tienen ninguna relación: en esa ciudad se había estrenado una película famosa y la gente se metió al mar a surfear. Variante conocida del mismo caso: más venta de helados coincide con más quemaduras de sol.
>
> Sin pruebas estadísticas no hay forma de descartar estas coincidencias. Son ellas las que dicen si algo es concluyente y si realmente depende de la variable analizada.

#### Lectura de los gráficos de densidad

| Escenario | p-valor | Conclusión |
|-----------|---------|-----------|
| Grupos A y B superpuestos | $p = 0.755$ | Medias iguales. La variable **no separa** los grupos en absoluto |
| Grupos C y D desplazados | $p = 0.00000$ | Medias significativamente distintas. La variable **sí separa** |

El criterio de decisión, igual que en la mayoría de tests univariados y en regresión, es $p < 0.05$.

> **Ejemplo con dominio real.** Grupos "tiene cáncer" / "no tiene cáncer" y la variable *cantidad de familiares con antecedentes de cáncer* (0 a 5). Las medias aparecen completamente separadas: entre más familiares con antecedentes, más probable el diagnóstico. La herencia familiar es entonces una variable **crucial** para el problema — exactamente el tipo de dato valioso que un científico de datos le pide a ingeniería de datos que extraiga.

**Fórmulas.**

Dos muestras independientes:
$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}}}, \qquad gl = n_1 + n_2 - 2$$

Forma con desviación combinada $s_p$:
$$t = \frac{\bar{X}_1 - \bar{X}_2}{s_p\sqrt{\dfrac{1}{n_1} + \dfrac{1}{n_2}}}$$

donde $\bar{X}_1, \bar{X}_2$ son las medias de los grupos, $s_p$ la desviación combinada y $n_1, n_2$ los tamaños de muestra.

Dos muestras relacionadas (pareadas):
$$t = \frac{\bar{D}}{s_D / \sqrt{n}}, \qquad gl = n - 1$$

**Usos:** comparar antes/después, grupo tratamiento vs. control (media).

Lo que hay que interpretar no es la fórmula sino el **p-valor** y los gráficos: quien no es estadístico ni matemático va a entender el gráfico, no el $p$.

**Implementación:**
```python
from scipy.stats import ttest_ind

ventas_A = [100, 110, 95, 105, 98, 102, 97]
ventas_B = [120, 125, 130, 128, 122, 118, 127]

t, p = ttest_ind(ventas_A, ventas_B)

print(f"t = {t:.2f}, p = {p:.5f}")

if p < 0.05:
    print("Diferencia significativa: la Estrategia B mejora las ventas.")
else:
    print("No hay diferencia significativa entre las estrategias.")
```
```
t = -6.12, p = 0.0001
Diferencia significativa: la Estrategia B mejora las ventas.
```

#### Interpretación del p-valor

| Resultado del test | Qué significa estadísticamente | Interpretación práctica |
|--------------------|--------------------------------|-------------------------|
| $p < 0.05$ → Rechazamos $H_0$ | Hay evidencia de diferencia | Las ventas de A y B son distintas |
| $p \geq 0.05$ → No rechazamos $H_0$ | No hay evidencia de diferencia | Las ventas son similares; cualquier diferencia podría ser por azar |

> - "Rechazar $H_0$" **no** significa que $H_1$ sea 100 % verdadera, sino que hay evidencia suficiente para creer que existe una diferencia.
> - "No rechazar $H_0$" **no** significa que las medias sean exactamente iguales, sino que no hay suficiente evidencia para afirmar que difieren.

---

### 6.2 A/B Testing (prueba de hipótesis para conversión)

Compara dos versiones (A vs. B) en una métrica (conversión, clics, ingresos). Usualmente para **proporciones**.

**Prueba Z para dos proporciones:**
$$Z = \frac{\hat{p}_A - \hat{p}_B}{\sqrt{\hat{p}(1-\hat{p})\left(\dfrac{1}{n_A} + \dfrac{1}{n_B}\right)}}, \qquad \hat{p} = \frac{x_A + x_B}{n_A + n_B}$$

- $H_0$: $p_A = p_B$ (no hay diferencia)
- $H_1$: $p_A \neq p_B$ ($>$ o $<$ según la métrica)

**Usos:** evaluar cambios en producto, marketing, diseño web. En la práctica se pide sobre todo en **marketing**.

**Caso completo:** un equipo diseña dos versiones de una página web. Durante 3 semanas se reparten **10 000 usuarios**: 5 000 consumen el Diseño A y 5 000 el Diseño B. Resultado: A convierte al **2 %**, B al **3 %**.

La pregunta que responde el test: ese **punto porcentual de diferencia, ¿es estadísticamente significativo** como para afirmar que el Diseño B es mejor? Sin el test, la comparación de porcentajes por sí sola no autoriza la conclusión.

---

### 6.3 ANOVA (análisis de varianza)

- Permite comparar **tres o más medias (usando varianza)** para determinar si al menos una de ellas es significativamente diferente.
- Permite saber si la diferencia observada es **real o producto del azar**.
- Es un test **paramétrico**.

**La diferencia clave con el t-Student:** el t-Student compara **medias** y solo admite **dos** distribuciones; el ANOVA compara **varianzas entre medias** y admite **tres o más grupos**. Sigue siendo **una sola variable** a la vez.

$H_0$: todas las medias son iguales.

**Estadístico F:**
$$F = \frac{MS_{entre}}{MS_{dentro}} = \frac{SS_{entre}/(k-1)}{SS_{dentro}/(N-k)}$$

donde:
$$SS_{entre} = \sum_{i=1}^{k} n_i (\bar{X}_i - \bar{X})^2, \qquad SS_{dentro} = \sum_{i=1}^{k}\sum_{j=1}^{n_i} (X_{ij} - \bar{X}_i)^2$$

$$gl = (k-1,\ N-k)$$

**Visualización:** boxplots comparativos por grupo.

> Si se rechaza $H_0$, usar pruebas **post-hoc** (Tukey, Bonferroni, etc.) para identificar *qué* grupos difieren.

**Ejemplo — tres estrategias de precios.** Las densidades de *ventas promedio* muestran a la Estrategia B claramente desplazada hacia valores más altos (~115), mientras A y C se solapan alrededor de 100-105. El ANOVA confirma si dentro de esas tres estrategias hay diferencias.

```python
from scipy.stats import f_oneway

A = [100, 105, 98, 110, 102]
B = [115, 118, 120, 117, 119]
C = [95, 92, 100, 97, 94]

F, p = f_oneway(A, B, C)
print(f"F = {F:.2f}, p = {p:.5f}")

if p < 0.05:
    print("Hay diferencias significativas entre las estrategias.")
else:
    print("No hay diferencias significativas entre las estrategias.")
```
```
F = 49.36, p = 0.00001
Hay diferencias significativas entre las estrategias.
```

El criterio de evaluación es idéntico al del t-Student: $p < 0.05$.

---

### 6.4 MANOVA (análisis multivariado de varianza)

- Extensión del ANOVA: en lugar de comparar **una sola variable dependiente**, compara **varias variables dependientes al mismo tiempo** (y también más grupos).
- Si el MANOVA es significativo, se realizan **ANOVAs univariadas o pruebas de Tukey** para ver en qué variable están las diferencias.
- Es un test **paramétrico**.
- **Sensible a violaciones de normalidad o varianzas desiguales; requiere muestras moderadas o grandes.**

Progresión de los tres tests:

| Test | Variables | Grupos |
|------|-----------|--------|
| t-Student | 1 | 2 |
| ANOVA | 1 | 3+ |
| MANOVA | $x_1, x_2, \dots, x_n$ | 2+ |

$H_0$: los vectores de medias son iguales.

**Estadísticos comunes:**

| Estadístico | Fórmula |
|-------------|---------|
| Wilks' Lambda | $\Lambda = \dfrac{\lvert W \rvert}{\lvert T \rvert}$ |
| Pillai's Trace | $V = \text{tr}(E(E+H)^{-1})$ |
| Hotelling's Trace | $T^2 = \text{tr}(HE^{-1})$ |
| Roy's Largest Root | $\theta = \lambda_{max}(HE^{-1})$ |

donde $H$ = matriz de suma de cuadrados y productos **entre** grupos, y $E$ = matriz de suma de cuadrados y productos **dentro** de grupos.

**Usos:** comparar grupos en varias métricas simultáneamente (ej. rendimiento en matemáticas y lenguaje).

**Ejemplo ilustrado — Drug X vs. Drug Y** en tres variables a la vez: *Blood Pressure*, *Cholesterol* y *Heart Rate*. Los boxplots se organizan por variable (no por grupo, como en los ejemplos anteriores) y muestran cómo se comporta cada grupo dentro de cada una.

```python
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

data = {
    "Rendimiento": [80, 82, 78, 75, 77, 79, 88, 90, 92],
    "Estres":      [60, 58, 65, 70, 72, 68, 55, 52, 50],
    "Grupo":       ["A","A","A","B","B","B","C","C","C"]
}

df = pd.DataFrame(data)

manova = MANOVA.from_formula("Rendimiento + Estres ~ Grupo", data=df)
print(manova.mv_test())
```
```
Multivariate linear model
--------------------------------------------------------
          Value  Num DF  Den DF  F Value  Pr > F
--------------------------------------------------------
Grupo  Wilks' lambda  0.08   4.0   10.0    5.85   0.010
--------------------------------------------------------
```

> ⚠️ **Limitación importante.** La salida del MANOVA da el valor F, el p-valor, los grados de libertad y la varianza, pero **no indica en qué variable está la diferencia**. Para localizarla hay que visualizar los grupos o recurrir a ANOVAs univariadas o pruebas de Tukey.

---

### 6.5 Chi Cuadrado ($\chi^2$)

Responde: **¿dos variables categóricas están relacionadas o son independientes?**

#### Por qué no sirve una matriz de correlación

Una matriz de correlación (valores de 0 a 1, diagonal en 1, simétrica) **solo funciona con variables numéricas**. Si lo que se quiere medir es la relación entre dos variables **categóricas**, un gráfico de correlación no tiene sentido: el chi cuadrado es el test que permite llegar a esa conclusión.

**Caso de e-commerce.** Tabla cruzada de dos variables cualitativas — *hizo clic / no hizo clic* contra *compró / no compró*:

| | Hizo clic | No hizo clic |
|---|---:|---:|
| **Compró** | 10 | 2 |
| **No compró** | 2 | 50 |

La asociación parece clara —quien hace clic compra—, pero se trata de dos variables categóricas. **Cuantificar esa relación es lo que hace el chi cuadrado.**

#### a) Independencia en tablas de contingencia
$H_0$: las variables son independientes.

$$\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \qquad E_{ij} = \frac{(\text{total fila}_i)(\text{total columna}_j)}{N}$$

$$gl = (r-1)(c-1)$$

#### b) Bondad de ajuste
$H_0$: la distribución observada sigue la esperada.

$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}, \qquad gl = k - 1 - p$$

($p$ = parámetros estimados)

**Usos:** analizar asociación entre variables categóricas, verificar si una distribución sigue lo esperado.

> El desarrollo matemático completo se retoma en la clase de **Análisis de Correspondencias Simple y Múltiple**, donde el $\chi^2$ es la base del método. Lo esencial por ahora es saber **cuándo** aplicarlo.

#### Ejemplo trabajado: género vs. deporte favorito

**1. Datos observados** — encuesta a 200 personas:

| Género \ Deporte | Fútbol | Básquetbol | Tenis | Total |
|------------------|-------:|-----------:|------:|------:|
| Masculino | 70 | 40 | 20 | 130 |
| Femenino | 30 | 25 | 15 | 70 |
| **Total** | **100** | **65** | **35** | **200** |

**2. Hipótesis**
- $H_0$: el género y el tipo de deporte favorito son independientes (no existe asociación).
- $H_1$: el género y el tipo de deporte favorito no son independientes (existe asociación).

**3. Frecuencias esperadas** — $E_{ij} = \dfrac{(\text{total fila}_i)(\text{total columna}_j)}{\text{Total general}}$

| Género \ Deporte | Fútbol | Básquetbol | Tenis | Total |
|------------------|-------:|-----------:|------:|------:|
| Masculino | 65.00 | 42.25 | 22.75 | 130 |
| Femenino | 35.00 | 22.75 | 12.25 | 70 |
| **Total** | **100** | **65** | **35** | **200** |

**4. Contribuciones por celda al estadístico**

| Género \ Deporte | Fútbol | Básquetbol | Tenis | Total |
|------------------|-------:|-----------:|------:|------:|
| Masculino | 0.385 | 0.120 | 0.332 | 0.837 |
| Femenino | 0.714 | 0.219 | 0.516 | 1.448 |
| **Total** | **1.099** | **0.339** | **0.848** | **2.286** |

$$\chi^2_{calc} = 2.286$$

**5. Distribución y decisión**
- Nivel de significancia: $\alpha = 0.05$
- Grados de libertad: $gl = (r-1)(c-1) = (2-1)(3-1) = 2$
- Valor crítico: $\chi^2_{0.05,\,2} = 5.991$

Como $\chi^2_{calc} = 2.286 < \chi^2_{crit} = 5.991$ → **no se rechaza $H_0$**.

**6. p-valor**
$$p\text{-valor} = P(\chi^2 \geq 2.286 \mid gl = 2) = 0.319$$

Como $p = 0.319 > \alpha = 0.05$ → **no se rechaza $H_0$**.

**Conclusión:** no hay evidencia suficiente para afirmar que existe asociación entre el género y el tipo de deporte favorito.

---

## 7. Demostración en código — EDA sobre el dataset de personalidad

Aplicación de todo lo anterior sobre `personality_synthetic_dataset.csv` usando la estructura POO: el notebook importa desde `scripts/` las clases `GraficosCuantitativos`, `GraficosCualitativos`, `TestEstadisticos`, `RegresionLineal` y `RegresionLogistica`. Cada gráfico se produce con **dos líneas de código** porque toda la implementación vive en los módulos.

### El dataset

Datos sintéticos de un supuesto experimento de psicología. Cada persona está categorizada en `personality_type` con **tres clases**: introvertido, extrovertido y un perfil intermedio. El resto de columnas son **índices** de habilidades: energía social, desempeño hablando en público, nivel de reflexión, habilidad de escucha, empatía, creatividad, organización, liderazgo, aversión al riesgo, curiosidad, entre otras.

**Dimensiones:** 20 000 filas × 31 columnas.

**Conteo por clase:** 6 857 extrovertidos, 6 570 introvertidos, 6 573 intermedios — clases prácticamente balanceadas.

### Preparación
- Carga con `pd.read_csv`.
- `personality_type` se convierte a tipo categoría.
- Se crea una variable derivada binaria: "alta energía" si el valor supera la **mediana** de la columna.
- `print(df.shape)` y conteos por clase para saber con qué se está trabajando.

### Lectura de los gráficos

**Matriz de correlación (heatmap).** Solo variables numéricas. Correlaciones altas: `social_energy` con la habilidad de hablar en público (≈ 0.71) y con liderazgo (≈ 0.70). Aparece también una **correlación negativa** entre empatía y liderazgo — débil, pero presente, y conceptualmente interesante.

**Scatter plot.** `social_energy` contra desempeño hablando: los extrovertidos se concentran en el cuadrante de alta energía social y buen desempeño; los introvertidos en el opuesto; el perfil intermedio queda en el medio. Trazando una recta de regresión se ve una **relación positiva**: a mayor índice de energía social, mejor desempeño.

**Boxplot.** Las bolitas fuera de los bigotes son **outliers**: personas con comportamiento atípico. Hay extrovertidos con notas muy bajas en variables donde se esperaría lo contrario — por eso son casos raros.

**Gráfico de violín.** Similar al boxplot, pero además de la dispersión muestra la **distribución**: permite ver si es normal o si está sesgada hacia un lado. Los dos lados del violín son espejo, no información adicional.

**Pair plot.** Matriz de scatter plots de todas las combinaciones de variables, con las distribuciones por clase en la diagonal. En la variable *liderazgo* se ve directamente que la distribución de los extrovertidos está **desplazada a la derecha** respecto a los introvertidos — la separación que después confirma el test estadístico.

**Gráficos cualitativos.** Barras de frecuencia y gráfico de pastel para los porcentajes de cada clase.

### Tests aplicados

**ANOVA** — habilidad de hablar en público vs. `personality_type` (3 grupos): las tres distribuciones están notablemente separadas y $p \approx 0.0 < 0.05$. La variable **tiene peso** para predecir si una persona es introvertida o extrovertida: es exactamente el tipo de variable que sirve para entrenar un modelo predictivo.

**MANOVA** — `social_energy` y una segunda variable contra `personality_type`. Todos los p-valores por debajo de 0.05: existe un criterio de separación entre los perfiles. Interesa el p-valor más que los estadísticos de variación.

**Chi cuadrado** — sobre las variables categóricas del dataset.

### Regresión lineal múltiple

**$R^2$ — la inercia explicada por el modelo.** Cuánto poder explicativo tiene el modelo sobre la variable a predecir. Entre más cercano a 1, mejor; llegar a 1 es prácticamente imposible y fuera del laboratorio incluso 0.99 es excepcional. Se lee como un porcentaje: 0.66 → el modelo explica el 66 %.

> Si se emite una conclusión con un $R^2$ de 0.00001, todo lo que se afirme es mentira.

**Ejemplo de lo que aporta y lo que no.** Para predecir si a un cliente le gusta un plato de comida, las variables *servicio al cliente*, *sabor* y *precio* tienen buena explicatividad y dan un $R^2$ razonable (≈ 0.70). Agregar *si ese día llovió* no aporta nada: el $R^2$ pasa de 0.70 a 0.71.

**Estadístico F.** Cumple una función parecida al $R^2$: indica qué tan usable y concluyente es el modelo en conjunto. Se evalúa igual, con $p < 0.05$.

**Coeficientes.** La parte más valiosa de la salida.
- **Coeficiente positivo:** relación directa. A mayor `social_energy`, mejor desempeño hablando.
- **Coeficiente negativo:** relación inversa. Ser introvertido resta **−2.48 puntos** en el índice de desempeño hablando; ser extrovertido lo aumenta.
- **Cada coeficiente tiene su propio p-valor** y hay que revisarlo uno por uno. En el modelo del demo, los coeficientes de introvertido y extrovertido son concluyentes ($p < 0.05$), pero los de `social_energy` y empatía superan 0.05: sobre esas dos **no se puede emitir juicio**, aunque el modelo global sea bueno.

---

## Conceptos clave de la clase

- El **aprendizaje no supervisado** busca $f: \mathbb{R}^n \to \mathcal{Z}$ sin etiquetas $y$; no hay "respuesta correcta" contra la cual medirse.
- Se reduce la dimensionalidad porque el ojo humano no pasa de $\mathbb{R}^3$: comprimir a $\mathbb{R}^2$ perdiendo la menor información posible es lo que hace visualizable un dataset de cientos de columnas.
- **El escalado es crítico:** sin él K-Means falla y PCA se distorsiona. La razón de fondo es que estos algoritmos **dependen de distancias**.
- **Los outliers afectan las distancias** y distorsionan clusters; también arruinan la imputación por promedio (usar mediana o KNN).
- **Imputar mal en no supervisado crea clusters falsos.** Criterio práctico: hasta un 10 % de nulos se puede eliminar; por encima, hay que imputar.
- El modelo se evalúa **siempre**. Aplicar el algoritmo sin verificar si el resultado es concluyente es lo que hace la mayoría del mercado.
- Los tests estadísticos sirven para saber **qué variables separan las clases antes de modelar**. Si ninguna variable es significativa, el modelo predictivo no es viable, por más que lo pida el negocio.
- Un test estadístico **controla la variabilidad**; un gráfico o una tabla dinámica no descartan correlaciones espurias.
- Todos los tests vistos (t, ANOVA, MANOVA) son **paramétricos**: asumen normalidad. Con distribuciones asimétricas usar alternativas no paramétricas (Mann-Whitney).
- El p-valor no prueba $H_1$; solo mide la evidencia contra $H_0$. Criterio: $p < 0.05$.
- **Correlación solo aplica a variables numéricas.** Para asociación entre categóricas, chi cuadrado.
- El MANOVA detecta que hay diferencia, pero **no dice en qué variable está**.

---

## Fuera del PDF — logística, tarea y metodología

### Requisitos previos
Dos videos vistos antes de la clase: **programación orientada a objetos** y **regresión en Excel**. La regresión lineal y logística son algoritmos **supervisados** (necesitan etiqueta para pronosticar), pero se usan mucho para **exploración de datos** — de ahí su lugar en este curso.

### Metodología de trabajo
- Todo el curso se trabaja con **programación orientada a objetos**: clases y métodos en módulos `.py` separados, invocados desde el notebook.
- Entorno libre: Colab, VS Code u otro IDE.
- De aquí en adelante los entregables deben tener sus clases **por aparte**. Ejemplo: si se entrega un K-means, el método del codo va en otro módulo.
- **Excepción para la Tarea 2:** quien todavía no se sienta cómodo con POO puede entregarla de forma tradicional.
- Formato de entrega: notebook, HTML o enlace al repositorio.
- Se permite usar IA para que explique el código de clase; la idea es replicarlo y reutilizarlo en la tarea.

### Tarea de la semana — dos datasets

**Dataset 1 — `personality_synthetic_dataset.csv`** (sintético, sobre introvertidos y extrovertidos):
1. **EDA:** distribuciones cruzadas con *pair plot*; violín o boxplot para analizar outliers; estructura de correlación con matriz de correlación / heatmap para las cuantitativas y chi cuadrado para las categóricas. Con interpretación en términos de negocio y psicología.
2. **Tests estadísticos:** t-Student, ANOVA, MANOVA y chi cuadrado, para determinar si el comportamiento de las variables varía significativamente entre grupos (introvertidos vs. extrovertidos).
3. **Clasificación con regresión logística:** cuantificar liderazgo alto/bajo — **liderazgo alto cuando el índice supera 7.5**. Tres variables explicativas: `public_speaking_comfort`, `stress_handling` y `organization`. Interpretar los coeficientes y determinar el impacto del manejo del estrés sobre la probabilidad de ser líder, evaluando coeficiente y p-valor.

**Dataset 2 — Food Nutrition Dataset** (datos reales):
1. **EDA:** gráficos de barras de proteína, carbohidratos y grasas para las 10 primeras categorías de alimentos; *scatter plot* de grasas vs. calorías con línea de regresión lineal para observar la tendencia.
2. **Tests y distribuciones:** ANOVA; distribuciones univariadas y bivariadas — por ejemplo, la distribución del azúcar en las categorías *fruity* y *grain*.
3. **Regresión lineal múltiple:** predecir calorías a partir de proteína, carbohidratos, grasas y azúcar. Determinar por coeficientes qué variable aporta más e interpretar el $R^2$.

### Material de apoyo
Repositorio de la Semana 2 en GitHub con el notebook `eda_demo.executed.ipynb` (ya ejecutado, con resultados) y la carpeta `scripts/`. No resuelve la tarea, pero usa los mismos datos y el mismo código que se necesita para resolverla.

### Otros
- Las próximas clases empiezan a las 7:00 p. m. y duran 3 horas; algunas terminarán antes según el tema.
- Hay disponibilidad para sesiones extra de refuerzo si el grupo lo coordina.
- El curso se vuelve progresivamente más difícil: mucha álgebra, ecuaciones de distancia y sobre todo programación.
