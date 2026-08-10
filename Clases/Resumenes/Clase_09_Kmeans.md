# Clase 9 — Código de clustering jerárquico y K-Means

**Curso:** Minería de Datos · Lead University
**Docente:** Rachit Vargas (MBA | Engineer)
**PDF fuente:** `Clases/PDFs/Clase 9 - Kmeans.pdf`

> La clase abrió con una revisión del roadmap del curso (semanas 9 a 15) y del alcance del examen final (semana 3 a semana 11: PCA, ACS & ACM, T-SNE & UMAP, Clustering jerárquico, Kmeans, DBSCAN y Evaluación de clustering). Luego se retomó el código pendiente de **clustering jerárquico** (dataset de notas de estudiantes, en Colab), sin contraparte en el PDF de esta clase. Tras un receso de 20 minutos, se pasó a la presentación de **K-Means**, que sí sigue la estructura del PDF: introducción al clustering (repaso), creación de clusters y criterio de Fisher, codo de Jambú, y el algoritmo K-Means con su ejemplo numérico paso a paso. El código de K-Means en Python quedó pendiente para la siguiente clase.

---

## 1. Código de clustering jerárquico en Python *(no está en el PDF de esta clase)*

### 1.1 Dos versiones del código

Existen dos notebooks disponibles:

| Versión | Repositorio | Complejidad | Uso recomendado |
|---|---|---|---|
| **Código completo** | GitHub | Complejo, cubre absolutamente todo el análisis de clustering | Para un proyecto real en la empresa que requiera un análisis exhaustivo |
| **Código en Colab** | Colab | Simple, didáctico | El que se usa en clase |

### 1.2 Dataset y librerías

Dataset didáctico: **10 estudiantes × 5 asignaturas** (matemáticas, ciencias, español, historia, educación física).

- `pandas`, `numpy`, `matplotlib` para manejo de datos y visualización.
- El objeto matemático `pi` (de la librería `math`) se usa únicamente para el gráfico de radar (*rarchart*) — no tiene relación con el análisis de clustering en sí.
- `scipy.spatial.distance` → `pdist`: calcula la **matriz de distancias**.
- `scipy.cluster.hierarchy` → `dendrogram`, `linkage` (con los cuatro criterios de enlace: `ward`, `single`, `complete`, `average`) y `fcluster` (asigna la etiqueta de cluster a cada individuo una vez calculadas las distancias).

**Preparación de los datos:** al igual que en PCA o cualquier otro algoritmo de reducción de dimensionalidad, los individuos (aquí los estudiantes) deben pasarse como **índice** del DataFrame (`datos.set_index(...)`), nunca como una columna más — sea el individuo una empresa, un país, un cliente o un comportamiento de tarjeta de crédito.

### 1.3 Matriz de distancias y criterio de enlace

- `pdist(datos, metric='euclidean')` calcula la matriz de distancias; la métrica puede cambiarse por Manhattan, Chebyshev o Minkowski.
- `linkage()` acepta **dos formas de uso** con resultados idénticos: pasarle la matriz de distancias ya calculada con `pdist`, o pasarle los datos crudos directamente (la función internamente calcula la matriz de distancias).
- Se generan los cuatro dendrogramas (`single`, `complete`, `average`, `ward`) sobre el mismo dataset — en este ejemplo particular los tres clusters resultantes se ven prácticamente iguales en los cuatro casos.

### 1.4 Dónde cortar y verificación con PCA

Al trazar una línea horizontal en el dendrograma a la altura 2 se obtienen 3 clusters: {Luis, Sonia}, {Pedro, Inés, Ana, José} y {Lucía, María, Andrés, Carlos}.

- Pregunta en clase: ¿qué pasa si la línea se traza justo por debajo de 2? Luis y Sonia (el par que queda justo bajo la línea) se separan en clusters individuales — cada punto que queda por debajo del corte forma su propio grupo.
- El gráfico de **PCA** (proyección en 2D) sirve para verificar visualmente por qué ciertos individuos se separan: Ana y José aparecen muy próximos entre sí; Luis y Sonia están un poco más alejados del resto, lo cual justifica que puedan separarse en un cluster propio.
- **Advertencia:** el PCA es una reducción de dimensionalidad, por lo que cierta información se pierde. Puede ocurrir que dos puntos que en el dendrograma pertenecen claramente al mismo cluster (p. ej. Lucía y Carlos) no se vean tan cercanos en el plano PCA como se esperaría — es una aproximación, no una representación exacta del espacio completo de variables.

### 1.5 Etiquetado, centroides y caracterización de clusters

- `fcluster(criterio_enlace, t=k, criterion='maxclust')`: recibe el criterio de enlace ya calculado, la métrica, y cuántos clusters (`k`) se necesitan formar — este último es un hiperparámetro adicional, distinto del criterio de enlace.
- El resultado es un array con el número de cluster (0, 1, 2…) asignado a cada individuo. Se concatena como una nueva columna (`datos['Grupos'] = grupos`) al DataFrame original.
- **Los números de cluster son arbitrarios entre ejecuciones:** si se vuelve a correr el modelo, el mismo grupo de estudiantes puede aparecer bajo un número de cluster distinto — la composición del grupo se mantiene, solo cambia la etiqueta numérica.
- A partir de las etiquetas se calculan los **centroides** (promedio por cluster), que sirven de insumo para gráficos de barras y de radar.

**Caracterización de los 3 clusters obtenidos:**

| Cluster | Perfil |
|---|---|
| Buenos en matemáticas y ciencias | Nota baja en educación física, historia y español |
| Buenos en educación física | (Luis, Sonia) |
| Buenos en historia y español | (Andrés, Carlos, Lucía, María) |

Historia y español están correlacionadas entre sí, igual que matemáticas y ciencias (ambas pertenecen a la misma "rama" de asignaturas, junto con física, química y biología).

**Preferencia del docente sobre gráficos de interpretación:** entre el barplot y el radar chart, prefiere el radar chart. Pero cuando la varianza explicada por los componentes principales es buena, prefiere el **PCA** por encima de ambos, porque en un solo gráfico en 2D se ven simultáneamente todas las variables y todos los individuos, con la forma en que se conforman los clusters.

### 1.6 Notebook adicional en GitHub (mismo dataset)

Un segundo notebook, más completo, agrega visualizaciones adicionales sobre el mismo dataset de estudiantes: tamaño de clusters, gráficos de barras alternativos, radar chart, heatmap, PCA, dendrograma, gráficos de correlación, boxplots, violin plots, gráficos apilados, PCA con distintos componentes principales, e indicador de **silueta** (pendiente para la clase de evaluación de clustering).

- **BAT (Bloque de Asignación de Tendencia):** se observan bloques oscuros que confirman que sí hay datos clusterizables.
- **Índice de Hopkins** calculado sobre este dataset: **0.663 (66.3%)** — por encima de 0.5, lo cual indica que sí hay un patrón a clusterizar, aunque no es un resultado excelente, solo bueno.
- El notebook incluye una carpeta de **scripts** con métodos de visualización ya programados: solo hay que pasarles los parámetros que cada método necesita, sin programar nada desde cero.

---

## 2. Introducción al clustering *(repaso)*

Contenido ya visto con profundidad en clases anteriores. El PDF lo retoma como introducción formal a K-Means:

- En muchos problemas se dispone de un conjunto de datos numéricos **y cualitativos**, sin etiquetar (matiz sobre el PDF, que solo menciona "numéricos"), y se busca descubrir estructuras naturales — es la misma lógica del análisis exploratorio de datos.
- Planteamiento matemático: dado $X=\{x_1,x_2,\dots,x_n\}\subset\mathbb{R}^p$, un conjunto finito de observaciones donde cada $x_i$ es el vector de características de un individuo, el objetivo es encontrar una partición $C=\{C_1,\dots,C_K\}$ que agrupe los datos en subconjuntos homogéneos internamente y heterogéneos entre sí.
- **Ventaja de reducir a centroides:** una vez clusterizados los datos, se pueden representar mediante sus centroides ($\mu_k$) en lugar de todas las observaciones individuales — esto facilita el almacenamiento, la visualización y el análisis posterior (p. ej. es más sencillo interpretar un gráfico de barras de centroides que un PCA completo).

---

## 3. Creación de clusters y criterio de Fisher

Todo algoritmo de clustering busca cumplir dos objetivos simultáneos:

| Objetivo | Fórmula | Significado |
|---|---|---|
| **Alta cohesión interna** | $D_{\text{intra}}(C_k)=\frac{1}{\lvert C_k\rvert}\sum_{x_i\in C_k}\lVert x_i-\mu_k\rVert^2$ | Los puntos de un mismo cluster deben ser lo más similares posible (distancia $d$, intraclase). |
| **Alta separación externa** | $D_{\text{inter}}(C_i,C_j)=\lVert \mu_i-\mu_j\rVert^2$ | Los clusters distintos deben ser lo más diferentes posible (distancia $L$, interclase). |

### 3.1 Criterio de inercia (Teorema de Fisher)

De estos dos objetivos nace el **criterio de Fisher** (también llamado criterio de inercia), que combina ambas métricas en un solo índice a maximizar:

$$J = \frac{L}{d}$$

donde $L$ es la distancia intercluster (separación entre centroides) y $d$ es la distancia intracluster (dispersión promedio dentro de cada cluster).

**Ejemplo comparativo entre dos configuraciones:**

| | Caso 1 (buen criterio) | Caso 2 (mal criterio) |
|---|---:|---:|
| $L$ | 10.0 | 2.0 |
| $d_1$, $d_2$ | 0.80, 0.70 | 1.60, 1.40 |
| $d = (d_1+d_2)/2$ | 0.75 | 1.50 |
| $J = L/d$ | **13.33** | **1.33** |

El caso 1 tiene un $J$ mucho mayor: clusters mejor separados (mayor $L$) y más compactos (menor $d$). El caso 2 tiene clusters poco separados y/o poco compactos.

**Sobre la interpretación de $J$ (pregunta en clase):** un valor aislado de $J$ (p. ej. 13.33) **no tiene ninguna interpretación por sí solo** — no existe un umbral universal de aceptación (no es como un p-valor con un corte en 0.05). Cada problema de clustering es distinto (aerolíneas, marketing, comportamiento de clientes), por lo que no se puede fijar un valor de referencia general. La única forma de usar $J$ es **comparando configuraciones entre sí** (distintas métricas de distancia, distintos criterios de enlace, distintos algoritmos) y quedándose con la que produzca el $J$ más grande.

---

## 4. Codo de Jambú

El **codo de Jambú** (*elbow method*) es una técnica para decidir cuántos clusters $K$ son adecuados, analizando cómo cambia la **inercia** al aumentar $K$:

$$W(K) = \sum_{i=1}^{K}\sum_{x\in C_i} \lVert x-\mu_i\rVert^2$$

La inercia mide qué tan compactos son los clusters — es análoga al concepto de distancia intraclase ($x-\mu_i$ es la distancia de cada punto a su centroide).

**Por qué se usa sobre todo con K-Means:** en el clustering jerárquico, $K$ no es obligatorio — el dendrograma se construye completo y luego se decide dónde cortar. En K-Means, en cambio, $K$ es un **parámetro obligatorio**: sin darle un valor de $K$, el algoritmo ni siquiera puede ejecutarse. Por eso el codo de Jambú, aunque puede aplicarse a cualquier algoritmo, se asocia principalmente a K-Means.

**Lectura del gráfico (ejemplo ilustrativo):**

| $K$ | Inercia aproximada | Interpretación |
|---|---:|---|
| 1 | ≈ 7500 | Todos los puntos en un solo grupo; muy lejos del centroide; inercia enorme. |
| 2 | ≈ 3500 | Reducción muy grande de inercia. |
| 3 | ≈ 1500 | Reducción todavía importante — los clusters se vuelven más compactos. |
| 4 | ≈ 1200 | La reducción ya es mínima. |

El "codo" se forma en el punto donde la curva deja de bajar de forma pronunciada (aquí, $K=3$): a partir de ahí, seguir aumentando $K$ ya no reduce significativamente la inercia.

### 4.1 Riesgo de hipersegmentación

Si se sigue bajando más allá del codo, se cae en **hipersegmentación**: se crean clusters que en la realidad no existen, porque las distancias intraclase entre esos "nuevos" grupos son demasiado parecidas.

- En marketing, cada cluster nuevo implica una campaña publicitaria distinta, nuevas políticas de precio y de retención — con presupuesto limitado, agregar un cluster que en realidad no existe significa literalmente **perder dinero** dirigiéndose al mismo tipo de cliente con dos estrategias distintas.
- La fijación de precios también depende del tipo de cliente (p. ej. la galleta Chiqui de Pozuelo no se vende al mismo precio a una tienda china que a Walmart) — elegir bien el $K$ tiene un impacto directo en las ganancias de la empresa.
- Es común que gerentes de marketing con años de experiencia se sorprendan (y no siempre de buena manera) al descubrir, vía clustering, que durante años se dirigían a un número de segmentos distinto del que realmente existe en los datos.

---

## 5. K-Means o K-medias

### 5.1 Definición y diferencia con clustering jerárquico

**K-Means** es un algoritmo de aprendizaje no supervisado usado para clustering. Divide los datos en $K$ grupos similares (clusters), trabajando sobre distancias entre puntos: los puntos cercanos tienden a quedar en el mismo cluster, los lejanos se separan en clusters distintos — la misma lógica que el clustering jerárquico, pero con un mecanismo distinto.

$$\text{Distancia euclidiana} = \sqrt{\sum_{i=1}^n (x_i-y_i)^2}$$

**Naturaleza estocástica vs. determinística:** el clustering jerárquico es determinístico (con excepciones puntuales: si dos individuos tienen exactamente la misma distancia mínima al momento de una fusión, el algoritmo elige entre ellos de forma aleatoria, aunque esto es muy poco frecuente). K-Means, en cambio, **es estocástico por diseño** — inicia colocando los centroides en posiciones aleatorias del espacio, no en el centro real de los datos.

- $K$ representa la cantidad de clusters deseados y debe definirse **antes** de ejecutar el algoritmo — elegir un buen $K$ es una parte crítica del proceso (ver codo de Jambú).
- Cada cluster tiene un **centroide**: el promedio de todos los puntos del grupo, su "centro geométrico".

### 5.2 Algoritmo

```
Algoritmo KMEANS(X, K)
  Inicializar centroides μ1,...,μK aleatoriamente entre los puntos de X
  Repetir
    // Paso 1: Asignación de puntos
    Para cada punto x en X:
      Asignar x al cluster Ci cuyo centroide μi sea el más cercano
    // Paso 2: Recalcular centroides
    Para cada cluster Ci:
      Recalcular μi como el promedio de los puntos en Ci
    // Paso 3: Verificar convergencia
    Si los centroides no cambian (o cambian menos que ε): Detener
    Sino: volver al Paso 1
  Hasta convergencia
  Devolver clusters C1,...,CK y centroides μ1,...,μK
```

El algoritmo repite el ciclo de asignación y recálculo hasta que los centroides dejan de moverse — si un centroide inicial queda mal ubicado (lejos del centro real de su grupo), el algoritmo simplemente necesita más iteraciones para converger.

### 5.3 Ejemplo numérico paso a paso

**Datos (8 puntos):** $(1,5),(2,4),(1,1),(2,1),(6,5),(7,4),(6,1),(7,1)$. $K=2$.

**Inicialización:** centroide 1 = $(2,2)$, centroide 2 = $(6,4)$ (posiciones aleatorias).

**Iteración 1 — Asignación** (distancia euclidiana de cada punto a cada centroide):

| Punto | a $C_1(2,2)$ | a $C_2(6,4)$ | Asignación |
|---|---:|---:|---:|
| $(1,5)$ | 3.16 | 5.10 | 1 |
| $(2,4)$ | 2.00 | 4.47 | 1 |
| $(1,1)$ | 1.41 | 5.10 | 1 |
| $(2,1)$ | 1.00 | 4.47 | 1 |
| $(6,5)$ | 5.00 | 1.00 | 2 |
| $(7,4)$ | 5.10 | 1.00 | 2 |
| $(6,1)$ | 5.00 | 3.00 | 2 |
| $(7,1)$ | 5.39 | 3.16 | 2 |

**Recalcular centroides** (promedio de $x$ y de $y$ de cada grupo): nuevo $C_1 = (1.50, 2.75)$, nuevo $C_2 = (6.50, 2.75)$.

**Iteración 2 — Asignación** (con los centroides recalculados): todos los puntos quedan asignados exactamente igual que en la iteración 1 (Cluster 1: los cuatro puntos de la izquierda; Cluster 2: los cuatro de la derecha).

**Convergencia:** al recalcular los centroides con esta asignación se obtienen los mismos valores ($C_1=(1.50,2.75)$, $C_2=(6.50,2.75)$) — los centroides no cambian, el algoritmo se detiene.

### 5.4 Criterio de parada (número de iteraciones)

Además de la convergencia exacta de los centroides, en la práctica se puede limitar el número de iteraciones como hiperparámetro. La pregunta de cuándo detenerse se resuelve igual que con el codo de Jambú: cuando el **índice de Fisher ($J$) deja de aumentar de forma significativa**.

**Ejemplo ilustrativo:**

| Iteraciones | Índice de Fisher ($J$) |
|---:|---:|
| 50 | ≈ 13 |
| 80 | ≈ 25 (avance importante) |
| 100 | ≈ 25 (sin cambio) |

Pasar de 80 a 100 iteraciones no mejora $J$: seguir iterando solo consume más tiempo de cómputo sin ganancia real. Incluso puede ocurrir que $J$ **empeore** con más iteraciones (p. ej. de 120 iteraciones el $J$ baja de 25.5 a 24), porque el algoritmo sigue recalculando y puede desplazarse en direcciones poco favorables. No existe (al igual que con el umbral de $J$) una regla fija sobre cuántas iteraciones son suficientes — se determina probando y observando en qué punto la curva de $J$ vs. iteraciones se estabiliza.

### 5.5 Ventajas y desventajas

| Ventajas | Desventajas |
|---|---|
| Simple y fácil de implementar | Requiere definir $K$ previamente |
| Rápido y escalable — funciona bien con grandes volúmenes de datos | Sensible a outliers: valores extremos desvían los centroides |
| Convergencia rápida (normalmente pocas iteraciones) | Sensible a la inicialización: distintos centroides iniciales pueden producir resultados distintos |
| Fácil interpretación de los clusters obtenidos | No funciona bien con formas complejas o distribuciones no esféricas |
| Efectivo con clusters compactos y esféricos de tamaño similar | Sensible a la escala de los datos: variables con mayor escala dominan la distancia |

### 5.6 Normalización de los datos

Al igual que todos los algoritmos de clustering, K-Means es sensible tanto a outliers como a la **escala** de las variables — tratar la escala de edades junto con la escala de salarios sin normalizar produce una clusterización distorsionada, porque la variable de mayor magnitud domina el cálculo de distancias. Al estandarizar o normalizar los datos antes de clusterizar, los clusters se forman de una manera notablemente mejor (comparación visual entre K-Means sin normalizar y K-Means normalizado sobre el mismo dataset).

En Python existen dos opciones equivalentes en su uso, aunque con resultados posiblemente distintos:

- `StandardScaler` (de scikit-learn) — estandariza.
- `MinMaxScaler` (de scikit-learn) — normaliza.

---

## Conceptos clave de la clase

- **Código de clustering jerárquico (Colab):** `pdist` calcula la matriz de distancias, `linkage` aplica el criterio de enlace (acepta matriz de distancias o datos crudos), `fcluster` asigna la etiqueta de cluster final; los números de cluster son arbitrarios entre ejecuciones, pero la composición de cada grupo se mantiene.
- El **PCA** es útil para verificar visualmente por qué se separan ciertos individuos en el dendrograma, aunque al ser una reducción de dimensionalidad puede no reflejar el 100% de las distancias del espacio original.
- **Criterio de Fisher (inercia):** $J=L/d$, donde $L$ es la distancia intercluster y $d$ la distancia intracluster; se busca maximizar $J$. No existe un umbral universal de aceptación — solo sirve para comparar configuraciones entre sí.
- **Codo de Jambú:** grafica la inercia $W(K)$ contra la cantidad de clusters $K$; el $K$ óptimo es el punto donde la curva forma un "codo". Se usa sobre todo en K-Means porque, a diferencia del clustering jerárquico, $K$ es un parámetro obligatorio.
- La **hipersegmentación** (crear más clusters de los que realmente existen) tiene un costo de negocio directo: campañas, políticas de precio y estrategias dirigidas a grupos inventados.
- **K-Means:** algoritmo estocástico (inicialización aleatoria de centroides) que itera entre asignar puntos al centroide más cercano y recalcular centroides, hasta que estos dejan de cambiar (convergencia). Requiere definir $K$ de antemano.
- El criterio de parada de las iteraciones de K-Means se decide igual que el número de clusters: cuando el índice de Fisher deja de mejorar de forma significativa.
- **Ventajas de K-Means:** simple, rápido, escalable, convergencia rápida, fácil de interpretar. **Desventajas:** requiere $K$ previo, sensible a outliers, sensible a la inicialización aleatoria, no funciona bien con formas no esféricas, sensible a la escala de los datos (requiere `StandardScaler` o `MinMaxScaler`).

---

## Fuera del PDF — logística, tareas y metodología

- **Roadmap del curso (semanas 9–15):** semana 9 (esta clase) código de clustering jerárquico + K-Means; semana 10 evaluación de clustering (gráficos, métricas, casos de negocio reales); semana 11 simulación de Montecarlo aplicada a finanzas; semanas 12–14 introducción a aprendizaje supervisado (regresión logística, KNN, decision trees) para descargar de contenido el curso siguiente; semana 15 examen final (posiblemente en dos sesiones dado el tamaño del grupo).
- **Alcance del examen final:** únicamente temas de **aprendizaje no supervisado** (semana 3 a semana 10: PCA, análisis de correspondencia simple y múltiple, t-SNE/UMAP, clustering jerárquico, K-Means y evaluación de clustering). Regresión logística, KNN, decision tree y la simulación de Montecarlo **no entran**. Son 5 preguntas en total durante **todo el curso** (no por tema), de formato tipo entrevista, muy similares a las ya resueltas en los quizzes; estudiantes con buena participación y desempeño en quizzes pueden eximirse del examen.
- **Anécdota:** un estudiante de otra consultoría entregó un examen virtual perfecto, pero en una entrevista posterior no supo responder ninguna de las preguntas que ya había "contestado bien" — tuvo que repetir el examen. Sirve de advertencia sobre completar tareas con IA sin entender el contenido.
- **Programar a mano vs. usar IA:** aunque el docente ya no programa en su trabajo actual (los códigos del curso vienen de cursos anteriores a la existencia de ChatGPT), recomienda que los estudiantes programen manualmente al menos una vez en la vida — para entender qué es una librería, un método, una clase, un for, e interpretar bien las salidas de un código. Algunas empresas (ejemplo: Bac Rimatic) todavía piden en las entrevistas programar en vivo sin ninguna IA. Advertencia adicional: subir a repositorios profesionales (GitHub, GitLab) código con descripciones que indiquen haber sido generado con IA (Cursor, etc.) se percibe mal profesionalmente.
- Receso de 20 minutos entre el código de clustering jerárquico y la presentación de K-Means.
- **Quiz la próxima semana** sobre clustering jerárquico y K-Means (aunque el código de K-Means en Python quede pendiente, no es necesario para el quiz).
- Se reforzó la disponibilidad de una **tutoría/repaso los sábados de 9:00 a 10:00 a.m.** para reforzar temas explicados rápido en clase.
