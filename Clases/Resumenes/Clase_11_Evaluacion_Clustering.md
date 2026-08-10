# Clase 11 — Evaluación de algoritmos de clustering

**Curso:** Minería de Datos · Lead University
**Docente:** Rachit Vargas (MBA | Engineer)
**PDF fuente:** `Clases/PDFs/Clase 11 - Evaluacion clustering.pdf`

> La clase repasó primero la tarea anterior (regresión lineal sobre el caso del aceite de mostaza) y luego cubrió cómo evaluar si un dataset es clusterizable y cómo calibrar (tunear) los hiperparámetros de un algoritmo de clustering, cerrando con una demostración de código y trabajo en grupos sobre un dataset de vinos.

---

## 1. Introducción: ¿tiene sentido clusterizar estos datos?

- Antes de aplicar cualquier algoritmo de clustering es fundamental preguntarse si los datos realmente poseen una **estructura agrupable**.
- No todos los datasets contienen grupos naturales: en muchos casos los datos están distribuidos de forma aleatoria, uniforme o sin separaciones claras entre observaciones.
- Si se ignora este análisis previo, el algoritmo va a producir clusters artificiales aunque no exista ningún patrón real — el algoritmo dirá que hay un cluster donde no existe ninguno. En contextos de productos defectuosos, marketing o finanzas, actuar sobre ese hallazgo falso es literalmente "meter las patas".
- Por esta razón, evaluar la **clusterabilidad** de un problema es una etapa más del análisis exploratorio de datos, tan importante como cualquier otro chequeo previo.
- Ventajas prácticas de hacer esta evaluación antes de programar el algoritmo:
  - Reduce el riesgo de construir modelos inútiles.
  - Evita desperdiciar recursos computacionales y tiempo de codificación (antes de la IA generativa, programar un modelo desde cero para descubrir que no servía era un costo real).
  - Permite validar que los patrones encontrados son reales y no producto del azar.
  - Mejora la interpretación y la confiabilidad de los resultados.
- Métodos disponibles para esta evaluación: **estadístico de Hopkins**, **VAT**, **Gap Statistic** y análisis visual mediante **PCA** y **UMAP**. Todos responden la misma pregunta clave: ¿los datos contienen grupos naturales, o cualquier cluster obtenido sería artificial?

## 2. Estadístico de Hopkins

- Mide la tendencia natural de los datos a formar grupos, comparando la distribución real contra una distribución completamente aleatoria generada en el mismo espacio.
- Procedimiento:
  1. Generar $n$ puntos aleatorios dentro del mismo espacio que ocupan los datos reales.
  2. Para cada punto (tanto real como aleatorio) calcular la distancia a su vecino más cercano dentro de los datos reales.
  3. Sumar por separado esas distancias: $\sum x_i$ para los puntos reales y $\sum y_i$ para los puntos aleatorios.
  4. Calcular $H = \dfrac{\sum y_i}{\sum x_i + \sum y_i}$.
- Si los datos tienen agrupamientos naturales, los puntos reales están más cerca entre sí que los puntos generados al azar, por lo que $\sum x_i$ resulta pequeño y $H$ se acerca a 1.

| Valor de H | Interpretación |
|---|---|
| ≈ 0.5 | Datos distribuidos aleatoriamente, no clusterizables |
| > 0.7 | Evidencia de estructura de clusters |
| Cercano a 1 | Fuerte tendencia a formar clusters |
| Cercano a 0 | Distribución uniforme o regular |

- Ejemplo trabajado en clase: un dataset con clusters reales dio $H = 0.91$ (tanto estadística como visualmente clusterizable); el mismo cálculo sobre datos sin estructura dio $H \approx 0.49$–$0.5$, evidencia de que no vale la pena clusterizar ni perder tiempo en ello.
- Regla práctica: valores iguales o inferiores a 0.5 indican que clusterizar no tiene sentido.

## 3. Silhouette Score (índice de silueta)

- Mide dos cosas para cada punto: qué tan cerca está de su propio cluster y qué tan lejos está de los demás clusters — es decir, combina distancia **intracluster** y distancia **intercluster**, igual que el valor $J$ (teorema de Fisher) visto anteriormente para elegir el número de clusters.

$$s = \frac{b - a}{\max(a, b)}$$

donde $a$ es la distancia promedio al propio cluster y $b$ es la distancia promedio al cluster vecino más cercano.

- Rango de valores: de −1 a 1. Si $a > b$ (el punto está más cerca de otro cluster que del suyo), el índice se vuelve negativo — un resultado desastroso para ese punto.
- **Gráfico de silueta:** en el eje Y aparece cada cluster y, dentro de cada uno, una barra horizontal ordenada con el índice de silueta de cada individuo (de menor a mayor). Entre más grande y sólido el bloque de un cluster, mejor conformado está. La forma irregular ("de cuchillo") de estos bloques no es un error: refleja que dentro de un mismo cluster hay individuos bien clasificados y otros ambiguos, mal clasificados o confundidos con otro cluster.
- Regla práctica del umbral: a partir de un índice de silueta de **0.4** ya conviene empezar a preocuparse por la calidad de esa clusterización.
- Un punto puede tener un índice de silueta bajo (incluso siendo un outlier respecto al centroide) y aun así estar correctamente asignado, porque a pesar de estar lejos sigue estando más cerca de ese cluster que de cualquier otro.
- Si dos clusters vecinos concentran muchos individuos mal clasificados entre sí (p. ej. dos clusters muy pegados), fusionarlos suele **subir el índice de silueta general** por encima del umbral recomendado — coherente con lo que ya sugería el codo de Jambú sobre el número óptimo de clusters.
- **Qué hacer con los individuos mal clasificados** depende del contexto de negocio, sobre todo en escenarios B2B (empresa que le vende a otra empresa): identificar a ese cliente específico importa, porque de su ubicación en un cluster u otro depende qué política de precios o campaña de marketing se le aplica. Dos caminos:
  - Re-clusterizar (otro algoritmo o el mismo con más iteraciones) para ver si el punto se ubica mejor.
  - Reasignarlo manualmente con criterio experto y contexto de negocio, cuando el algoritmo no puede capturar relaciones que el analista sí conoce (p. ej. una sucursal pequeña de una cadena grande se comporta distinto a un comercio independiente, pero conviene agruparla con las demás sucursales de su misma marca).
- El índice de silueta no es perfecto, pero junto con las distancias intra/intercluster es, en la práctica, de las pocas métricas transversales disponibles para evaluar clustering.

## 4. VAT (Visual Assessment of Tendency) y otros objetos visuales

- El **VAT** persigue el mismo objetivo que el estadístico de Hopkins —determinar si los datos son clusterizables— pero de forma visual: calcula las distancias entre todos los puntos, reordena esa matriz de distancias y construye un mapa de calor (*heatmap*).
- Si los datos tienen clusters naturales, al reordenar la matriz aparecen **bloques oscuros cuadrados sobre la diagonal principal** (submatrices bien definidas). Si no hay estructura de agrupamiento, el heatmap no muestra ningún manchón reconocible en la diagonal, solo ruido disperso.
- Otros objetos visuales que cumplen la misma función de diagnóstico:
  - **Dendrograma** (clustering jerárquico): permite ver directamente cuántos grupos distintos se forman al cortar el árbol.
  - **Gráfico de PCA con elipses de cluster**: proyecta los datos en dos componentes y dibuja la región que ocupa cada cluster, útil para verificar visualmente si las agrupaciones están separadas o se solapan.

## 5. Calibración de modelos de clustering (hyperparameter tuning)

- "Calibración de modelos" es la traducción al español de **hyperparameter tuning**: ajustar los parámetros de un algoritmo (número de clusters, métrica de distancia, densidad, criterio de enlace, etc.) para obtener grupos compactos, separados e interpretables.
- Sin una buena calibración, el algoritmo puede generar clusters artificiales o mezclar patrones distintos, produciendo resultados engañosos que nunca llegan a la mejor versión posible de esos clusters.
- Calibrar mejora la estabilidad y confiabilidad del modelo y ayuda a detectar estructuras reales en los datos — algo fundamental en aplicaciones como segmentación de clientes, medicina o detección de fraude, donde una decisión incorrecta (p. ej. no detectar un producto médico defectuoso por no identificarlo como outlier) puede tener un alto impacto.

### 5.1 Por qué el tuning es raro en aprendizaje no supervisado

- El hyperparameter tuning nace y se practica mucho más en aprendizaje **supervisado**, porque ahí existen métricas de evaluación estandarizadas y objetivas: accuracy global, accuracy por clase, F1 score, área bajo la curva, entre otras. Esas métricas permiten comparar sin ambigüedad distintos modelos (p. ej. SVM con 0.90 de accuracy, Random Forest con 0.95, Decision Tree con 0.80 → se elige Random Forest) y, dentro de un mismo modelo, distintas configuraciones de sus parámetros.
- En aprendizaje **no supervisado** no existe una "clase real" contra la cual comparar el resultado del cluster — la calidad de un agrupamiento es subjetiva, no hay una métrica de accuracy verdadera. Las únicas herramientas disponibles son el índice de silueta y las distancias intra/intercluster (el valor $J$ / teorema de Fisher).
- Por esta misma razón no existen librerías de tuning automatizado para clustering equivalentes a **Grid Search** o **Random Search** (ampliamente usadas en supervisado): en clustering hay que programar manualmente los bucles de búsqueda (fuerza bruta con `for` anidados) para probar combinaciones de hiperparámetros y quedarse con la que maximice el índice de silueta.
- En el siguiente curso (Minería de Datos Avanzada) este proceso ya aparece más estandarizado y automatizado gracias a esas librerías.

### 5.2 Código en Python — tuning manual, sin contraparte directa en el PDF

- **K-Means:** se define un rango de valores para los hiperparámetros a tunear — número de clusters (`k`), tipo de inicialización (`init`: `k-means++` o `random`), número de iteraciones (`max_iter`) y semilla (`random_state`) — y se recorren todas las combinaciones con bucles anidados, calculando en cada una el `silhouette_score` y guardando la mejor configuración. K-Means en scikit-learn **no tiene un hiperparámetro de métrica de distancia** (siempre usa distancia euclidiana); incluirlo en la búsqueda es un error de la demo, no algo que deba tunearse.
- **Clustering jerárquico aglomerativo:** se tunean el criterio de enlace (`ward`, `complete`, `average`/salto promedio) y la métrica de distancia (euclidiana, Manhattan, coseno). El enlace `ward` solo es compatible con distancia euclidiana; el resto de enlaces sí admite las demás métricas.
- **Dataset de demo — Customer Data:** variables numéricas de clientes (edad, ingreso anual, *spending score*, años de experiencia laboral, tamaño de familia), estandarizadas con `StandardScaler` antes de clusterizar. El codo de Jambú sugirió un punto de inflexión en **4 clusters**.
- Se evaluaron en total **43 configuraciones** combinando K-Means y clustering jerárquico. El mejor modelo resultó ser K-Means con $k=4$, `init='k-means++'`, `n_init=10`, `max_iter=100`, con un índice de silueta de solo **0.14** — evidencia de que este dataset es difícil de clusterizar (silueta baja en general para todas las configuraciones probadas).
- Perfiles de los 4 clusters resultantes del Customer Data:

| Cluster | Perfil |
|---|---|
| 0 | Mayor edad, pero la menor experiencia laboral del grupo |
| 1 | Mayor gasto (*spending score*), menor ingreso, menor experiencia y menor edad |
| 2 | Mayor experiencia laboral, buen ingreso anual, edad alta |
| 3 | Mayor ingreso anual, mayor edad, familias más numerosas — mayor estabilidad financiera |

- **Tarea de la clase — dataset de vino:** 13 variables físico-químicas (alcohol, magnesio, intensidad de color, flavonoides, prolina, pH, entre otras) usadas típicamente para benchmarking de clustering. Como estas variables requieren conocimiento de enología para interpretarse, se permite apoyarse en IA generativa únicamente para la interpretación de resultados, no para evitar programar.
- Interpretación de negocio de un resultado de ejemplo con 3 clusters sobre vinos tintos: cluster con menor densidad de alcohol (vinos más suaves), cluster intermedio, y cluster de vinos más robustos y fuertes al paladar — la interpretación depende del tipo de uva, y podría variar frente a vinos blancos (menor densidad de alcohol pero sensación en el paladar a veces igual de fuerte).

---

## Conceptos clave de la clase

- **Clusterabilidad:** antes de clusterizar hay que verificar si el dataset tiene estructura agrupable; de lo contrario cualquier cluster obtenido es artificial.
- **Estadístico de Hopkins:** compara distancias de puntos reales vs. aleatorios; $H$ cercano a 1 indica estructura de clusters, cercano a 0.5 indica datos aleatorios (no clusterizar).
- **Silhouette Score:** $s = (b-a)/\max(a,b)$, combina distancia intracluster ($a$) e intercluster ($b$); umbral práctico de preocupación en 0.4; permite detectar individuos mal clasificados.
- **VAT:** heatmap de la matriz de distancias reordenada; bloques oscuros en la diagonal = estructura de clusters.
- **Calibración de modelos = hyperparameter tuning:** ajustar número de clusters, métrica de distancia, criterio de enlace, etc. No existen librerías automatizadas (Grid/Random Search) para clustering como sí existen en aprendizaje supervisado, porque no hay métricas de accuracy objetivas — solo silueta y distancias intra/intercluster.
- La calibración correcta es crítica en aplicaciones de alto impacto: segmentación de clientes, medicina, detección de fraude.

---

## Fuera del PDF — logística, tareas y metodología

- **Repaso de la tarea anterior (caso Hi-Ol Industries, aceite de mostaza):** se repasó en vivo el código de regresión lineal con `statsmodels` sobre los datos sin transformar y con transformación logarítmica (log-log).
  - Sin transformar: $R^2 = 67\%$ (las cuatro variables predictoras explican ese porcentaje de la demanda). El F-statistic, expresado en notación científica, es muy inferior a 0.05 → el modelo es estadísticamente válido.
  - El **p-value** de cada coeficiente debe ser inferior a 0.05 para poder interpretarlo: el ingreso per cápita no cumple ($p=0.06$), por lo que no es relevante para el modelo; el gasto en promoción sí cumple y es positivo.
  - Interpretación sin transformar: subir el precio propio en \$1 reduce las ventas en ~136-137 unidades/mes; si la competencia sube su precio en \$1, la empresa vende ~117 unidades más (producto sustituto — la competencia, al ser marcas más grandes, puede subir precios con más margen).
  - Con transformación log-log ($R^2 \approx 67.5\%$) los coeficientes se leen directamente como **elasticidades**: precio propio ≈ −1 (elasticidad casi unitaria), precio de la competencia ≈ +0.89.
  - Conceptos de elasticidad-precio de la demanda: **elástica** ($>1$, un cambio pequeño en precio genera un cambio grande en cantidad — típico de bienes de lujo o ropa), **inelástica** ($<1$, la demanda casi no reacciona al precio — típico de bienes esenciales como medicinas o comida) y **unitaria** ($\approx 1$, el cambio en precio es proporcional al cambio en cantidad demandada — el caso de esta empresa).
  - La sesión de repaso quedó grabada y subida al Drive del curso.
- **Última tarea del curso — dataset de vino:** trabajo colaborativo en grupos de hasta 3 personas (organizados en salas de trabajo/breakout rooms durante la clase), aplicando K-Means y clustering jerárquico con calibración de hiperparámetros, visualización mediante PCA y gráfico radar, e interpretación de resultados apoyada en IA dado el vocabulario técnico de las variables. El ejercicio en clase no fue calificado; solo buscaba dar espacio de trabajo grupal, algo poco frecuente durante el curso.
- **Criterio de calificación de tareas:** el docente no es estricto; mientras se entregue el ejercicio y esté bien resuelto (código funcional, caso respondido), la nota es buena. Las tareas pendientes de calificar se revisarían esa semana; quien no haya entregado aún tendría una última revisión al final del curso.
- **Entrega por grupo:** solo una persona del grupo sube el archivo a la plataforma (carpeta "tarea caso HOI").
- **Próxima clase (martes siguiente):** simulación de Montecarlo aplicada a un contexto de finanzas. El docente no estaría disponible la semana intermedia por deber supervisar un examen.
