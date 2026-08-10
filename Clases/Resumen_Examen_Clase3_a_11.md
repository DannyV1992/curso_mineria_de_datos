# Resumen de estudio — Examen Minería de Datos (Clase 3 a Clase 11)

**Alcance del examen:** 5 preguntas tipo entrevista sobre todo el curso, cubriendo PCA, ACS/ACM, t-SNE/UMAP, Clustering Jerárquico, K-Means, DBSCAN y Evaluación de clustering.

---

## Parte 1 — Resumen del curso

### Bloque 1 — Reducción de dimensionalidad

| Clase | Algoritmo | Tipo de dato | Idea central |
|---|---|---|---|
| 3 | **PCA** | Numérico | Combinaciones lineales (componentes) que maximizan varianza; eigenvectores ortogonales entre sí. |
| 4 | **ACS / ACM** | Categórico | Mismo objetivo que PCA (maximizar inercia) pero con distancia chi-cuadrado sobre perfiles de frecuencia. ACS = 2 variables; ACM = 3+ variables (vía dummies/codificación disyuntiva). |
| 5–6 | **t-SNE / UMAP** | No lineal | Preservan vecindades (t-SNE: local, distribución t-Student, estocástico; UMAP: local+global, grafo difuso, determinístico, más rápido). No tienen varianza explicada ni biplot. |

**Elementos comunes a PCA/ACS/ACM:**
- **Coseno cuadrado (cos²):** calidad de representación de un individuo/variable en el plano. Umbral práctico: **60% acumulado**.
- **Contribución:** cuánto aporta una variable a un eje; suma 1 por componente; sin umbral fijo de corte.
- **Estandarización obligatoria** en PCA (el propio algoritmo la hace); no es necesaria para variables categóricas.
- PCA es 100% lineal; para no linealidad → t-SNE/UMAP; para variables cualitativas → ACS/ACM.
- No hay regla fija para elegir PCA vs. ACM: se corren ambos y se compara varianza/inercia acumulada.

### Bloque 2 — Clustering

| Clase | Tema | Idea central |
|---|---|---|
| 6–8 | **Fundamentos + Clustering jerárquico** | Centroide, matriz de distancias (euclidiana, Manhattan, Chebyshev...), dendrograma, algoritmo aglomerativo, criterios de enlace (single, complete, average, **Ward**). |
| 9 | **K-Means** | Estocástico, requiere K previo, itera asignación↔recentroide hasta convergencia. Criterio de Fisher (J = L/d). Codo de Jambú para elegir K. |
| 10 | **DBSCAN** | Basado en densidad (ε, MinPts). No requiere K. Clasifica puntos en centrales/frontera/ruido. Detecta outliers de forma nativa. |
| 11 | **Evaluación de clustering** | Hopkins (clusterabilidad), VAT, Silhouette Score, calibración de hiperparámetros (tuning manual, sin Grid Search por falta de métrica objetiva). |

**Ideas transversales al bloque de clustering:**
- Antes de clusterizar: verificar **clusterabilidad** (Hopkins > 0.7 sí, ≈0.5 no) y VAT (bloques oscuros en diagonal).
- Problemas previos: **escalas** distintas, **outliers**, **alta dimensionalidad** (maldición de la dimensionalidad).
- Objetivo doble de todo algoritmo: **minimizar distancia intraclase** + **maximizar distancia interclase** (criterio de Fisher J = L/d).
- **Clusterizar siempre sobre datos originales**, nunca sobre componentes de PCA (pérdida de interpretabilidad); PCA/t-SNE/UMAP se usan solo para *visualizar* clusters ya formados.
- **Hipersegmentación:** crear más clusters de los que existen realmente → costo de negocio.
- Jerárquico: determinístico, mejor para datasets **pequeños**. K-Means: estocástico, mejor para **grandes**, requiere K. DBSCAN: estocástico (punto inicial aleatorio), no requiere K, detecta ruido — mejor para anomalías, no para asignar el 100% de los puntos a un grupo.
- **Silhouette Score** ($s=(b-a)/\max(a,b)$): umbral de preocupación en 0.4; sirve para evaluar clusters y para calibrar hiperparámetros.

---

## Parte 2 — Preguntas y respuestas

### Conceptuales generales

**1. ¿Qué diferencia el aprendizaje supervisado del no supervisado? Da un ejemplo de negocio.**
El aprendizaje supervisado parte de datos etiquetados y busca predecir una variable objetivo (respuesta conocida). El no supervisado no tiene variable objetivo: busca descubrir estructura, patrones o agrupaciones naturales en los datos, sin intentar pronosticar nada puntual. Ejemplo: bancos, hoteles o Netflix agrupan (clusterizan) a sus clientes según comportamiento de consumo para luego recomendar productos o contenido a un cliente basándose en lo que consumieron otros clientes de su mismo grupo, sin necesidad de predecir un valor específico sobre ese individuo.

**2. ¿Por qué es necesario estandarizar los datos antes de aplicar PCA o clustering? ¿Qué pasa si no se hace?**
Porque estos algoritmos son sensibles a la escala: si una variable tiene una magnitud mucho mayor que otra (por ejemplo ingreso en dólares vs. edad en años), esa variable domina por completo el cálculo de varianza o de distancia, distorsionando el resultado. Al estandarizar ($X' = (X-\mu)/\sigma$), todas las variables quedan en la misma escala (media 0, desviación 1) y contribuyen de forma equilibrada. En PCA el propio algoritmo aplica la estandarización internamente; en clustering hay que hacerlo explícitamente antes de calcular las distancias.

**3. ¿Qué es la maldición de la dimensionalidad y cómo afecta al clustering?**
Es el fenómeno por el cual, a medida que aumenta el número de variables, las distancias entre pares de puntos tienden a concentrarse en un rango estrecho y pierden capacidad discriminativa: en baja dimensión las distancias separan bien cercanos de lejanos, pero en alta dimensión casi todos los puntos terminan "igual de lejos" entre sí. Esto afecta directamente a algoritmos basados en distancia (K-Means, jerárquico) y en densidad (DBSCAN, donde la noción de densidad pierde sentido). Por eso se recurre a PCA, ACM, t-SNE o UMAP como paso previo de reducción de dimensionalidad.

---

### PCA

**4. ¿Qué mide el coseno cuadrado y cuál es su umbral práctico de interpretación?**
Mide la calidad de representación de un individuo o variable en un componente o plano: qué tan bien queda reflejada esa observación al proyectarla en menos dimensiones. Cerca de 1 significa buena representación; cerca de 0, mala representación (no pérdida de "calidad" del individuo, sino falta de información para afirmar algo sobre él). El umbral práctico es **60% acumulado** (sumando las dimensiones del plano usado): por encima, se puede interpretar con confianza; por debajo, no conviene afirmar nada sobre ese caso.

**5. ¿Qué diferencia hay entre coseno cuadrado y contribución?**
El coseno cuadrado mide qué tan bien representado está un individuo o variable en un eje (tiene un umbral de corte práctico: 60%). La contribución mide cuánto aporta una variable a la construcción de ese eje —las contribuciones de todas las variables a un mismo componente siempre suman 1—, pero no tiene un umbral objetivo de corte: solo permite comparar una variable contra otra ("esta contribuye más que aquella"), quedando la decisión de exclusión al juicio del analista.

**6. ¿Por qué el primer componente principal maximiza la varianza y el segundo es ortogonal al primero?**
El primer eigenvector se traza en la dirección donde los datos maximizan la varianza, porque ahí está la mayor cantidad de información posible. El segundo eigenvector se traza de forma ortogonal (90°) al primero para eliminar al 100% la correlación con él y así capturar información que el primero no explicaba — si ambos apuntaran en direcciones similares, estarían repitiendo la misma información, que es justo lo que el PCA busca evitar.

**7. ¿Qué significa que dos variables estén en direcciones opuestas en el círculo de correlación?**
Que tienen una correlación fuerte negativa (cercana al 180°, 100% negativa): un individuo que puntúa alto en una de esas variables tiende a puntuar bajo en la otra, y viceversa. Ejemplo del dataset decathlon2: lanzamiento de disco vs. carrera de 100 m — quien es bueno lanzando disco tiende a ser malo en la carrera de 100 m.

---

### ACS/ACM

**8. ¿Cuál es la diferencia entre ACS y ACM? ¿Cuándo se usa cada uno?**
El ACS (Análisis de Correspondencias Simple) funciona sobre una tabla de contingencia de exactamente **dos** variables cualitativas. El ACM (Análisis de Correspondencias Múltiple) generaliza el mismo mecanismo a **tres o más** variables cualitativas, transformándolas primero en una tabla disyuntiva completa (dummies/one-hot encoding) y aplicando después la misma lógica del ACS sobre esa tabla. Se usa ACS cuando solo hay dos variables categóricas a relacionar; ACM cuando hay tres o más.

**9. ¿Qué distancia usa el análisis de correspondencias en vez de la covarianza del PCA, y por qué?**
Usa la **distancia chi-cuadrado (χ²)** sobre perfiles de frecuencia (de fila y de columna), en vez de la matriz de covarianza que usa el PCA. Esto es necesario porque las variables son cualitativas (categorías, no valores numéricos continuos), por lo que no tiene sentido calcular varianza o covarianza sobre ellas; en su lugar se comparan las frecuencias observadas contra las esperadas bajo independencia, y esa desviación se mide con chi-cuadrado.

---

### t-SNE/UMAP

**10. ¿Por qué t-SNE usa una distribución t-Student en el espacio reducido en lugar de una gaussiana?**
Porque en espacios de muchas dimensiones muchos puntos están "moderadamente lejos" entre sí, y al comprimir todo en 2D con una distribución normal esos puntos reciben probabilidades extremadamente pequeñas, provocando que el algoritmo amontone todo hacia el centro (el mismo problema que sufre el PCA). La t-Student tiene colas más pesadas y alargadas, por lo que los puntos moderadamente lejanos conservan una probabilidad relativamente importante, lo que genera fuerzas repulsivas más fuertes entre ellos y permite separar mejor los clusters en el mapa 2D.

**11. ¿Cuál es la principal limitación de t-SNE y UMAP frente al PCA?**
No calculan una métrica de varianza explicada (ni inercia) equivalente a la del PCA, por lo que no hay forma de cuantificar cuánta información se conservó al reducir la dimensionalidad. Además, solo trabajan con individuos, no con variables, así que no ofrecen un biplot: permiten ver que existe un cluster, pero no explican qué variables lo caracterizan (para eso se necesita el paso posterior de clustering formal).

**12. Diferencias clave entre t-SNE y UMAP.**

| Característica | t-SNE | UMAP |
|---|---|---|
| Naturaleza | Estocástico | Determinístico |
| Proyectar nuevos individuos | No | Sí (uso semi-supervisado) |
| Estructura preservada | Principalmente local | Local y global |
| Velocidad | Más lento | Más rápido y escalable |

---

### Clustering — fundamentos y jerárquico

**13. ¿Qué es un centroide y cómo se ve afectado por un outlier?**
El centroide es el representante promedio de un cluster: el promedio de todas las observaciones que pertenecen a ese grupo, una por variable. Como el cálculo es un promedio simple, un valor extremadamente alejado (outlier) desplaza el centroide hacia una región poco representativa de la mayoría de los puntos del cluster — por ejemplo, incluir al CEO en el cluster de salarios del resto de empleados hace que el "salario promedio" deje de tener sentido para describir al grupo.

**14. Explica los criterios de enlace (single, complete, average, Ward) y cuál es el más usado.**
- **Salto mínimo (single linkage):** distancia entre los dos puntos más cercanos de cada cluster; tiende a formar clusters alargados o en cadena.
- **Salto máximo (complete linkage):** distancia entre los dos puntos más lejanos de cada cluster; forma clusters compactos y de tamaño similar.
- **Promedio (average linkage):** promedio de todas las distancias cruzadas entre ambos clusters; es un compromiso entre los dos anteriores.
- **Ward:** fusiona los clusters que produzcan el menor incremento en la varianza intra-cluster; tiende a formar clusters esféricos de tamaño similar y es el **criterio más usado en la práctica**.

No existe un criterio universalmente mejor: la elección depende de la estructura de los datos, y se evalúa comparando resultados (silueta, etc.).

**15. ¿Por qué se recomienda clusterizar sobre los datos originales y no sobre las componentes de PCA?**
Porque el PCA es una combinación lineal de las variables originales, así que los centroides calculados sobre componentes quedan en coordenadas artificiales, sin interpretación directa en términos de negocio (no se puede explicar con precisión "de dónde sale" cada centroide frente a un gerente). El flujo correcto es clusterizar sobre los datos originales y usar el PCA (o t-SNE/UMAP) únicamente como herramienta de visualización de los clusters ya formados.

**16. ¿Cómo se decide en cuántos clusters cortar un dendrograma?**
El algoritmo, por diseño, siempre termina fusionando todo en un único gran cluster; la cantidad final de clusters se decide después, trazando una línea horizontal sobre el dendrograma. El número de clusters resultante es igual al número de intersecciones que corta esa línea: más arriba = menos clusters (más generales); más abajo = más clusters (más específicos). Para objetivar esa decisión (en vez de dejarla a la interpretación visual) se usan el **codo de Jambú** y el **indicador de silueta**.

---

### K-Means

**17. Explica el algoritmo de K-Means paso a paso.**
1. Se inicializan K centroides de forma aleatoria entre los puntos de los datos.
2. **Asignación:** cada punto se asigna al centroide más cercano (normalmente por distancia euclidiana).
3. **Recálculo:** cada centroide se recalcula como el promedio de los puntos asignados a su cluster.
4. **Convergencia:** se repiten los pasos 2 y 3 hasta que los centroides dejan de cambiar (o cambian menos que un umbral ε); en ese momento el algoritmo se detiene y devuelve los clusters finales.

**18. ¿Qué es el codo de Jambú y para qué se usa?**
Es una técnica para decidir cuántos clusters K son adecuados, graficando la inercia $W(K) = \sum_{i=1}^{K}\sum_{x\in C_i}\lVert x-\mu_i\rVert^2$ (qué tan compactos son los clusters) contra distintos valores de K. El K óptimo es el punto donde la curva forma un "codo": a partir de ahí, seguir aumentando K ya no reduce significativamente la inercia. Se usa sobre todo con K-Means porque ahí K es un parámetro obligatorio (en clustering jerárquico no lo es, ya que el dendrograma se construye completo y se corta después).

**19. ¿Qué es el criterio de Fisher (J = L/d) y cómo se interpreta?**
Combina en un solo índice los dos objetivos de todo clustering: $L$ es la distancia intercluster (separación entre centroides) y $d$ es la distancia intracluster (dispersión promedio dentro de cada cluster); se busca **maximizar** $J = L/d$. Un valor aislado de J no tiene interpretación por sí solo (no hay un umbral universal como un p-valor); solo sirve para **comparar configuraciones entre sí** (distintas métricas, enlaces o algoritmos) y quedarse con la que produzca el J más alto.

**20. Ventajas y desventajas de K-Means frente a clustering jerárquico.**

| | Clustering jerárquico | K-Means |
|---|---|---|
| Naturaleza | Determinístico | Estocástico |
| Tamaño de dataset recomendado | Pequeño (recursivo, más lento) | Grande (más rápido) |
| Requiere K previo | No (se corta el dendrograma después) | Sí (obligatorio) |
| Calidad de resultados | Suele ser mejor | Depende de la inicialización |

K-Means es sensible a outliers, a la inicialización aleatoria de centroides, a la escala de los datos, y no funciona bien con formas no esféricas; a cambio es simple, rápido y escalable.

---

### DBSCAN

**21. Explica qué son los puntos centrales, de frontera y de ruido en DBSCAN.**
- **Punto central (core point):** tiene al menos MinPts vecinos (incluyéndose a sí mismo) dentro de su radio ε.
- **Punto de frontera (border point):** no cumple MinPts por sí mismo, pero cae dentro del ε-vecindario de algún punto central — sí se agrega al cluster de ese punto central, aunque desde él no se sigue expandiendo la búsqueda.
- **Punto de ruido / outlier:** no es central y no pertenece al vecindario de ningún punto central; queda fuera de cualquier cluster.

**22. ¿Qué son ε y MinPts, y cómo se eligen?**
**ε (epsilon)** define el radio de la vecindad alrededor de un punto; **MinPts** establece el número mínimo de vecinos dentro de ese radio para considerar la región densa. ε se elige con el gráfico k-distance (equivalente al codo de Jambú pero "invertido": se grafica la distancia al k-ésimo vecino más cercano ordenada de menor a mayor, y se elige el punto donde la curva forma un codo). MinPts se elige con la regla práctica $MinPts \geq D+1$ (D = número de dimensiones), recomendándose $MinPts \approx 2 \times D$.

**23. ¿Cuándo conviene usar DBSCAN en lugar de K-Means?**
DBSCAN conviene cuando el objetivo es **detectar anomalías** (fraude, defectos de manufactura): identifica automáticamente los puntos de ruido sin necesidad de especificarlos previamente, y ese ruido es precisamente el hallazgo valioso. K-Means conviene cuando el objetivo es **asignar una estrategia (marketing, precio, retención) a todos los clientes/puntos**, porque siempre asigna cada observación a algún cluster, incluidos los atípicos — DBSCAN, en cambio, deja esos puntos fuera de cualquier estrategia.

---

### Evaluación de clustering

**24. ¿Qué es el Índice de Hopkins y qué significa un valor cercano a 0.5 vs. cercano a 1?**
Es un estadístico que mide la tendencia natural de los datos a formar grupos, comparando la distribución real contra una distribución generada de forma completamente aleatoria en el mismo espacio: $H = \frac{\sum y_i}{\sum x_i + \sum y_i}$, donde $x_i$ son las distancias de los puntos reales y $y_i$ las de los puntos aleatorios. Un valor **cercano a 1** indica fuerte tendencia a formar clusters (los datos reales están mucho más cerca entre sí que los puntos aleatorios); un valor **cercano a 0.5** indica que los datos se comportan como aleatorios y no hay evidencia de estructura — no vale la pena clusterizar.

**25. ¿Qué es el Silhouette Score y qué mide (a y b)?**
Es un índice que mide, para cada punto, qué tan bien asignado está a su cluster: $s = \frac{b-a}{\max(a,b)}$, donde **a** es la distancia promedio de ese punto a los demás puntos de su propio cluster (cohesión) y **b** es la distancia promedio al cluster vecino más cercano (separación). El rango va de −1 a 1; si $a > b$ (el punto está más cerca de otro cluster que del suyo) el índice se vuelve negativo, señal de mala asignación. El umbral práctico de preocupación es **0.4**.

**26. ¿Por qué no existen librerías de tuning automatizado (Grid Search) para clustering como sí en aprendizaje supervisado?**
Porque el hyperparameter tuning automatizado necesita una métrica de evaluación objetiva para comparar configuraciones (en supervisado: accuracy, F1, AUC, comparadas contra una etiqueta real conocida). En clustering no existe una "clase real" contra la cual comparar el resultado: la calidad de un agrupamiento es subjetiva, y las únicas herramientas disponibles son el índice de silueta y las distancias intra/intercluster (criterio de Fisher). Por eso la calibración en clustering se hace con bucles manuales (fuerza bruta) probando combinaciones de hiperparámetros y quedándose con la que maximice la silueta.

---

### Comparación entre algoritmos

**27. Dado un caso de negocio (ej. segmentación de clientes, detección de fraude), ¿qué algoritmo usarías y por qué?**
Depende del objetivo de negocio:
- **Segmentación de clientes para marketing/precios:** K-Means, porque asigna cada cliente a algún segmento (incluso los atípicos), lo que permite aplicar una estrategia a toda la base.
- **Detección de fraude o anomalías:** DBSCAN, porque su fortaleza central es identificar de forma nativa los puntos de ruido/outlier, que son precisamente el hallazgo de interés.
- **Datasets pequeños donde interese la mejor calidad de agrupamiento y trazabilidad completa (dendrograma):** Clustering jerárquico, aceptando el costo computacional mayor.
En todos los casos, antes de decidir conviene validar la clusterabilidad (Hopkins, VAT) y calibrar los hiperparámetros comparando silueta.

**28. Compara PCA, t-SNE y UMAP para separar visualmente clusters.**
El PCA es lineal y tiende a amontonar los individuos hacia el centro del plano cuando las relaciones son no lineales, mezclando clusters que en realidad están separados; a cambio, sí ofrece varianza explicada y biplot (variables + individuos). t-SNE separa los clusters de forma mucho más compacta y nítida gracias a su distribución t-Student de colas pesadas, pero es estocástico y no proyecta variables. UMAP suele dar la separación más clara de los tres, es determinístico, más rápido, y preserva estructura local y global — pero, igual que t-SNE, no calcula varianza explicada ni ofrece biplot.

**29. Compara K-Means, clustering jerárquico y DBSCAN.**

| | Jerárquico | K-Means | DBSCAN |
|---|---|---|---|
| Naturaleza | Determinístico | Estocástico | Estocástico (punto inicial aleatorio) |
| Requiere K previo | No | Sí | No |
| Tamaño de dataset ideal | Pequeño | Grande | Grande (pero degrada en alta dimensión) |
| Maneja outliers | No de forma nativa | No (los asigna a algún cluster igual) | Sí, de forma nativa (los marca como ruido) |
| Forma de los clusters | Depende del linkage | Esféricos, tamaño similar | Cualquier forma geométrica |
| Mejor uso | Datasets pequeños, trazabilidad completa | Asignar estrategia a todos los puntos | Detección de anomalías |

---

### Matices y correcciones frecuentes en los quizzes (alta probabilidad de examen)

Estas preguntas repiten errores conceptuales que el docente corrigió explícitamente durante la retroalimentación de los quizzes — el examen final se describe como "muy similar" a esas preguntas.

**30. ¿Los componentes principales son variables originales reescaladas o medidas de correlación entre variables?**
No. Los componentes principales son **combinaciones lineales** de las variables originales, nunca variables originales reescaladas ni medidas de correlación. De hecho, el objetivo del PCA es justamente **eliminar** la correlación entre sus componentes: la ortogonalidad entre eigenvectores equivale a independencia, es decir, correlación 0.

**31. ¿El primer componente principal es el que menor varianza explica?**
No, es al revés: el primer componente principal es, por definición, el que **maximiza** la varianza explicada — nunca el de menor varianza. Los componentes se ordenan de mayor a menor varianza explicada (mayor a menor eigenvalor).

**32. ¿Es cierto que la estandarización nunca es necesaria en PCA por defecto?**
No. La estandarización es indispensable cuando las variables tienen escalas distintas (dólares, libras, colones), y el propio modelo de PCA la aplica automáticamente. Para variables **categóricas** no es necesaria, porque su escala (0/1) no genera el mismo tipo de distorsión que una escala numérica amplia. La estandarización también es relevante en algoritmos supervisados basados en distancias (KNN) o sensibles a la escala (redes neuronales); los basados en árboles (Random Forest, Gradient Boosting) no la necesitan.

**33. ¿El ACM es una extensión del PCA?**
No. El ACM es la extensión correcta del **ACS** (para manejar múltiples variables categóricas en vez de solo dos), no una extensión del PCA. ACS y ACM son modelos lineales igual que el PCA y comparten su objetivo de fondo (reducir dimensionalidad maximizando información conservada), pero no son una extensión matemática del PCA: cambian covarianza por perfiles de frecuencia y distancia chi-cuadrado.

**34. En un mapa factorial de ACM, ¿pueden dos categorías cercanas pertenecer a la misma dimensión?**
No necesariamente por cercanía: cada categoría se lee en su propia dimensión. Dos categorías cercanas en el mapa están asociadas frecuentemente (aparecen juntas en los mismos individuos); dos categorías alejadas presentan poca o nula asociación. La proximidad en el mapa es un indicador de asociación, no de pertenencia a la misma dimensión original.

**35. ¿Existe un umbral fijo de probabilidad para decidir si dos puntos pertenecen al mismo cluster en UMAP?**
No. No hay un umbral fijo de probabilidad de conexión para decidir pertenencia a un mismo cluster en UMAP: conviene comparar la probabilidad de conexión más alta de un punto contra las demás probabilidades de conexión de otros grupos antes de decidir, en lugar de fijar un corte universal.

**36. ¿Una "conexión fuerte" entre dos puntos en UMAP implica que pertenecen a la misma clase?**
No. Una conexión fuerte en UMAP indica que son vecinos cercanos en el espacio original, pero eso no implica necesariamente que pertenezcan a la misma **clase** (concepto de aprendizaje supervisado). Sí puede indicar que pertenecen al mismo **cluster**, y UMAP puede usarse en un esquema semi-supervisado para proyectar nuevos individuos.

**37. Calcula la distancia euclidiana entre dos puntos y explica su interpretación aislada.**
Con $A=(1,2)$ y $B=(2,3)$: $d(A,B) = \sqrt{(1-2)^2+(2-3)^2} = \sqrt{2} \approx 1.41$. Un valor de distancia aislado entre solo dos puntos **no tiene interpretación por sí solo**: la interpretación surge únicamente al comparar esa distancia contra las distancias entre todos los demás pares de puntos, organizadas en una matriz de distancias completa (para saber si esos dos puntos están relativamente cerca o lejos respecto al resto).

**38. ¿Por qué no se puede clusterizar directamente calculando una sola distancia entre dos individuos?**
Porque esa distancia no tiene sentido por sí sola: solo cobra significado cuando se compara contra la matriz de distancias completa entre todos los pares de observaciones. Es el mismo principio que un RMSE aislado en series de tiempo (por ejemplo 0.65): no dice nada hasta compararlo con el RMSE de otro modelo. La contribución en PCA/MCA sigue la misma lógica comparativa, no absoluta.

**39. ¿Qué es la hipersegmentación y qué la distingue de una segmentación correcta?**
Es el error de crear más clusters de los que realmente existen en los datos: el algoritmo puede técnicamente formar, por ejemplo, 10 grupos cuando en realidad solo existen 4 comportamientos reales distintos. Se detecta comparando contra el codo de Jambú (más allá del codo, seguir aumentando K no reduce significativamente la inercia) y el índice de silueta (fusionar clusters muy pegados suele subir la silueta general). Tiene un costo de negocio directo: cada cluster de más implica una campaña, política de precio o estrategia de retención distinta dirigida al mismo tipo de cliente — literalmente perder dinero.
