# Clase 10 — DBSCAN

**Curso:** Minería de Datos · Lead University
**Docente:** Rachit Vargas (MBA | Engineer)
**PDF fuente:** `Clases/PDFs/Clase 10 - DSBCAN.pdf`

> La clase combinó la explicación teórica del algoritmo con una demostración de código en Python (dataset de vinos) y la presentación de un caso de negocio real (empresa de aceite de mostaza en la India) que queda como tarea final del curso.

---

## 1. Introducción a DBSCAN

- **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*) es un algoritmo de clusterización, al igual que K-Means y el clustering jerárquico, pero llega a la clusterización de una forma completamente distinta: se basa en el concepto de **densidad**.
- Densidad ≈ qué tan poblada está una región del espacio. El algoritmo va "conquistando" regiones densas del dataset y las agrupa como clústeres; los puntos alejados de cualquier región poblada se clasifican como **ruido** u **outliers** — para DBSCAN, ambos términos son sinónimos.
- Es ampliamente utilizado en **detección de anomalías**, porque identifica de forma natural qué observaciones no pertenecen a ninguna población. Detectar valores atípicos en un espacio multivariado con DBSCAN es mucho más sencillo que con técnicas univariadas (boxplot, gráfico de distribución) cuando hay muchas dimensiones involucradas.
- A diferencia de K-Means, DBSCAN **no requiere conocer previamente el número de clústeres** — en esto se comporta igual que el clustering jerárquico.
- Dos hiperparámetros esenciales: **ε (epsilon)**, que define el radio de la vecindad alrededor de un punto, y **MinPts**, que establece el número mínimo de vecinos necesarios dentro de ese radio para considerar que un punto pertenece a una región de alta densidad.
- **Intuición:** los datos tienden a organizarse de forma natural en regiones con alta densidad de observaciones. Los puntos próximos entre sí y con suficientes vecinos pertenecen al mismo grupo; las zonas de baja densidad actúan como separación natural entre clústeres distintos. Esto permite identificar estructuras de **cualquier forma geométrica** sin necesidad de fijar K de antemano. Los puntos que no pertenecen a ninguna región suficientemente densa se excluyen del proceso de agrupamiento — no se les fuerza a formar parte de ningún clúster, simplemente se marcan como ruido.

## 2. DBSCAN vs. K-Means

- Visualmente, ambos algoritmos llegan a una clusterización, pero de forma distinta.
- Ejemplo trabajado en clase: un bloque de puntos completamente denso y homogéneo (un "manchón" sin caídas de densidad visibles) es clasificado por DBSCAN como un **único clúster**, porque no existe ninguna separación de densidad que lo divida. K-Means, en cambio, sí lo particiona en varios grupos, porque su lógica es de distancia a centroides, no de densidad.
- En las figuras clásicas de comparación (círculos concéntricos, formas de luna, blobs con outliers, cuadrado uniforme): DBSCAN respeta la forma geométrica real de las agrupaciones y aísla el ruido; K-Means fuerza fronteras convexas y asigna **todo punto** — incluidos los outliers — a algún clúster.

## 3. Tipos de puntos en DBSCAN

Para cada punto $p$ se define su **ε-vecindario** $N_\varepsilon(p) = \{x \in D : d(x,p) \le \varepsilon\}$: todos los puntos a distancia menor o igual que ε, contando al propio $p$. Según cuántos puntos contenga ese vecindario y cómo se relacione con otros vecindarios, cada punto se clasifica en una de tres categorías:

| Tipo | Definición | Ejemplo con MinPts = 5 |
|---|---|---|
| **Punto central (core point)** | $\lvert N_\varepsilon(p)\rvert \ge MinPts$ | Al contar los vecinos dentro del radio ε se encuentran 7 → cumple, es central |
| **Punto de frontera (border point)** | $p$ no es central, pero existe $q$ central tal que $p \in N_\varepsilon(q)$ | El punto tiene solo 3-4 vecinos propios (no cumple MinPts), pero cae dentro del vecindario de otro punto que sí es central |
| **Punto de ruido / outlier** | $p$ no es central y no existe ningún $q$ central tal que $p \in N_\varepsilon(q)$ | No cumple MinPts y tampoco pertenece a la vecindad de ningún punto central |

Un punto de frontera **no es un outlier**: aunque no reúne por sí mismo la cantidad mínima de vecinos, ya pertenece a la vecindad de algún punto central, por lo que sí se agrega al clúster de ese punto central.

**Selección del punto inicial:** es **100% aleatoria** — por esto DBSCAN es un algoritmo estocástico. Cualquier punto del dataset pudo haber sido el elegido para iniciar la exploración de una región.

## 4. Algoritmo paso a paso

1. **Seleccionar un punto inicial $p$**, no visitado, de forma aleatoria.
2. **Construir su ε-vecindario**: todos los puntos a distancia ≤ ε de $p$.
3. **Clasificar $p$** contando cuántos puntos hay en su vecindario (incluyéndose a sí mismo): si el conteo ≥ MinPts, es punto central.
4. **Si $p$ es central**, se crea un nuevo clúster (p. ej. $C_1$) y $p$ se marca como parte de él.
5. **Explorar los puntos del ε-vecindario de $p$**: para cada punto $q$ del vecindario aún no visitado, si $q$ también es central, todos los puntos de su propio vecindario se agregan a la lista de puntos por explorar y $q$ se asigna al mismo clúster $C_1$.
6. **Repetir la exploración** con cada nuevo punto agregado a la lista (tomar el siguiente, verificar si es central, expandir su vecindario) — así el clúster va "conquistando" regiones adyacentes mientras siga encontrando puntos centrales conectados entre sí.
7. **Clasificar los puntos no centrales del vecindario explorado** como puntos de frontera y agregarlos al clúster actual — pertenecen a él, pero desde ellos ya no se sigue expandiendo la búsqueda.
8. **Cuando no quedan puntos por explorar**, el clúster actual queda finalizado. Se selecciona el siguiente punto no visitado del dataset y se repite el proceso desde el paso 2, para ver si inicia un nuevo clúster en una región completamente distinta.
9. **Al terminar de procesar todos los puntos**, cualquier observación que no haya sido asignada a ningún clúster se etiqueta como ruido. El algoritmo termina cuando todos los puntos quedan clasificados (central, frontera o ruido).

**Naturaleza secuencial:** el proceso se ejecuta punto por punto, nunca en paralelo — como cualquier algoritmo de clustering. Para decidir si una región terminó de expandirse, el algoritmo necesita el resultado del punto anterior antes de continuar, por lo que paralelizarlo no tendría sentido.

**Puntos de borde ambiguos (consecuencia de este proceso):** un punto ubicado en el límite entre dos vecindarios puede terminar asignado a uno u otro clúster dependiendo del orden en que el algoritmo visitó los puntos — no hay garantía de un resultado idéntico e independiente del orden de procesamiento.

## 5. Densidad

- Una región se considera **densa** cuando un punto posee un número suficiente de vecinos (≥ MinPts) ubicados a una distancia ≤ ε.
- Regiones con pocos puntos cercanos presentan baja densidad y es menos probable que formen parte de un clúster.
- DBSCAN no identifica grupos por su forma geométrica, sino por la **distribución y concentración local** de los datos — la relación entre distancia y número de vecinos es el fundamento del algoritmo.
- **Conectividad por densidad:** si dos puntos están dentro del radio ε uno del otro y ambos son centrales, sus vecindades pertenecen a la misma región densa y, por tanto, al mismo clúster. Un clúster se expande mediante la unión de regiones densas adyacentes — así es como va "conquistando" cada vez más territorio, hasta que ya no encuentra más puntos centrales conectados.

## 6. Los dos hiperparámetros: efecto sobre el resultado

| Hiperparámetro | Qué controla | Extremo bajo | Extremo alto |
|---|---|---|---|
| **ε (epsilon)** | Tamaño de la vecindad | Vecindad muy pequeña → mucho ruido, clústeres muy pequeños o ninguno | Vecindad muy grande → los clústeres se fusionan entre sí |
| **MinPts** | Densidad mínima requerida | Fácil ser punto central → pocos outliers, sobre-agrupación | Muy estricto → muchos puntos pasan a ser ruido, clústeres más pequeños |

El valor adecuado está en un punto intermedio: ni tantos clústeres diminutos (ε muy chico) ni un solo clúster gigante (ε muy grande) — se busca llegar a los **clústeres naturales** del dataset.

### 6.1 Selección de ε: gráfico k-distance

Es el equivalente al codo de Jambú, pero "al revés" (la curva se lee de abajo hacia arriba en vez de arriba hacia abajo):

1. Calcular la distancia al k-ésimo vecino más cercano de cada punto (con $k = MinPts$).
2. Ordenar esas distancias de menor a mayor y graficarlas.
3. Elegir ε en el punto donde se forma el "codo" de la curva — donde la distancia empieza a crecer de forma abrupta.

En scikit-learn, este cálculo se apoya en el algoritmo supervisado **KNN** (k-nearest neighbors): se entrena para obtener las distancias entre vecinos y con eso se construye el gráfico. La métrica de **silueta** sirve como apoyo adicional para evaluar el ε elegido.

### 6.2 Selección de MinPts: regla práctica

$$MinPts \ge D + 1$$

Donde $D$ es la cantidad de dimensiones (variables) del dataset. La recomendación más robusta es:

$$MinPts \approx 2 \times D$$

| Dimensiones (D) | MinPts mínimo (D+1) | MinPts recomendado (2×D) |
|---:|---:|---:|
| 2 | 3 | 4 |
| 3 | 4 | 6 |
| 5 | 6 | 10 |
| 10 | 11 | 20 |
| 20 | 21 | 40 |

No hay ciencia adicional detrás de esta regla — dimensiones y variables son sinónimos (p. ej. el dataset de 5 asignaturas de estudiantes visto en la clase anterior tiene $D=5$, por lo que el MinPts recomendado sería 10).

## 7. DBSCAN es 100% multivariado

- El algoritmo debe ejecutarse sobre **todos los datos brutos**, sin reducir dimensionalidad primero — igual que en K-Means, donde primero se clusteriza con todas las variables y solo después, si se quiere visualizar, se aplica PCA. Reducir dimensiones antes de clusterizar implica perder información, y esa pérdida puede impedir que el algoritmo llegue a una clusterización óptima.
- La razón de fondo: DBSCAN se basa en distancias entre vecinos, y esa distancia (euclidiana o la que se use) debe calcularse con el **vector completo de variables** de cada observación — todas ellas aportan a por qué un punto está más cerca de otro que de un tercero, aunque en dimensiones altas esa cercanía no pueda visualizarse directamente.
- Reducir dimensiones antes de clusterizar solo se justifica cuando hay redundancia extrema (p. ej. 1000 variables donde 5 componentes explican el 90% de la varianza): ahí las variables sobrantes no aportan información nueva, solo encarecen el cómputo.

## 8. Ventajas y desventajas

| Ventajas | Desventajas |
|---|---|
| Descubre clústeres de forma arbitraria (cualquier forma geométrica, no solo convexa) | Sensible a la elección de ε y MinPts: valores inadecuados generan mucho ruido o fusionan clústeres distintos |
| **Manejo de outliers**: identifica automáticamente los puntos de ruido sin necesidad de especificarlos previamente — ventaja crucial, y el principal motivo de uso del algoritmo en la práctica | Dificultad con densidades muy variables: si la diferencia de densidad entre clústeres es muy grande, puede fallar en detectar los menos densos |
| No requiere el número de clústeres por adelantado (más una diferencia frente a K-Means que una ventaja en sí) | Dependiente de la escala: como todos los algoritmos de clustering, necesita las variables normalizadas o estandarizadas — no es una desventaja exclusiva de DBSCAN |
| Funciona bien con diferentes densidades en la misma ejecución | Problemas en alta dimensionalidad: la noción de densidad pierde significado porque todas las distancias tienden a volverse similares (el mismo problema que enfrenta PCA en alta dimensión, donde los puntos se acumulan hacia el centro; t-SNE mitiga esto en el plano 2D apoyándose en una distribución t de Student, de colas más largas, que sí logra separar visualmente los puntos) |
| Escalable: buena escalabilidad en datasets grandes | Puntos de borde ambiguos: pueden asignarse de forma inconsistente según el orden de procesamiento |

**Manejo de outliers y decisiones de negocio.** Que un algoritmo clasifique un punto como ruido no resuelve automáticamente qué hacer con él en un caso real — ese cliente, punto de venta o producto sigue existiendo y el negocio a menudo necesita ofrecerle algo. La recomendación depende del contexto:

- Si el objetivo es **detectar anomalías** (fraude en transacciones bancarias, defectos en manufactura), el ruido es precisamente lo valioso del análisis — ahí está el hallazgo que justifica usar DBSCAN.
- Si el objetivo es **asignar una política de marketing, precio o retención a todos los clientes**, K-Means puede ser más apropiado: siempre asigna cada punto a algún clúster (aunque sea atípico), lo que permite aplicarle al menos la estrategia del clúster más cercano. DBSCAN, en cambio, deja ese punto fuera de cualquier estrategia.
- Crear una política personalizada solo para un outlier rara vez se justifica, salvo que ese individuo (o un segmento pequeño de outliers similares entre sí) represente un peso económico grande para la compañía — ahí la decisión ya no es técnica sino financiera: se evalúa con métricas como el **ROI**, verificando si el retorno esperado de invertir en captar o retener ese segmento justifica el gasto.

> *"Si vos tenés clústeres en donde casi no existen valores atípicos, mejor para vos, porque quiere decir que la mayoría de las políticas... encierra la mayoría de los clientes que vos tenés."*

## 9. Código en Python (scikit-learn) — sin contraparte directa en el PDF

- `from sklearn.cluster import DBSCAN`. A diferencia de K-Means (que en scikit-learn no expone una métrica de distancia configurable), DBSCAN sí permite elegir la métrica (euclidiana, Manhattan, etc.) como hiperparámetro.
- Hiperparámetros clave en la implementación: `eps` (radio, default 0.5) y `min_samples` (equivalente a MinPts). No existe un parámetro de inicialización "inteligente" como el `k-means++` de K-Means — la exploración de puntos es aleatoria por diseño del algoritmo.
- **Dataset de ejemplo:** vino, con variables como alcohol, magnesio, intensidad de color, etc. Como con cualquier algoritmo de clustering, primero se escalan los datos (`StandardScaler`).
- **Selección de ε:** se usa `NearestNeighbors` (KNN, un algoritmo supervisado que se ve más adelante en el curso, aquí solo como herramienta auxiliar) para obtener las distancias y graficar la curva k-distance. En este ejemplo particular el "codo" no se distingue con claridad — se usa para ilustrar que la elección de ε en la práctica no siempre es obvia.
- En este ejemplo se usó `min_samples = 5` para un dataset de 5 variables — es solo el mínimo ($D+1$), no la recomendación robusta ($2\times D = 10$); se aclara explícitamente que no es el valor óptimo, solo sirve como ilustración del código.
- **Resultado:** el modelo encontró 2 clústeres (etiquetas 0 y 1) y clasificó 55 observaciones como ruido (etiqueta -1).
- **Visualización con PCA:** al proyectar en 2D, el resultado no se ve muy convincente — el PCA de 2 componentes capturaba menos del 50% de la varianza total, por lo que no puede afirmarse con certeza si los puntos marcados como outliers realmente lo son en el espacio completo de variables; con más componentes (más información retenida) podría confirmarse o refutarse esa clasificación.
- **DBSCAN vs. K-Means sobre el mismo dataset de vino:** K-Means encontró 3 clústeres bien formados y sólidos; DBSCAN solo encontró 2 clústeres y clasificó el resto como ruido. Esto refuerza la conclusión de la sección de ventajas: la fortaleza de DBSCAN recae más en la **detección de valores atípicos** que en producir la mejor partición posible del dataset — para clusterización general, K-Means puede rendir mejor.

---

## Conceptos clave de la clase

- **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*): algoritmo de clustering basado en densidad; no requiere definir K previamente y detecta ruido/outliers de forma nativa.
- Tres tipos de puntos: **central** ($\ge$ MinPts vecinos propios), **frontera** (no central, pero dentro del vecindario de un central) y **ruido/outlier** (ni central ni frontera).
- Dos hiperparámetros: **ε** (radio de vecindad) y **MinPts** (mínimo de vecinos para densidad). ε se elige con el gráfico k-distance (codo invertido); MinPts con la regla $MinPts \ge D+1$, recomendado $MinPts \approx 2\times D$.
- El algoritmo es **estocástico** (punto inicial aleatorio) y **secuencial** (nunca paralelo) — a diferencia de K-Means, no tiene un mecanismo de inicialización inteligente.
- Debe alimentarse con **todas las variables brutas**, sin reducción de dimensionalidad previa, salvo redundancia extrema entre variables.
- Ventaja central: manejo automático de outliers, útil en detección de anomalías (fraude, manufactura). Desventaja central: sensibilidad a ε y MinPts, y pérdida de sentido de la densidad en alta dimensionalidad.
- La elección entre DBSCAN y K-Means para un caso de negocio depende de si el objetivo es detectar anomalías (DBSCAN) o asignar una estrategia a todos los clientes/puntos, incluidos los atípicos (K-Means).

---

## Fuera del PDF — logística, tareas y metodología

- **Caso de negocio (tarea):** empresa **Hi-Ol Industries** (India), fundada después del año 2000, dedicada a comercializar aceite de mostaza. Caso real de Harvard/INCAE, trabajado antes por el docente en un curso de organización industrial. Estrategia histórica de la empresa: vender 6-10% por debajo de sus competidores, compensando con buena calidad pese a ser una marca pequeña y poco conocida.
- **Contexto del caso:** en septiembre de 2015, una crisis en la semilla de mostaza disparó los costos de producción. La pregunta de negocio a resolver: ¿debería la empresa subir el precio final del producto, y en qué porcentaje?
- **Datos disponibles (CSV):** fecha, demanda del producto, precio de la empresa, precio de los competidores, ingreso per cápita de los consumidores en India, gasto en promoción/publicidad.
- **Tareas a resolver:** describir la evolución temporal y estacionalidad de la demanda (feature engineering, análisis de correlación); identificar meses atípicos primero sin clusterización y luego aplicando DBSCAN; determinar si septiembre de 2015 es un mes atípico; cuantificar con regresión lineal (log-log) el efecto del precio propio, el precio de la competencia, el ingreso per cápita y la publicidad sobre las ventas, interpretando los coeficientes como elasticidades; simular escenarios de aumento de precio bajo distintos supuestos de reacción de la competencia.
- **Pistas dadas por el docente:** la brecha de precio con la competencia se abrió de ~10% a ~16% en los últimos meses del dataset — indicio de que subir el precio podría ser razonable. Ejemplo de clústeres de referencia (no resueltos por el estudiante, solo mostrados como guía): demanda alta con promoción fuerte (dos variantes) y precio alto con demanda débil — este último sugiere un error histórico de la empresa al subir precios cuando la demanda estaba floja. La regresión log-log mostró: subir el precio propio 1% reduce las ventas ~1%; si la competencia sube su precio 1%, las ventas de la empresa suben ~0.89% (producto sustituto); un mayor ingreso per cápita reduce las ventas (los consumidores migran a marcas más grandes y reconocidas); mayor gasto en publicidad las aumenta.
- **Entrega:** trabajo individual o en grupos de hasta 3 personas, pero solo una persona sube el resultado a la plataforma. Es la última tarea del curso.
- **Uso de IA:** permitido sin problema — lo que se evalúa es la capacidad de interpretar los resultados del análisis, no la redacción del código en sí. Se recomienda, aun así, programar a mano al menos una vez para entender bien qué hace cada línea.
- **Dataset adicional usado en la demo de código:** wine dataset (variables como alcohol, magnesio, color), de uso común para benchmarking de aprendizaje no supervisado.
- **Semana 10 (próxima clase):** evaluación de clústeres (gráficos, métricas, casos de negocio) — no habrá quiz esa semana.
- Se reforzó la disponibilidad de tutoría/repaso los sábados, de aproximadamente 9:00 a 9:30 a.m., duración cercana a 1 hora, para reforzar temas explicados con rapidez en clase.
