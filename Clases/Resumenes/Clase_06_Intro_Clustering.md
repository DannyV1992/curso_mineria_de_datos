# Clase 6 — Cierre de t-SNE/UMAP e introducción al clustering

**Curso:** Minería de Datos · Lead University
**Docente:** Rachit Vargas (MBA | Engineer)
**PDF fuente:** `Clases/PDFs/Clase 6 - Intro clustering.pdf`

> El curso llevaba una sesión de atraso respecto al cronograma original, por lo que esta clase se dedicó casi en su totalidad a **repasar y profundizar t-SNE y UMAP** (tema de la Clase 5) con un ejemplo de código nuevo sobre un dataset de e-commerce, en vez de arrancar con clustering. El docente confirmó al cierre que el tema de **clustering empieza "100 %" la siguiente semana**. Por esa razón, el contenido propio del PDF de esta clase —secciones 1 a 4: definición de clustering, matriz de distancias, tipos de distancia, Hopkins Statistic y VAT— **no se desarrolló de forma oral en esta sesión**; queda documentado aquí a partir del PDF como adelanto de lo que se profundizará con ejemplos prácticos en la próxima clase.

---

## 1. Repaso profundizado — t-SNE y UMAP

### 1.1 El debate sobre cuál algoritmo es "mejor"

A partir de un video del profesor Aldemar Rodríguez (pionero en la enseñanza de t-SNE/UMAP en el país, mencionado también en la Clase 5), surgió la discusión de si UMAP es categóricamente mejor que PCA y que t-SNE. La postura del curso:

- t-SNE y UMAP son algoritmos **recientes** (2008 y 2018 respectivamente) que carecen de las herramientas estadísticas de interpretación que sí tiene el PCA —principalmente la **varianza explicada**—, por lo que evaluarlos "a ciegas" en ese sentido es una limitación real.
- Sin embargo, en la tarea específica de **separar visualmente individuos en clusters bien definidos**, los resultados de t-SNE y UMAP son "extremadamente brutales" en comparación con el PCA: logran compactar cada grupo y alejarlo claramente de los demás, mientras que el PCA tiende a amontonar los individuos en el centro y a mezclar clusters que deberían estar separados.
- **Recomendación práctica para clustering:** cuando se combine reducción de dimensionalidad con clustering (tema de la siguiente semana), usar preferiblemente **t-SNE o UMAP** en lugar de PCA, precisamente porque su resultado en 2D separa mejor los grupos — el PCA puede generar confusión al mezclar individuos de distintos clusters en una misma nube central.

### 1.2 Comparación repetida — Iris y dígitos escritos a mano

**Iris:** con PCA, *setosa* se separa bien pero *versicolor* y *virginica* quedan muy cerca y mezcladas — un problema real para clustering, porque ahí se necesita **maximizar la distancia entre clusters (interclase)** y **minimizar la distancia dentro de cada cluster (intraclase)**, y con esa mezcla no se logra ninguna de las dos. Con t-SNE los tres clusters quedan compactos y muy separados entre sí — calcular después una distancia euclidiana entre ellos es sencillo y contundente. UMAP separa mejor que el PCA, pero no tan nítido como t-SNE.

**Dígitos escritos a mano** (~254 variables, una por píxel): el PCA lanza casi todos los puntos al centro y los confunde entre sí. t-SNE separa los clusters con más claridad. **UMAP da el resultado más impresionante de los tres**, con clusters compactos y bien diferenciados — la tarea que buscan los algoritmos de clusterización.

### 1.3 t-SNE — repaso del mecanismo (con más detalle numérico)

- Es un método de reducción no lineal que reutiliza la lógica del **KNN** (K-Nearest Neighbors, algoritmo supervisado): mide la distancia entre todas las observaciones, calcula la matriz de distancias (típicamente euclidiana, aunque existen alternativas: Manhattan, Chebyshev, coseno, Jaccard) y luego **aleatoriza** los puntos en un nuevo espacio de baja dimensión — de ahí lo "estocástico" del nombre.
- La matriz de distancias es simétrica (espejo): la distancia de 1 a 2 es la misma que de 2 a 1, por lo que en la práctica solo se conserva un triángulo de la matriz (igual que en una matriz de correlación).
- Las distancias se escalan dividiendo cada una entre la suma total, convirtiéndolas en **probabilidades** que siguen una distribución gaussiana en el espacio original.
- En el nuevo espacio aleatorio de baja dimensión, las distancias se convierten también en probabilidades, pero usando una distribución **t-Student** en lugar de la normal — la t-Student tiene colas más largas, lo que evita que todos los puntos colapsen al centro (el mismo problema que arrastra el PCA) y permite separar mejor los clusters.
- El algoritmo itera comparando la matriz de distancias original ($P$) con la matriz del mapa reducido ($Q$) mediante la **divergencia KL**: si es pequeña, los puntos que estaban cerca en el espacio original siguen cerca en 2D — el algoritmo mueve los puntos, acercando los que originalmente eran similares y alejando los que no, hasta converger.
- El docente decidió explícitamente **no profundizar en el detalle matemático de la divergencia KL ni del algoritmo de optimización**: en el trabajo real de un científico de datos, lo relevante es aplicar el modelo ya existente e interpretar sus resultados de negocio, no derivar la matemática detrás — esa profundidad solo importaría si se fuera a crear un método nuevo derivado de este.

### 1.4 UMAP — repaso del mecanismo y comparación con t-SNE

**Diferencias clave confirmadas en el repaso:**

| | t-SNE | UMAP |
|---|---|---|
| Naturaleza | Estocástico | Determinístico |
| Velocidad | Más lento | Más rápido (aunque sea determinístico — un dato "curioso": no hay consenso claro de que los algoritmos estocásticos sean siempre más rápidos que los determinísticos; depende del problema) |
| Proyectar nuevos individuos | No | Sí — puede usarse de forma semi-supervisada |
| Estructura preservada | Principalmente local | Local y global |

> Analogía adelantada con clustering: el **clustering jerárquico** es determinístico y siempre llega a la solución óptima, pero se recomienda solo para datasets pequeños; el **K-means** es estocástico, es el recomendado para datasets grandes, pero no garantiza encontrar siempre el resultado óptimo por su componente aleatorio.

**El problema de asimetría en UMAP y su solución (fuzzy union).** A diferencia de t-SNE, UMAP no calcula una distancia simétrica desde el inicio: usa la fórmula

$$p_{ij} = \exp\left(-\frac{\max(0, d(x_i,x_j) - \rho_i)}{\sigma_i}\right)$$

que puede dar, por ejemplo, que la probabilidad de conexión de $x_1$ hacia $x_2$ sea 0.90 pero la de $x_2$ hacia $x_1$ sea 0.10 — un resultado sin sentido, porque la distancia entre dos puntos tiene que ser la misma en ambas direcciones (el mismo principio de una ruta en Waze: la distancia de mi casa a la de un compañero es igual a la distancia de su casa a la mía). Para corregir esto, UMAP aplica una **unión difusa (fuzzy union)**:

$$p_{ij}^{(sym)} = p_{ij} + p_{ji} - p_{ij}\,p_{ji}$$

Esto se resuelve matricialmente: se toma la matriz original de probabilidades, se calcula su transpuesta y se combinan con esta fórmula para obtener una matriz simétrica, que ahora sí puede usarse para construir un grafo de vecindad consistente. Puntos con probabilidad de conexión igual a 0 forman vecindades (grafos) completamente separadas de otros grupos — esto es, en la práctica, lo que da origen a clusters bien diferenciados en el resultado final.

**Sobre el número de iteraciones.** Al igual que t-SNE, UMAP parte de una posición aleatoria en 2D y ajusta iterativamente acercando vecinos y alejando no-vecinos. La cantidad de iteraciones la define quien corre el modelo — es una decisión de optimización: meter más iteraciones de las necesarias (por ejemplo, 500 cuando bastaban 15) desperdicia poder de cómputo sin mejorar el resultado. El criterio para saber cuándo detenerse es un indicador llamado **silueta (silhouette)** —que se retomará en el tema de clustering—: si al pasar de una iteración a la siguiente la silueta prácticamente no mejora, ya no vale la pena seguir iterando. Es la misma lógica que decidir cuántos árboles poner en un Random Forest: agregar más árboles sin ganancia de precisión es sobrecargar el modelo sin beneficio.

> El docente evitó deliberadamente entrar en el **teorema del nervio** que sustenta la parte geométrica de UMAP (sí cubierto por Aldemar Rodríguez en su video): es un tema de nivel de un curso completo, no aplicable en la práctica profesional ni en entrevistas de trabajo. Anécdota relacionada: en una entrevista de trabajo antigua le preguntaron por las librerías internas de NumPy (C, BLAS, etc.) — conocimiento que rara vez marca una diferencia real en el desempeño de un científico de datos aplicando estos métodos.

### 1.5 Demostración en código — dataset de e-commerce

Dataset de una tienda en línea (`customer ID`, edad, género, categoría de producto, ítem comprado, monto, estacionalidad, método de pago, calificación del producto, descuento aplicado, frecuencia de compras previas) — 5 variables numéricas y 5 categóricas, que tras aplicar `pd.get_dummies` a las categóricas pasan de 10 a **47 columnas**.

- Se excluye `Customer ID` del modelo (es solo un identificador). La estandarización de las variables numéricas, aunque se aplicó en el código, no es estrictamente necesaria porque el propio PCA la realiza internamente.
- **Varianza explicada del PCA: 7 % (dim. 1) + 5.3 % (dim. 2) = 12.3 % acumulado** — un valor bajo, señal de que el dataset guarda relaciones probablemente no lineales, coherente con que después t-SNE y UMAP encuentren una estructura de agrupación más clara que el PCA.
- **Limitación práctica compartida por los tres algoritmos aplicados sobre este dataset:** al ver únicamente los puntos agrupados en el plano —sin un biplot— es muy difícil concluir *qué* caracteriza a cada grupo. El PCA sí ofrece biplot (variables + individuos superpuestos); **t-SNE y UMAP no lo tienen**, por lo que solo puede afirmarse "aquí hay un cluster", pero no "este cluster se caracteriza por...", sin un paso de análisis adicional.
- Al graficar con **Plotly** (librería interactiva, similar en filosofía a Power BI: permite pasar el cursor sobre los puntos y ver el detalle de cada observación) se hizo evidente que t-SNE, en este dataset, termina separando principalmente por **tipo de producto** (accesorios: lentes de sol, bolsos, billeteras; ropa: shorts, camisetas, chaquetas; deportes: yoga mat, balón, mancuernas) — una separación que, en este caso puntual, se podría haber obtenido igual con una tabla pivote en Excel, sin necesidad de un algoritmo tan sofisticado. La observación del docente: ni t-SNE ni UMAP están "inventando" algo que no pudiera obtenerse por otros medios más simples en este dataset particular.

**Casos de uso de negocio (leídos del material, sin desarrollo oral extenso):**
- **Electrónica** es la única categoría que se separa de forma clara y consistente en los tres mapas (PCA, t-SNE, UMAP), por tener un ticket promedio mucho más alto (~$1700, frente a diferencias de 40–160 dólares entre el resto de categorías) — es un segmento premium.
- Implicaciones sugeridas: programas VIP de retención para compradores de electrónica (garantías extendidas, acceso anticipado, financiamiento preferencial), cross-selling de accesorios compatibles en el checkout, tratar electrónica como categoría estratégica con reglas de inventario propias, usar el PCA como filtro rápido de detección de anomalías (transacciones con montos o combinaciones de producto fuera de lo habitual).
- Varias categorías restantes comparten perfiles muy similares (edades, descuentos, temporada, monto medio) — oportunidad de **unificar estrategia comercial**: en lugar de 9 segmentos por categoría, usar 2 o 3 segmentos (premium, estándar, básico) para reducir el costo de personalización. Esto conecta con las dos formas básicas en que compite una empresa: diferenciación (personalización) o liderazgo en costos (estandarización).

> Estos hallazgos de negocio, obtenidos solo con reducción de dimensionalidad, quedan **incompletos** sin el paso siguiente: el clustering formal permitirá no solo visualizar los grupos, sino caracterizarlos con precisión — que es justamente lo que el biplot le da gratis al PCA pero t-SNE/UMAP no ofrecen por sí solos.

---

## 2. Introducción al clustering *(contenido del PDF — no desarrollado oralmente en esta sesión)*

**Definición.** Clustering (o agrupamiento) es una técnica de aprendizaje no supervisado que busca identificar grupos naturales de observaciones dentro de un conjunto de datos, de manera que los elementos de un mismo grupo sean más similares entre sí que con los elementos de otros grupos.

- Es una técnica **exploratoria** que ayuda a comprender la organización interna de un conjunto de datos antes de formular hipótesis o construir modelos predictivos.
- Permite **resumir** grandes cantidades de datos mediante un número reducido de grupos representativos (centroides).
- El resultado del clustering depende directamente de **cómo se defina la similitud** entre observaciones: distintas medidas de similitud pueden producir agrupaciones distintas sobre los mismos datos.

**Por qué la distancia es central.** Muchos algoritmos de clustering utilizan la distancia para determinar qué observaciones son similares: distancias pequeñas indican observaciones parecidas; distancias grandes, observaciones diferentes. Por ejemplo, en **K-Means** cada observación se asigna al centroide más cercano, normalmente usando distancia euclidiana.

---

## 3. Puntos importantes antes de la clusterización

### 3.1 ¿Realmente hay grupos, o se están buscando clusters donde no los hay?

No todos los problemas son clusterizables. Antes de aplicar K-Means o Clustering Jerárquico conviene responder: ¿existen grupos reales en los datos, o se está forzando una estructura que no existe? El **Índice de Hopkins** mide la tendencia al agrupamiento (*cluster tendency*) antes de aplicar cualquier algoritmo — se desarrolla con detalle en la sección 5.

Tres escenarios de referencia:
- **Escenario 1 — datos con clústeres reales:** los puntos forman grupos claramente diferenciados.
- **Escenario 2 — datos aleatorios:** los puntos están dispersos sin ningún patrón.
- **Escenario 3 — datos uniformemente distribuidos:** no existen agrupaciones naturales.

### 3.2 Problema de las escalas

Con variables en escalas muy distintas (por ejemplo, edad entre 18 y 80, ingresos entre 10 000 y 2 000 000), la variable de mayor magnitud —ingresos— domina por completo el cálculo de la distancia, distorsionando los clusters resultantes. Por ello, antes de aplicar la mayoría de algoritmos de clustering suele ser necesario **estandarizar, normalizar o escalar** los datos. Al escalar, ambas variables contribuyen por igual a la distancia y los grupos resultantes son más claros y representativos.

### 3.3 Sensibilidad a valores atípicos

La distancia euclidiana usa diferencias al cuadrado, lo que provoca que observaciones extremadamente alejadas (outliers) tengan un impacto desproporcionado. Consecuencias: los outliers pueden distorsionar los resultados del clustering, y los centroides pueden desplazarse hacia regiones poco representativas de la mayoría de los datos — un solo punto atípico puede "arrastrar" el centroide de un cluster completo.

### 3.4 Alta dimensionalidad

A medida que aumenta el número de variables, las distancias entre puntos tienden a parecerse cada vez más entre sí, y resulta más difícil distinguir observaciones cercanas de lejanas — en baja dimensionalidad (2D) las distancias permiten separar bien los clústeres, pero en alta dimensionalidad (20D o más) las distancias euclidianas entre pares de puntos tienden a concentrarse en un rango estrecho, perdiendo su capacidad discriminativa. Este fenómeno se conoce como la **maldición de la dimensionalidad**, y es la razón por la que las técnicas de reducción de dimensionalidad (PCA, t-SNE, UMAP) son un paso previo habitual antes de clusterizar.

> **Idea clave resumida:** escalar variables, tratar outliers y considerar la dimensionalidad son pasos clave para lograr clústeres significativos y confiables.

---

## 4. Matriz de distancia

Sea $X = \{x_1, x_2, x_3, x_4, x_5\} \subset \mathbb{R}^5$ un conjunto de 5 observaciones con 5 variables ($v_1$ a $v_5$):

| Observación | $v_1$ | $v_2$ | $v_3$ | $v_4$ | $v_5$ |
|---|---:|---:|---:|---:|---:|
| $x_1$ | 2 | 1 | 0 | 4 | 3 |
| $x_2$ | 3 | 4 | 1 | 0 | 2 |
| $x_3$ | 0 | 2 | 3 | 1 | 4 |
| $x_4$ | 5 | 0 | 2 | 3 | 1 |
| $x_5$ | 1 | 3 | 4 | 2 | 0 |

La distancia euclidiana entre dos observaciones se define como $d(x_i, x_j) = \sqrt{\sum_{k=1}^{5}(x_{ik}-x_{jk})^2}$. Ejemplo entre $x_1$ y $x_2$:

$$d(x_1,x_2) = \sqrt{(2-3)^2+(1-4)^2+(0-1)^2+(4-0)^2+(3-2)^2} = \sqrt{28} \approx 5.292$$

Repitiendo el cálculo para todos los pares se obtiene la matriz de distancias $D = (d_{ij}) \in \mathbb{R}^{5\times 5}$:

| | $x_1$ | $x_2$ | $x_3$ | $x_4$ | $x_5$ |
|---|---:|---:|---:|---:|---:|
| $x_1$ | 0 | 5.292 | 4.243 | 5.831 | 4.359 |
| $x_2$ | 5.292 | 0 | 4.243 | 5.099 | 3.606 |
| $x_3$ | 4.243 | 4.243 | 0 | 5.196 | 3.162 |
| $x_4$ | 5.831 | 5.099 | 5.196 | 0 | 5.385 |
| $x_5$ | 4.359 | 3.606 | 3.162 | 5.385 | 0 |

**Propiedades de la matriz de distancias euclidianas:**
- $d_{ij} \geq 0 \; \forall i,j$
- $d_{ii} = 0 \; \forall i$ (distancia de un punto a sí mismo)
- $d_{ij} = d_{ji} \; \forall i,j$ — **simetría**
- $d_{ij} \leq d_{ik} + d_{kj} \; \forall i,j,k$ — **desigualdad triangular**

> Esta matriz es la base para métodos de agrupamiento jerárquico, MDS (escalamiento multidimensional), k-medoids, DBSCAN, entre otros.

Ejemplo geométrico adicional con 4 puntos en un plano ($A=(1,1)$, $B=(4,5)$, $C=(7,2)$, $D=(8,6)$): $d(A,B) = \sqrt{(4-1)^2+(5-1)^2} = \sqrt{25} = 5.00$; siguiendo la misma lógica se completa la matriz de distancias entre los 4 puntos (A-C: 6.08, A-D: 8.60, B-C: 4.24, B-D: 4.12, C-D: 4.12).

---

## 5. Distancias

### 5.1 Euclidiana

La medida de distancia más común entre dos puntos en un espacio geométrico; representa la longitud de la línea recta que conecta dos observaciones.

$$d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}$$

### 5.2 Manhattan (distancia $L^1$ o *city-block distance*)

Mide la distancia entre dos puntos sumando las diferencias absolutas en cada dimensión. El nombre viene de la ciudad de Manhattan, donde para desplazarse normalmente se recorren calles horizontales y verticales, en lugar de moverse en línea recta.

$$d(A,B) = \sum_{i=1}^{n} |x_i - y_i|$$

**Ejemplo:** $A=(2,3)$, $B=(7,8)$ → $|7-2| + |8-3| = 5 + 5 = 10$.

Mientras la distancia euclidiana mide la línea recta entre dos puntos, la Manhattan mide cuánto habría que recorrer si solo se pudiera avanzar horizontal y verticalmente (como las calles en cuadrícula de Manhattan).

### 5.3 Chebyshev (distancia del máximo)

Mide la separación entre dos puntos considerando únicamente la **mayor diferencia absoluta** entre sus variables.

$$d(x,y) = \max_i |x_i - y_i|$$

**Ejemplo:** con variables Edad (diferencia 5), Ingreso (diferencia 15) y Deuda (diferencia 8) entre dos individuos A y B, la distancia Chebyshev es $\max(5,15,8) = 15$ — solo importa la diferencia más grande, las demás se ignoran.

**Comparación visual de las tres distancias sobre una cuadrícula:** Euclidiana mide la diagonal directa; Manhattan asigna distancia 1 a los vecinos ortogonales y 2 a los diagonales (recorrido por calles); Chebyshev asigna distancia 1 tanto a vecinos ortogonales como diagonales (se mueve como el rey en el ajedrez).

### 5.4 Otras distancias (mención)

| Distancia | Uso típico |
|---|---|
| **Coseno** | Mide el ángulo entre dos vectores, no su magnitud — útil cuando importa la dirección/orientación de los datos más que su tamaño. |
| **Hamming** | Cuenta las posiciones en que dos secuencias binarias/categóricas difieren. |
| **Minkowski** | Generalización que incluye Manhattan ($p=1$), Euclidiana ($p=2$) y Chebyshev ($p=\infty$) como casos particulares de un mismo parámetro $p$. |
| **Jaccard** | Mide similitud entre conjuntos como la razón entre la intersección y la unión ($|A \cap B| / |A \cup B|$). |
| **Haversine** | Distancia entre dos puntos sobre la superficie de una esfera (coordenadas geográficas). |
| **Sørensen-Dice** | Similar a Jaccard, pondera distinto la intersección: $2|A \cap B| / (|A|+|B|)$. |

---

## 6. Evaluación de clustering (previo a aplicar los algoritmos)

### 6.1 Hopkins Statistic

Técnica que mide la tendencia natural de los datos a formar grupos **antes** de aplicar cualquier algoritmo de clustering. La idea: comparar la distribución de los datos reales contra una distribución completamente aleatoria generada en el mismo espacio. Si los datos tienen agrupamientos naturales, los puntos reales estarán más cerca entre sí que los puntos aleatorios generados.

El estadístico $H$ toma valores entre 0 y 1:

| Valor de H | Interpretación |
|---|---|
| $\approx 0.5$ | Datos distribuidos aleatoriamente, no clusterizables |
| $> 0.7$ | Evidencia de estructura de clusters |
| cercano a 1 | Fuerte tendencia a formar clusters |
| cercano a 0 | Distribución uniforme o regular |

**Procedimiento y fórmula:**
1. Se generan $n$ puntos aleatorios dentro del mismo espacio que los datos reales.
2. Para cada punto (real y aleatorio) se calcula la distancia a su vecino real más cercano.
3. Se suman las distancias de los puntos reales ($\sum x_i$) y las de los puntos aleatorios ($\sum y_i$).
4. $$H = \frac{\sum y_i}{\sum x_i + \sum y_i}$$

**Ejemplo 1 — datos con clusters reales:** $\sum x_i = 2.35$ (distancias muy cortas, porque los puntos reales están agrupados), $\sum y_i = 22.80$ → $H = 22.80/(2.35+22.80) = 0.91$ → fuerte tendencia a formar clusters.

**Ejemplo 2 — datos aleatorios sin estructura:** $\sum x_i = 15.10$, $\sum y_i = 14.80$ → $H = 14.80/(15.10+14.80) = 0.49$ → los datos se comportan como aleatorios, no hay evidencia de clusters.

### 6.2 VAT (Visual Assessment of Tendency)

Técnica visual diseñada específicamente para responder si los datos tienen tendencia a agruparse. Procedimiento: calcular las distancias entre todos los puntos, **reorganizar** esa matriz de distancias, y construir una imagen tipo mapa de calor (*heatmap*).

- Si los datos tienen clusters naturales, aparecen **bloques oscuros cuadrados sobre la diagonal principal** del heatmap tras la reorganización.
- Ejemplo con matriz $R$ de 5 observaciones: al reordenar filas/columnas ($R \to \tilde{R}$) los valores bajos (puntos cercanos) se agrupan en bloques cuadrados junto a la diagonal, revelando visualmente qué observaciones forman un grupo natural entre sí.
- En un heatmap con datos reales que sí tienen estructura, se observan bloques bien definidos y compactos. En un heatmap de **datos aleatorios** (sin estructura), el patrón se ve disperso y sin bloques diferenciables — solo ruido alrededor de la diagonal.

### 6.3 Otros objetos visuales

Los resultados de clustering suelen graficarse también como una nube de puntos en 2D (por ejemplo, sobre los dos primeros componentes de un PCA) con elipses de confianza dibujadas alrededor de cada cluster, coloreadas por grupo — una forma visual habitual de comunicar qué tan bien separados (o superpuestos) quedan los clusters finales.

---

## Conceptos clave de la clase

- El curso quedó una sesión atrasada respecto al cronograma; esta clase se usó para **cerrar t-SNE y UMAP** en profundidad, no para arrancar clustering — eso queda para la siguiente sesión.
- Para tareas de **agrupamiento visual**, t-SNE y UMAP superan claramente al PCA porque separan mejor los clusters (el PCA tiende a amontonar y confundir individuos de distintos grupos en el centro del plano); la recomendación explícita del docente es usar t-SNE/UMAP como paso de reducción previo al clustering.
- **Limitación compartida por t-SNE y UMAP:** no ofrecen biplot, por lo que agrupan bien visualmente pero no explican *qué* caracteriza a cada grupo — ese paso lo completa el clustering formal (próxima clase).
- UMAP resuelve la asimetría de sus probabilidades de conexión mediante la fórmula de **unión difusa (fuzzy union)**, indispensable para poder construir un grafo de vecindad consistente.
- El número de iteraciones en algoritmos como t-SNE/UMAP se decide por optimización: se detiene cuando un indicador (**silueta**) deja de mejorar de una iteración a la siguiente.
- **Clustering** es una técnica exploratoria de aprendizaje no supervisado que agrupa observaciones similares entre sí y las separa de las diferentes; el resultado depende directamente de cómo se defina la similitud (distancia).
- No todo dataset es clusterizable: el **Índice/Estadístico de Hopkins** ($H > 0.7$ indica tendencia a agrupar; $H \approx 0.5$ indica datos aleatorios) y el **VAT** (heatmap con bloques sobre la diagonal) son las herramientas para verificarlo antes de aplicar un algoritmo.
- Antes de clusterizar conviene atender tres problemas comunes: **escalas distintas** entre variables (dominan la distancia si no se estandariza), **sensibilidad a outliers** (distorsionan centroides) y **alta dimensionalidad** (maldición de la dimensionalidad: las distancias pierden poder discriminativo).
- Las distancias más usadas: **Euclidiana** (línea recta), **Manhattan** (recorrido en cuadrícula) y **Chebyshev** (solo la mayor diferencia); existen además Coseno, Hamming, Minkowski, Jaccard, Haversine y Sørensen-Dice para casos particulares.

---

## Fuera del PDF — logística y metodología

- Encuesta de mitad de periodo pendiente de completar por varios estudiantes; se habilitó tiempo extra en clase para llenarla.
- El curso está una sesión atrasada; el docente planea eliminar algunas sesiones de temas complementarios (cadenas de Markov, simulación de Montecarlo) —vistos en otros cursos como modelado matemático o cálculo para ciencia de datos— para recuperar el tiempo, sin afectar el contenido central de clustering.
- **Quiz la próxima semana** cubriendo cuatro temas: Análisis de Correspondencia Simple (ACS), Análisis de Correspondencia Múltiple (ACM), t-SNE y UMAP. Se reitera que la nota del quiz no es el objetivo principal — importa que cada estudiante identifique honestamente qué tan bien entendió estos temas antes de avanzar a clustering.
- Se ofreció una sesión de tutoría opcional el sábado de 9:00 a 10:00 a.m.
- Revisión individual de código de tarea con un estudiante (Jordan) sobre el biplot de MCA: confirmado que meter **todas** las variables numéricas y categóricas al modelo (no solo un subconjunto) es el enfoque correcto cuando el PDF no especifica un criterio de selección — se prioriza ver el comportamiento completo de la data en el plano antes de descartar variables.
