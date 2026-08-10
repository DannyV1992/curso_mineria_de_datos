# Clase 8 — Verificación de tendencia al agrupamiento y clustering jerárquico

**Curso:** Minería de Datos · Lead University
**Docente:** Rachit Vargas (MBA | Engineer)
**PDF fuente:** `Clases/PDFs/Clase 8 - Clustering jerarquico.pdf`

> La clase retomó justo donde había quedado pendiente la sesión anterior: primero se desarrolló el **Índice de Hopkins** y el **VAT**, apoyados en una presentación distinta a la de esta clase (por eso la sección 1 no tiene contraparte en el PDF de Clustering Jerárquico). Después de un receso, se pasó a la presentación de **Clustering Jerárquico** propiamente dicha —dendrograma, algoritmo aglomerativo, matriz de distancias, criterios de enlace y gráficos de interpretación—, reforzada con un Excel paso a paso y un caso real de una aerolínea. Quedó pendiente para la siguiente clase revisar el código en Python; esta semana no hay tarea.

---

## 1. Verificación de tendencia al agrupamiento *(no está en el PDF de esta clase)*

Antes de aplicar cualquier algoritmo de clustering hay que confirmar que los datos realmente guardan una estructura agrupable. Clusterizar datos sin estructura natural —y presentar esos clusters a una gerencia como si fueran reales— es un error grave: se estarían dando recomendaciones (de marketing, pricing, etc.) sobre grupos que **no son estadísticamente significativos**.

### 1.1 Índice de Hopkins

Es la técnica más formal para esto porque es un **indicador estadístico**: cuando un resultado está respaldado por estadística, nadie puede refutarlo apelando solo a la interpretación visual (es la misma lógica que el p-value en una regresión: si es menor al umbral, el efecto es significativo).

**Idea del método:** comparar la distribución de los datos reales contra una distribución generada de forma completamente aleatoria. Si los datos reales tienen una tendencia natural a agruparse, sus puntos deben estar más cerca entre sí que los puntos aleatorios.

**Procedimiento:**
1. Generar un conjunto de puntos artificiales completamente aleatorios (mismo rango que los datos reales).
2. Calcular la matriz de distancias del vecino más cercano tanto para los puntos reales como para los aleatorios.
3. Sumar todas las distancias de los datos reales ($\sum w_i$) y todas las distancias de los datos aleatorios ($\sum u_i$).
4. Calcular el estadístico:

$$H = \frac{\sum u_i}{\sum u_i + \sum w_i}$$

**Tabla de interpretación** ($H$ toma valores entre 0 y 1):

| Valor de $H$ | Interpretación |
|---|---|
| $\leq 0.5$ | Los datos se distribuyen de forma aleatoria; **no son clusterizables**. |
| $> 0.7$ | Hay evidencia de estructura de clusters (aunque no necesariamente bien formados). |
| $\to 1$ | Fuerte tendencia a formar clusters — el escenario que se busca. |
| $\to 0$ | Distribución uniforme o regular — tampoco es clusterizable. |

**Ejemplo con dos casos:**

- **Caso 1:** suma de distancias reales $=2.35$, suma de distancias aleatorias $=22.80$. $H = \frac{22.80}{2.35+22.80} \approx 0.91$ → cercano a 1, hay tendencia a clusterizar. Visualmente, este dataset muestra 4 clusters claramente diferenciados.
- **Caso 2:** suma de distancias aleatorias $=14.80$, suma de distancias reales $=15.10$. $H = \frac{14.80}{15.10+14.80} \approx 0.50$ → los datos son completamente aleatorios, **no hay ninguna estructura que clusterizar**. Forzar una clusterización sobre este dataset sería "un invento" que no sirve para nada.

### 1.2 VAT (Visual Assessment of Tendency)

Técnica visual (no estadística) diseñada para responder la misma pregunta: ¿hay tendencia a agrupar o no?

**Procedimiento:**
1. Calcular la distancia entre todos los puntos (matriz de distancias).
2. Reordenar esas distancias de menor a mayor, colocando las más pequeñas sobre la **diagonal principal**.
3. Construir un *heatmap* (mapa de calor) con esa matriz reordenada.

**Lectura del heatmap:**
- Si aparecen **bloques oscuros** sobre la diagonal principal → esas distancias son pequeñas → los individuos correspondientes son similares entre sí → hay estructura de clusters.
- Cuanto **mayor sea la superficie oscura cubierta**, más fuerte es la tendencia a clusterizar en la mayoría de los datos.
- Si el bloque oscuro es **muy pequeño** (una esquina), significa que solo una minoría de los puntos guarda una distancia corta entre sí; la mayoría de los datos está dispersa y no vale la pena clusterizar sobre el total (dos individuos parecidos entre 20 no justifican un cluster si los otros 18 quedan completamente dispersos).
- Bloques **más claros** → distancias mayores → mayor dispersión, menor tendencia a agrupar.

> *"Yo puedo clusterizar acá, pero lo que yo estoy haciendo es un invento, eso no sirve para nada."*

---

## 2. Objetivos de los algoritmos de clustering *(no está en el PDF de esta clase)*

Todo algoritmo de clustering —jerárquico, K-Means o el que sea— persigue dos objetivos simultáneos, ligados directamente al concepto de centroide:

| Objetivo | Definición | Qué significa lograrlo |
|---|---|---|
| **Minimizar la distancia intraclase** | Distancia entre los puntos que pertenecen a un **mismo** cluster y su centroide. | Los individuos de un cluster están compactos, cerrados, poco dispersos. |
| **Maximizar la distancia interclase** | Distancia entre los centroides de clusters **distintos**. | Los clusters están bien separados entre sí. |

Si ambos objetivos se cumplen, los clusters son buenos: compactos internamente y bien diferenciados entre ellos. El problema aparece cuando la distancia interclase entre dos clusters es pequeña (sus centroides quedan cerca): los puntos en la **frontera** entre ambos clusters pueden quedar **mal clasificados** —el algoritmo asignó un punto al cluster 2 cuando en realidad su comportamiento se parece más al cluster 3.

- No hay forma de prevenir esto de raíz: es inherente a que los clusters estén cerca. El **indicador de silueta** (desarrollo pendiente, en la clase de evaluación de clustering) permite identificar cuáles individuos están mal clasificados.
- Cuando esto ocurre, la decisión de a qué cluster asignar definitivamente a un individuo fronterizo se vuelve **manual y de criterio del analista** — el algoritmo por sí solo es insuficiente.
- **Ejemplo (tarjetas de crédito):** un cliente puede tener un comportamiento mixto (viaja bastante y también consume mucho en un mercado específico), quedando a caballo entre dos clusters al mismo tiempo. El analista decide si lo mantiene en ambos segmentos o lo asigna a uno solo — decisión especialmente delicada si ese cliente es una empresa grande.

---

## 3. Clustering jerárquico vs. K-Means: cuándo usar cada uno *(no está en el PDF de esta clase)*

| | Clustering jerárquico | K-Means |
|---|---|---|
| **Naturaleza** | Determinístico (con excepciones muy puntuales) y **recursivo** | Estocástico |
| **Calidad de resultados** | En la mayoría de los casos produce **mejores resultados** | — |
| **Tamaño de dataset recomendado** | **Pequeño** — el algoritmo recursivo tarda bastante en datasets grandes | **Grande** — es más rápido |
| **Umbral de "grande" o "pequeño"** | No está formalmente definido en la literatura; depende del criterio del analista | — |

Ambos algoritmos buscan lo mismo (encontrar clusters dentro de los datos), pero llegan por caminos distintos: el jerárquico construye un árbol de fusiones; K-Means (que se verá más adelante) funciona de otra forma. Cuál usar depende de las características del dataset (tamaño, tiempo disponible, necesidad de determinismo).

---

## 4. El problema de la hipersegmentación *(no está en el PDF de esta clase)*

Al diseñar una campaña de marketing sobre un volumen grande de clientes, hay un balance necesario en la cantidad de clusters (segmentos) a definir:

- **Muy pocos clusters (o ninguno):** la campaña es demasiado general — nadie se identifica con el mensaje y las tasas de conversión bajan.
- **Demasiados clusters:** se cae en la **hipersegmentación** — el analista termina clusterizando en grupos que ni siquiera existen en la realidad (p. ej., decir que hay 10 tipos de clientes cuando solo existen 4 comportamientos reales). El algoritmo puede, técnicamente, formar esos 10 grupos, pero varios de ellos son inventados.

Una vez que los clusters reales están identificados (p. ej. clientes "azules", "verdes", "amarillos", "rojos", cada uno con un comportamiento de consumo distinto), es mucho más sencillo destinar una campaña de marketing, política de precios o promoción específica a cada comportamiento — ese es, en esencia, el propósito de negocio detrás de clusterizar personas, empresas o países.

---

## 5. Clustering jerárquico: definición y dendrograma

El **clustering jerárquico** es un método de agrupamiento que organiza los datos en una estructura de árbol llamada **dendrograma**, basada en las **similitudes** (distancias) entre los individuos — cada vez que el PDF dice "similitud", se refiere a distancia.

Existen dos enfoques:

| Enfoque | Dirección | Uso |
|---|---|---|
| **Aglomerativo** (*bottom-up*) | Cada punto empieza como su propio cluster; se fusionan progresivamente los más cercanos hasta llegar a un único cluster. | Es el que normalmente se utiliza. |
| **Divisivo** (*top-down*) | Se empieza con un único cluster que contiene todos los datos y se va dividiendo sucesivamente. | Mencionado, no desarrollado en detalle. |

En el eje $y$ del dendrograma se lee la **distancia** a la que ocurre cada fusión: en la base, cada punto es su propio cluster; a medida que aumenta la altura, se van formando clusters cada vez más grandes, hasta llegar a un único cluster con todos los datos (parte superior del árbol).

### 5.1 Definición formal (algoritmo aglomerativo)

Dado un conjunto de datos $X = \{x_1, x_2, \dots, x_n\}$ y una medida de disimilitud $d(\cdot,\cdot)$:

1. **Inicialización:** cada punto $x_i \in X$ forma un cluster individual $C_i = \{x_i\}$.
2. **Iteración:** fusionar los dos clusters $C_p$ y $C_q$ que minimizan $d(C_p, C_q)$ según un **criterio de enlace** (*linkage*).
3. **Repetir** el paso 2 hasta obtener un único cluster que contiene todos los puntos.

Formalmente, con $I=\{1,\dots,n\}$ el conjunto de objetos, $P_h$ la partición en el paso $h$ y $\delta$ el criterio de agregación:

- $P_0 = \{\{1\},\{2\},\dots,\{n\}\}$, $h=0$.
- Se fusionan los dos nodos de $P_h$ más cercanos: $\delta(x,y) = \min\{\delta(l,k) \mid l,k \in P_h,\, l \neq k\}$.
- Se actualiza $P_h$: $h \leftarrow h+1$; $P_h \leftarrow [P_h \cup \{x \cup y\}] - \{x,y\}$.
- **Criterio de parada:** si $h < n-2$, se regresa al paso de fusión; si no, se hace la última fusión y termina.

### 5.2 Ejemplo visual paso a paso (6 puntos: A–F)

Con 6 puntos en $\mathbb{R}^2$, el algoritmo avanza así:

| Paso | Qué ocurre |
|---|---|
| 1 | Inicialización: los 6 puntos son 6 clusters individuales. |
| 2 | Se fusionan los dos más cercanos: $A$ y $B$ (distancia $\approx 1$). |
| 3 | Se fusionan $C$ y $D$ (distancia $\approx 1$). |
| 4 | Se fusionan los clusters $(A,B)$ y $(C,D)$ — ya no son puntos individuales sino clusters, y la fusión ocurre entre clusters, no entre puntos sueltos. |
| 5 | Se fusiona el par restante más cercano: $E$ y $F$. |
| 6 | Última fusión: todos los puntos quedan en un único cluster (altura $\approx 4$–$5$ en el dendrograma). |

Un detalle importante de la mecánica del algoritmo: **nunca se fusionan tres puntos a la vez** — la fusión siempre ocurre de dos en dos (dos puntos, dos clusters, o un punto con un cluster), nunca en grupos de tres o más en un mismo paso.

### 5.3 ¿Dónde cortar el dendrograma?

El algoritmo, por diseño, siempre termina fusionando **todo** en un único gran cluster — construye el árbol completo sin importar cuántos clusters se quieran al final. La cantidad de clusters se decide **después**, trazando una línea horizontal sobre el dendrograma:

- El número de clusters resultante es igual al número de **intersecciones** que corta esa línea (cada intersección con forma de "T invertida" representa una fusión).
- Trazar la línea más arriba → menos intersecciones → menos clusters (más generales).
- Trazar la línea más abajo → más intersecciones → más clusters (más específicos, más fragmentados).

**Esta es una desventaja del método:** la elección de cuántos clusters formar queda sujeta a la interpretación de quien traza la línea — no hay una única respuesta correcta a simple vista. Elegir demasiados clusters reintroduce el problema de la **hipersegmentación** (ver sección 4): mensajes de marketing más específicos implican más presupuesto, más comerciales, más estrategias y políticas de descuento diferenciadas.

Para resolver objetivamente en cuántos clusters cortar existen dos técnicas estadísticas (desarrollo pendiente para clases futuras):
- **Codo de Jambu** (*elbow method*): grafica la inercia contra la cantidad de clusters; el número óptimo es el punto donde se forma un "codo" en la curva.
- **Indicador de silueta**: técnica complementaria que también ayuda a determinar el número óptimo de clusters.

En última instancia, la cantidad de clusters también puede estar limitada por restricciones de negocio (p. ej. presupuesto suficiente solo para 3 campañas diferenciadas, aunque el codo sugiera 5).

---

## 6. Matriz de distancias para clustering jerárquico

Para ejecutar un clustering jerárquico se necesitan tres elementos: **los datos**, una **función de distancia** (para construir la matriz de distancias) y un **criterio de enlace**.

### 6.1 Cálculo de una matriz de distancias (ejemplo con notas escolares)

Con $n$ estudiantes y $p$ asignaturas, cada estudiante es un punto en $\mathbb{R}^p$. La distancia euclídea entre dos estudiantes $i$ y $j$ es:

$$d_{ij} = \sqrt{\sum_{k=1}^{p}(x_{ik}-x_{jk})^2}$$

**Ejemplo:** Lucía $=(7.0, 6.5, 9.2, 8.6, 8.0)$, Pedro $=(7.5, 9.4, 7.3, 7.0, 7.0)$:

$$d_{\text{Lucía,Pedro}} = \sqrt{(-0.5)^2+(-2.9)^2+(1.9)^2+(1.6)^2+(1.0)^2} = \sqrt{15.83} \approx 3.98$$

Solo es necesario calcular una vez cada par $(i,j)$ con $i<j$: la matriz de distancias es **simétrica** ($d_{ij}=d_{ji}$, $d_{ii}=0$), así que el resto se obtiene reflejando el triángulo ya calculado.

---

## 7. Ejemplo completo: cálculo de distancias y construcción del dendrograma (5 puntos, enlace promedio)

**Dataset** (5 puntos en $\mathbb{R}^2$):

| Punto | $x_1$ | $x_2$ |
|---|---:|---:|
| A | 1 | 2 |
| B | 2 | 1 |
| C | 4 | 1 |
| D | 5 | 4 |
| E | 7 | 3 |

**Matriz de distancias inicial** (euclídea, $d(i,j)=\sqrt{(x_{1i}-x_{1j})^2+(x_{2i}-x_{2j})^2}$):

| | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| A | 0 | 1.41 | 3.16 | 5.00 | 6.32 |
| B | 1.41 | 0 | 2.00 | 4.24 | 5.39 |
| C | 3.16 | 2.00 | 0 | 3.16 | 3.61 |
| D | 5.00 | 4.24 | 3.16 | 0 | 2.24 |
| E | 6.32 | 5.39 | 3.61 | 2.24 | 0 |

**Fusiones sucesivas (enlace promedio):**

1. **Mínima distancia:** $d(A,B)=1.41$ → se fusionan $A$ y $B \to (AB)$.
2. **Mínima distancia:** $d(D,E)=2.24$ → se fusionan $D$ y $E \to (DE)$.
3. **Mínima distancia:** $d(AB,C)=2.55$ → se fusionan $(AB)$ y $C \to (ABC)$. Se calcula como $\frac{1}{2}(d(A,C)+d(B,C)) = \frac{1}{2}(3.16+2.00) = 2.58 \approx 2.55$ (promedio de enlace simple/promedio).
4. **Última fusión:** única distancia restante, $d(ABC,DE)=4.44$ — promedio de **todos** los pares cruzados entre $\{A,B,C\}$ y $\{D,E\}$: $\frac{1}{6}(5.00+6.32+4.24+5.39+3.16+3.61)=4.44$.

**Mecánica clave del criterio de enlace:** en cuanto dos puntos se fusionan en un cluster, sus distancias individuales **desaparecen** de la matriz — ya no tiene sentido, por ejemplo, seguir hablando de "la distancia de $A$ a $C$" una vez que $A$ pasó a formar parte del cluster $(AB)$. Esa distancia se **reemplaza** por un valor nuevo calculado con el criterio de enlace elegido (aquí, un promedio) entre el cluster recién formado y cada punto o cluster restante. Los puntos que **aún no se han fusionado** conservan sus distancias originales intactas.

**Dendrograma final:** $A$–$B$ se fusionan a altura 1.41; $D$–$E$ a altura 2.24; $(AB)$–$C$ a altura 2.55; $(ABC)$–$(DE)$ a altura 4.44 (fusión final, un único cluster).

---

## 8. Criterios de enlace (linkage)

En cada paso del algoritmo hay que decidir cómo medir la distancia entre dos **clusters** (no entre dos puntos individuales) para saber cuáles fusionar. Existen cuatro criterios principales — no hay uno universalmente mejor; cuál usar depende de la estructura de los datos, y se resuelve en la clase de **evaluación de clustering** probando todos y comparando resultados (silueta, etc.). Esto aplica igual que con las funciones de distancia: en minería de datos y machine learning no hay un camino único, hay que **probar todos los métodos** y quedarse con el que dé mejores resultados.

| Criterio | Fórmula | Comportamiento |
|---|---|---|
| **Salto mínimo** (*single linkage*) | $d_{\min}(A,B) = \min_{x\in A,\,y\in B} d(x,y)$ | Usa el par de puntos más cercano entre ambos clusters. Tiende a formar clusters alargados o en cadena (efecto de encadenamiento). |
| **Salto máximo** (*complete linkage*) | $d_{\max}(A,B) = \max_{x\in A,\,y\in B} d(x,y)$ | Usa el par más lejano. Exige que todos los puntos estén relativamente cerca; forma clusters compactos y de tamaño similar. |
| **Promedio** (*average linkage*) | $d_{\text{prom}}(A,B) = \frac{1}{|A||B|}\sum_{x\in A}\sum_{y\in B} d(x,y)$ | Promedia todas las distancias cruzadas entre ambos clusters. Compromiso entre los dos anteriores: evita cadenas largas pero permite formas y tamaños moderadamente variables. |
| **Ward** (mínima varianza) | $d_{\text{Ward}}(A,B) = \frac{n_A n_B}{n_A+n_B}\lVert \bar{x}_A - \bar{x}_B\rVert^2$ | Fusiona los clusters que produzcan el **menor aumento** en la varianza intra-cluster (usa centroides y tamaños de cluster). Tiende a formar clusters esféricos y de tamaño similar; es el criterio **más usado en la práctica**. |

### 8.1 Ejemplo comparativo (4 puntos: $X1=(0,0)$, $X2=(5,0)$, $X3=(3,0)$, $X4=(2,1)$)

En todos los métodos, el primer paso es igual: la distancia mínima inicial es $d(X3,X4)=1$, así que $X3$ y $X4$ se fusionan primero (altura 1). A partir de ahí, cada criterio recalcula las distancias del nuevo cluster $C_1=\{X3,X4\}$ de forma distinta:

| Criterio | Distancia $C_1$–$X1$ | Distancia $C_1$–$X2$ | Altura de la fusión final |
|---|---:|---:|---:|
| **Salto mínimo** | $\min(3,2)=2$ | — | **2** |
| **Salto máximo** | $\max(3,2)=3$ (paso 2); recalculado tras la 2ª fusión | — | **5** |
| **Promedio** | $(3+2)/2=2.5$ | $(5+4.24)/2=3.12$ | **3.12** |
| **Ward** | 2.33 (fórmula de Ward con centroides) | 3.33 | **4.80** |

Con exactamente el mismo dataset de 4 puntos, los cuatro criterios producen alturas de fusión final distintas (2, 5, 3.12 y 4.80 respectivamente) — la elección del criterio de enlace cambia tanto la forma como el número final de clusters que se obtienen al cortar el dendrograma a una altura dada.

---

## 9. Gráficos para la interpretación de clusters

Una vez obtenido el agrupamiento, estos gráficos ayudan a comprender y caracterizar cada cluster (qué variables lo definen, cómo se comporta):

| Gráfico | Uso |
|---|---|
| **Scatter plot coloreado por cluster** | Proyección en 2D (p. ej. vía PCA) mostrando la separación entre clusters. |
| **Boxplots por cluster** | Distribución de cada variable dentro de cada cluster; identifica qué variables diferencian los grupos. |
| **Barplot de medias por cluster** | Promedio de cada variable por cluster — resumen simple del comportamiento de cada grupo. |
| **Heatmap de medias** | Mapa de calor de las medias estandarizadas (colores cálidos = valores altos, fríos = valores bajos). |
| **Radar chart** | Perfil promedio de cada cluster en todas las variables escaladas, superpuesto en un mismo gráfico. |
| **Violin plots** | Distribución y densidad de cada variable dentro de cada cluster. |
| **Silhouette plot** | Mide qué tan bien asignado está cada punto a su cluster — no solo es interpretativo, sirve para evaluar si los clusters están bien formados (desarrollo en la clase de evaluación de clustering). |
| **Tamaño de clusters** | Cantidad de observaciones por cluster — útil para detectar grupos muy pequeños o desbalanceados. |
| **Parallel coordinates plot** | Cada línea es una observación; permite ver patrones multivariados y diferencias entre clusters. |
| **Sankey** | Mencionado pero considerado confuso — el docente no lo utiliza. |

**Preferencia práctica del docente:** con audiencias no técnicas (p. ej. un gerente financiero), los **barplots** y **radar charts** ("*rarcharts*") son los más fáciles de explicar — todo el mundo entiende una barra de inmediato. Gráficos como el PCA scatter requieren explicar antes qué representan los ejes, y con audiencias sin bagaje matemático el resultado puede ser una explicación fallida aunque el gráfico sea correcto. La elección del gráfico depende del tipo de audiencia y de cuánto tiempo hay para explicarlo.

### 9.1 Caso real: aerolínea y membresía premium

Una aerolínea clusterizó a sus clientes en 3 grupos según: frecuencia de viaje, lealtad con la aerolínea, interés en beneficios premium, gasto promedio y sensibilidad al precio — buscando decidir a qué cluster ofrecer una nueva membresía premium (embarque prioritario, acceso VIP, acumulación de millas).

| Cluster | Perfil |
|---|---|
| **Cluster 1** | Poco sensible al precio pero de forma engañosa: baja lealtad, baja frecuencia de viaje, poco interés en beneficios premium, el gasto más bajo de los tres. |
| **Cluster 2** | Mayor gasto promedio, mayor frecuencia de viaje, **menos** sensible al precio, mayor lealtad, alto interés en beneficios premium. |
| **Cluster 3** | Más sensible al precio, interesado en beneficios premium, gasto y frecuencia de viaje intermedios entre los otros dos clusters. |

Usando un radar chart (*rarchart*) con escala 1–5, el **Cluster 2** resultó ser el más leal a la aerolínea (escala 4, contra 2 del Cluster 1 y 3.5 del Cluster 3) y el que más gasta en promedio — es el candidato correcto para ofrecer la membresía premium, aunque a primera vista otro cluster pudiera parecer "mejor distribuido" visualmente. Sin el análisis de clustering previo esta conclusión no habría sido posible.

---

## 10. Aplicación del modelo a nuevos individuos *(no está en el PDF de esta clase)*

Una vez entrenado el algoritmo de clustering sobre la data histórica, **no es necesario reprocesar todo el dataset** cada vez que llega un individuo nuevo: el algoritmo ya "aprendió" los patrones y clusters existentes, y puede asignar directamente al nuevo individuo (con sus propias variables) al cluster que le corresponda.

- **Reentrenamiento periódico:** sí es recomendable actualizar el modelo cada cierto tiempo, porque el comportamiento de consumo cambia (p. ej. la creciente relevancia de la sostenibilidad/ESG en las decisiones de compra, o eventos disruptivos como la pandemia). La frecuencia de reentrenamiento depende del negocio, no hay una regla fija.

---

## Conceptos clave de la clase

- **Índice de Hopkins:** estadístico entre 0 y 1 que mide la tendencia al agrupamiento comparando datos reales contra datos aleatorios ($H=\sum u_i/(\sum u_i+\sum w_i)$); $H\to 1$ indica fuerte tendencia a clusterizar, $H\leq 0.5$ o $H\to 0$ indica ausencia de estructura.
- **VAT:** técnica visual que reordena la matriz de distancias en un heatmap; bloques oscuros extensos sobre la diagonal indican tendencia a clusterizar.
- Todo algoritmo de clustering busca **minimizar la distancia intraclase** (compactar cada cluster) y **maximizar la distancia interclase** (separar los clusters entre sí); cuando ambos centroides quedan cerca, aparecen individuos mal clasificados en la frontera (indicador de silueta, decisión manual).
- **Clustering jerárquico** vs. **K-Means:** el jerárquico es determinístico y recursivo, funciona mejor en datasets pequeños y suele dar mejores resultados; K-Means es estocástico y más rápido en datasets grandes.
- La **hipersegmentación** (demasiados clusters) inventa grupos que no existen realmente; hay que balancear especificidad de negocio contra sobre-fragmentación.
- El **clustering jerárquico** construye un **dendrograma** mediante el algoritmo aglomerativo (*bottom-up*): inicializa cada punto como cluster individual y fusiona de dos en dos según la distancia mínima, hasta llegar a un único cluster.
- El número final de clusters se decide **cortando el dendrograma** a una altura elegida por el analista — decisión subjetiva que el **codo de Jambu** y el **indicador de silueta** ayudan a objetivar (desarrollo pendiente).
- Al fusionar dos puntos o clusters, sus distancias individuales **desaparecen** de la matriz y se reemplazan según el **criterio de enlace** elegido: **salto mínimo**, **salto máximo**, **promedio** o **Ward** (el más usado en la práctica); cada uno produce dendrogramas y número de clusters distintos sobre el mismo dataset.
- Gráficos como **barplots**, **radar charts**, **boxplots**, **heatmaps** y **silhouette plots** son las herramientas estándar para interpretar clusters; en audiencias no técnicas, barplots y radar charts son los más fáciles de comunicar.
- Un modelo de clustering ya entrenado puede asignar directamente nuevos individuos a los clusters existentes, sin reprocesar todo el dataset; el reentrenamiento periódico se recomienda cuando el comportamiento subyacente cambia con el tiempo.

---

## Fuera del PDF — logística y metodología

- Receso de 20 minutos a la mitad de la clase (7:40 h aprox.) antes de pasar al Excel de criterios de enlace.
- **No hay tarea esta semana ni la siguiente semana** (dado que el tema es más algorítmico y se avanza paso a paso).
- Quedó pendiente revisar el **código en Python** de clustering jerárquico — no hubo tiempo en la sesión regular.
- Se propuso una **tutoría opcional el viernes a las 19:00 h** (ajustada desde las 19:00/20:30 h originalmente sugeridas) para repasar el código antes de la siguiente clase regular del martes.
- La clase seguía una semana de atraso respecto al cronograma original del curso.
