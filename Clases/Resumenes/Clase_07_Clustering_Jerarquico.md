# Clase 7 — Quiz de repaso, fundamentos de clustering y clustering jerárquico

**Curso:** Minería de Datos · Lead University
**Docente:** Rachit Vargas (MBA | Engineer)
**PDF fuente:** `Clases/PDFs/Clase 7 - Clustering jerarquico.pdf`

> La clase abrió con el **Quiz 2** (repaso de ACS/ACM, t-SNE y UMAP) y continuó con el desarrollo oral —por primera vez con ejemplos en vivo— de los **fundamentos del clustering**: definición, centroide, matriz de distancias y tipos de distancia. Este bloque corresponde temáticamente al PDF de la Clase 6, cuya estructura ya había quedado documentada sin audio; en esta sesión el docente sí lo explicó en detalle. El contenido propio del PDF de esta clase —**clustering jerárquico**: dendrograma, algoritmo aglomerativo, criterios de enlace (*linkage*) y gráficos de interpretación— **no llegó a desarrollarse oralmente**: el docente se detuvo justo antes del Índice de Hopkins, notó cansancio general en el grupo y decidió posponer Hopkins, VAT y clustering jerárquico completo para la siguiente sesión. Por eso, la sección 4 de este resumen documenta ese contenido a partir del PDF, como adelanto de lo que se desarrollará con ejemplos la próxima clase.

---

## 1. Fundamentos del clustering

### 1.1 Definición

**Clustering** (agrupamiento) es una técnica de **aprendizaje no supervisado** que busca identificar **grupos naturales** en un conjunto de datos: los elementos de un mismo grupo deben ser más similares entre sí que con los elementos de otros grupos. Las palabras clave del concepto son *aprendizaje no supervisado* y *grupos o estructuras naturales que existen en los datos y deben descubrirse*.

- Es una técnica **exploratoria**, parte de un EDA (análisis exploratorio de datos): ayuda a comprender la organización interna de los datos **antes** de formular hipótesis o entrenar modelos supervisados/predictivos. El clustering, la reducción de dimensionalidad y los demás algoritmos no supervisados del curso se ubican en esta fase exploratoria previa al modelado predictivo.
- Permite **resumir** grandes volúmenes de datos mediante un número reducido de grupos representativos: cada grupo se resume con un **centroide** en vez de tener que leer la distribución completa de todos sus puntos — es más manejable decir "el cluster de países ricos exporta X y tiene Y % de pobreza" que describir cada país individualmente.
- El resultado depende directamente de **cómo se defina la similitud** (la función de distancia elegida): con distancia euclidiana el agrupamiento final no necesariamente coincide con el que resulta de usar, por ejemplo, distancia Manhattan.

### 1.2 El centroide

El **centroide** es el representante promedio de las características de un cluster: el promedio de todas las observaciones (todos los puntos) que pertenecen a ese cluster.

**Ejemplo (edad y cantidad de hermanos de varios estudiantes):**

| Estudiante | Edad | Hermanos |
|---|---:|---:|
| Jordan | 22 | 3 |
| Jordi | 19 | 2 |
| Braulio | 33 | 5 |
| Roberto | 40 | 0 |

- Jordan y Jordi están cerca en el plano (edad-hermanos parecidas) → tienden a formar un mismo cluster. Braulio y Roberto quedan más alejados de ese grupo y entre sí, aunque igualmente podrían agruparse dependiendo de la decisión del analista sobre cuántos clusters formar.
- Centroide de {Jordan, Jordi} en edad: $(22+19)/2 = 20.5$.
- Si se agrega Braulio al mismo cluster, el centroide se recalcula: $(22+19+33)/3 \approx 24.7$ — el centroide **se desplaza** al incorporar un nuevo punto; esto es la base de por qué un outlier puede arrastrar el centroide hacia una región poco representativa (ver 1.6).

**Aplicación de negocio (ejemplo del profesor):** clusterizando países por ingreso, tasa de pobreza, ingreso per cápita, exportaciones e importaciones, un cluster puede caracterizarse por un centroide de "~2 000 000 USD/día de ingreso, ~7 000 000 USD/día de exportaciones" y otro por "~5 000 000 USD/día de ingreso, ~3 000 000 USD/día de exportaciones" — el centroide resume en un solo número por variable el comportamiento típico del grupo.

### 1.3 La similitud (distancia) determina la agrupación

Muchos algoritmos de clustering usan la distancia para decidir qué observaciones son similares: **distancias pequeñas → observaciones parecidas; distancias grandes → observaciones diferentes**. En K-Means, por ejemplo, cada observación se asigna al centroide más cercano, normalmente con distancia euclidiana.

Como existen muchas funciones de distancia (euclidiana, Manhattan, Chebyshev, coseno, Jaccard, entre otras) y cada una llega a un resultado distinto, el clustering final **depende de cuál distancia se elija** — usar euclidiana no necesariamente produce el mismo agrupamiento que usar Manhattan sobre los mismos datos.

### 1.4 ¿Los datos son realmente clusterizables?

No todos los datos son clusterizables. Presentar clusters que no existen realmente a una gerencia (por ejemplo, sugerir una estrategia de marketing basada en segmentos inventados) es un error grave y frecuente: se estaría dando un *insight* falso sobre clientes que ni siquiera existen como grupo diferenciado.

Para evitarlo existe el **Índice de Hopkins**, que mide la tendencia al agrupamiento (*cluster tendency*) antes de aplicar K-Means o clustering jerárquico. El desarrollo completo de este índice —fórmula, procedimiento y ejemplos numéricos— **quedó pendiente para la siguiente clase**; solo se mencionaron los tres escenarios de referencia (datos con clusters reales, datos aleatorios sin patrón, datos uniformemente distribuidos sin agrupaciones naturales).

### 1.5 Problema de las escalas

Con variables en escalas muy distintas (p. ej. edad de 18 a 80 años vs. ingreso de 10 000 a 2 000 000 USD/año), la variable de mayor magnitud domina por completo el cálculo de la distancia y distorsiona los clusters — es comparar "una manzana con una pera". La solución es **estandarizar, normalizar o escalar** los datos antes de clusterizar, llevando todas las variables a rangos comparables (p. ej. de −1 a 1), de modo que cada una contribuya de forma equilibrada a la distancia final.

### 1.6 Sensibilidad a valores atípicos (outliers)

La distancia euclidiana usa diferencias al cuadrado, por lo que un punto extremadamente alejado tiene un impacto desproporcionado en el resultado.

**Ejemplo (salarios en una empresa):** si se clusterizan salarios y edades incluyendo al CEO con un salario muchísimo mayor al del resto, el algoritmo puede terminar detectando solo dos "clusters" — CEO vs. todos los demás — en vez de los grupos reales (operativo, call center, alta gerencia). Al eliminar ese outlier, los datos restantes sí permiten separar correctamente los tres grupos representativos.

- El outlier **arrastra el centroide** hacia una región poco representativa: si se incluye a Jeff Bezos en el mismo cluster que el resto de empleados, el "salario promedio del cluster" deja de tener sentido.
- El efecto se replica en cascada: si el outlier existe desde el origen de los datos y se conserva al hacer reducción de dimensionalidad (PCA, t-SNE, UMAP), sigue apareciendo como outlier en el espacio reducido — reducir dimensionalidad solo reescala y proyecta los puntos, no elimina el desplazamiento que provoca un valor atípico.
- Distinto a los modelos supervisados: KNN es sensible a outliers, pero otros como Naive Bayes o árboles de decisión no necesariamente requieren eliminarlos — la decisión de tratar o no un outlier depende del algoritmo que se vaya a aplicar después.

### 1.7 Alta dimensionalidad

A mayor número de variables, las distancias entre pares de puntos tienden a concentrarse en un rango estrecho y resulta más difícil distinguir observaciones cercanas de lejanas — fenómeno conocido como la **maldición de la dimensionalidad**. Por eso se recurre a PCA, ACM, t-SNE o UMAP antes de clusterizar: no solo para visualizar en 2D, sino para retener la información esencial sin la redundancia de variables que no aportan nada nuevo.

### 1.8 ¿Clusterizar sobre datos originales o sobre componentes reducidos?

Pregunta abierta relevante cuando el dataset original tiene muchas variables y el costo computacional de calcular todas las distancias por pares es alto.

- **Recomendación del docente:** clusterizar siempre sobre los **datos originales**. Es posible clusterizar sobre las primeras componentes de un PCA si estas retienen, por ejemplo, el 95 % de la información (evitando cargar variables que no suman nada), pero esto tiene una desventaja importante: el PCA es una **combinación lineal** de las variables originales, por lo que los centroides resultantes quedan en **coordenadas artificiales** y se pierde la interpretabilidad directa (no se puede explicar con precisión a un gerente "de dónde sale" cada centroide en términos de las variables de negocio originales). Revertir esa combinación lineal a los valores originales no es una práctica establecida ni claramente viable.
- **Flujo recomendado (pipeline correcto):** aplicar la clusterización sobre los datos originales y usar el PCA únicamente como herramienta de **visualización** de los clusters ya formados.

---

## 2. Matriz de distancias

Se retoma el ejemplo de 5 observaciones ($x_1$ a $x_5$) con 5 variables ($v_1$ a $v_5$) ya documentado en la Clase 6 (sección 4), ahora desarrollado oralmente con ejemplos de cálculo:

- La distancia entre $x_1$ y $x_2$ se obtiene con la fórmula euclidiana generalizada a 5 dimensiones: se resta cada par de variables, se eleva al cuadrado, se suma todo y se saca raíz cuadrada — el mismo procedimiento que con 2 variables, solo que con más términos dentro de la raíz. El resultado (5.29) por sí solo **no tiene ninguna interpretación**; la interpretación surge únicamente después de calcular la distancia entre **todos los pares** de puntos y organizarla en una matriz de distancias.
- La matriz resultante es **simétrica**: la distancia de $x_1$ a $x_2$ es idéntica a la de $x_2$ a $x_1$, por lo que en la práctica se descarta el triángulo superior de la matriz —es información repetida— y solo se conserva el triángulo inferior.
- Una vez armada la matriz completa, identificar el valor más pequeño indica cuáles dos puntos son los primeros candidatos a agruparse (en el ejemplo, $x_3$ y $x_5$ son los más cercanos entre sí).
- **Aclaración sobre dónde se calcula:** la clusterización debe hacerse siempre sobre el dataset original (con todas sus dimensiones), aunque el dataset sea grande y el cálculo de todas las distancias por pares implique más costo computacional; la representación visual de los clusters resultantes es lo que se apoya en PCA (ver 1.8).

**Propiedades de toda matriz de distancias válida:**
- Ninguna distancia es negativa: $d_{ij} \geq 0$.
- La distancia de un punto a sí mismo es cero: $d_{ii} = 0$.
- Es simétrica: $d_{ij} = d_{ji}$.
- Cumple la **desigualdad triangular**: ir directo de un punto A a un punto C siempre es igual o más corto que pasar por un punto intermedio B. Ejemplo cotidiano: si una persona debe ir de su casa (A) a la universidad (C) pero antes pasa por una tienda (B), la ruta directa A→C nunca es más larga que A→B→C.

---

## 3. Distancias

### 3.1 Euclidiana

La medida más común; representa la longitud de la **línea recta** entre dos observaciones.

$$d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}$$

### 3.2 Manhattan (o *city-block*)

Suma las diferencias absolutas en cada dimensión, simulando un desplazamiento solo horizontal y vertical —como moverse por las calles en cuadrícula de la ciudad de Manhattan— en vez de una línea recta.

$$d(A,B) = \sum_{i=1}^{n} |x_i - y_i|$$

**Ejemplo:** $A=(2,3)$, $B=(7,8)$ → $|7-2| + |8-3| = 5 + 5 = 10$, frente a una distancia euclidiana de $\sqrt{(7-2)^2+(8-3)^2} = \sqrt{50} \approx 7.07$ para el mismo par de puntos — la euclidiana no siempre es menor que la Manhattan, pero en este caso concreto sí lo es.

> No existe una respuesta única sobre cuándo conviene usar Manhattan en vez de euclidiana: la distancia euclidiana no siempre produce el mejor agrupamiento. Esa comparación se resuelve en el tema de **evaluación de clustering** (pendiente), con métricas como la **silueta**, que permiten determinar cuál distancia forma los clusters más coherentes para un dataset específico — el criterio práctico es "la distancia que mejor agrupe a los individuos", no una preferencia fija por una función de distancia.

### 3.3 Chebyshev (distancia del máximo)

Considera únicamente la **mayor diferencia absoluta** entre las variables comparadas; ignora por completo las demás diferencias.

$$d(x,y) = \max_i |x_i - y_i|$$

**Ejemplo:** con diferencias de Edad = 5, Ingreso = 15 y Deuda = 8 entre dos individuos, la distancia Chebyshev es $\max(5,15,8) = 15$ — se pierde la información de las otras dos variables, pero es el comportamiento propio del método.

### 3.4 Otras distancias (mención)

Coseno (ángulo entre vectores), Jaccard (para variables cualitativas/categóricas — funciona muy bien en ese caso), Minkowski (generalización de Manhattan/Euclidiana/Chebyshev según el parámetro $p$), Hamming, entre otras.

---

## 4. Clustering jerárquico *(contenido del PDF — no desarrollado oralmente en esta sesión)*

### 4.1 Definición y dendrograma

El **clustering jerárquico** organiza los datos en una estructura de árbol (**dendrograma**) basada en su similitud. Existen dos enfoques:

| Enfoque | Dirección |
|---|---|
| **Aglomerativo** | *Bottom-up*: cada punto empieza como su propio cluster y se van fusionando los más cercanos hasta llegar a un único cluster. |
| **Divisivo** | *Top-down*: se empieza con un único cluster que contiene todos los datos y se va dividiendo sucesivamente. |

El dendrograma resultante permite leer, en el eje de distancia, en qué momento (a qué altura de disimilitud) se fusionan los clusters: en la base cada punto es su propio cluster; conforme aumenta la altura se forman clusters más grandes, hasta llegar a un único cluster que agrupa todos los datos.

### 4.2 Algoritmo aglomerativo

Sea $I = \{1,\dots,n\}$ el conjunto de objetos:

1. **Inicialización:** cada punto es su propio cluster, $P_0 = \{\{1\},\{2\},\dots,\{n\}\}$, $h=0$.
2. **Formación de nuevos nodos:** se fusionan los dos clusters de $P_h$ más cercanos según el criterio de agregación $\delta$ (criterio de enlace): $\delta(x,y) = \min\{\delta(l,k) \mid l,k \in P_h,\, l \neq k\}$.
3. **Actualización:** $h \leftarrow h+1$; se reemplazan los dos clusters fusionados por el nuevo cluster combinado.
4. **Criterio de parada:** si $h < n-2$ se regresa al paso 2; en otro caso se hace la última fusión y termina.

### 4.3 Ejemplo completo de cálculo (5 puntos en $\mathbb{R}^2$)

| Punto | $x_1$ | $x_2$ |
|---|---:|---:|
| A | 1 | 2 |
| B | 2 | 1 |
| C | 4 | 1 |
| D | 5 | 4 |
| E | 7 | 3 |

Matriz de distancias inicial (euclídea):

| | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| A | 0 | 1.41 | 3.16 | 5.00 | 6.32 |
| B | 1.41 | 0 | 2.00 | 4.24 | 5.39 |
| C | 3.16 | 2.00 | 0 | 3.16 | 3.61 |
| D | 5.00 | 4.24 | 3.16 | 0 | 2.24 |
| E | 6.32 | 5.39 | 3.61 | 2.24 | 0 |

**Fusiones sucesivas (con enlace promedio):**

1. Mínima distancia $d(A,B)=1.41$ → se fusionan $A$ y $B \to (AB)$.
2. Mínima distancia $d(D,E)=2.24$ → se fusionan $D$ y $E \to (DE)$.
3. Distancia $(AB,C) = \frac{1}{2}(d(A,C)+d(B,C)) = \frac{1}{2}(3.16+2.00) = 2.55$ → se fusionan $(AB)$ y $C \to (ABC)$.
4. Última fusión: $d(ABC,DE) = 4.44$ (promedio de todas las distancias cruzadas entre $\{A,B,C\}$ y $\{D,E\}$) → se obtiene un único cluster.

El dendrograma final muestra las alturas de fusión: $A$–$B$ a 1.41, $D$–$E$ a 2.24, $(AB)$–$C$ a 2.55, y la fusión total a 4.44.

### 4.4 Criterios de enlace (linkage)

En cada fusión hay que decidir cómo medir la distancia entre dos *clusters* (no entre dos puntos individuales):

| Criterio | Fórmula | Comportamiento |
|---|---|---|
| **Salto mínimo** (*single linkage*) | $d_{\min}(A,B) = \min_{x\in A,\,y\in B} d(x,y)$ | Usa el par más cercano entre ambos clusters. Tiende a formar clusters alargados o en cadena (efecto de encadenamiento). |
| **Salto máximo** (*complete linkage*) | $d_{\max}(A,B) = \max_{x\in A,\,y\in B} d(x,y)$ | Usa el par más lejano. Tiende a formar clusters compactos y de tamaño similar, pues exige que todos los puntos estén relativamente cerca. |
| **Promedio** (*average linkage*) | $d_{\text{prom}}(A,B) = \frac{1}{|A||B|}\sum_{x\in A}\sum_{y\in B} d(x,y)$ | Promedia todas las distancias cruzadas. Es un compromiso: evita cadenas largas pero permite formas y tamaños algo variables. |
| **Ward** (mínima varianza) | $d_{\text{Ward}}(A,B) = \frac{n_A n_B}{n_A+n_B}\lVert \bar{x}_A - \bar{x}_B\rVert^2$ | Fusiona los clusters que produzcan el menor incremento en la varianza intra-cluster. Tiende a formar clusters esféricos y de tamaño similar; es el criterio **más usado en la práctica**. |

No existe un criterio universalmente mejor — depende de la estructura de los datos y del objetivo del análisis. Con un mismo conjunto de 4 puntos ($X1=(0,0)$, $X2=(5,0)$, $X3=(3,0)$, $X4=(2,1)$), los cuatro métodos producen alturas finales de fusión distintas: salto mínimo → 2; salto máximo → 5; promedio → 3.12; Ward → 4.80 — la elección del criterio de enlace cambia tanto la forma como el número final de clusters obtenidos al cortar el dendrograma a una altura dada.

### 4.5 Cómo se construye una matriz de distancia en la práctica

Con $n$ observaciones (p. ej. estudiantes) y $p$ variables (p. ej. notas en $p$ asignaturas), cada observación es un punto en $\mathbb{R}^p$. La distancia euclídea entre dos observaciones $i$ y $j$ es:

$$d_{ij} = \sqrt{\sum_{k=1}^{p}(x_{ik}-x_{jk})^2}$$

Se calcula una sola vez por cada par $(i,j)$ con $i<j$ — el resto se obtiene por simetría ($d_{ij}=d_{ji}$, $d_{ii}=0$) — y se organiza en una matriz de distancia completa.

### 4.6 Gráficos para la interpretación de clusters

Una vez obtenido el agrupamiento (por ejemplo, vía clustering jerárquico), estos gráficos ayudan a caracterizar cada cluster:

| Gráfico | Uso |
|---|---|
| **Scatter plot coloreado por cluster** | Proyección en 2D (p. ej. con PCA) mostrando la separación visual entre clusters. |
| **Boxplots por cluster** | Distribución de cada variable dentro de cada cluster; identifica qué variables diferencian los grupos. |
| **Barplot de medias por cluster** | Promedio de cada variable por cluster — resumen simple del comportamiento de cada grupo. |
| **Heatmap de medias** | Mapa de calor de las medias estandarizadas por cluster y variable (colores cálidos = valores altos, fríos = valores bajos). |
| **Radar chart** | Perfil promedio de cada cluster en todas las variables escaladas, superpuesto en un mismo gráfico. |
| **Violin plots** | Distribución y densidad de cada variable dentro de cada cluster. |
| **Silhouette plot** | Mide qué tan bien asignado está cada punto a su cluster; valores cercanos a 1 indican buena asignación. |
| **Tamaño de clusters** | Cantidad de observaciones por cluster — útil para detectar grupos muy pequeños o desbalanceados. |
| **Parallel coordinates plot** | Cada línea es una observación; permite ver patrones multivariados y diferencias entre clusters. |

---

## Conceptos clave de la clase

- **Clustering** es una técnica exploratoria de aprendizaje no supervisado que agrupa observaciones similares y separa las diferentes; depende directamente de cómo se defina la similitud (distancia) entre observaciones.
- El **centroide** es el promedio de todas las observaciones de un cluster; sirve para resumir un grupo con un solo valor representativo por variable, pero se desplaza (puede volverse poco representativo) cuando se incorporan puntos atípicos.
- Antes de clusterizar hay que verificar **si los datos son realmente clusterizables** (Índice de Hopkins — desarrollo pendiente) y resolver tres problemas comunes: **escalas distintas** (dominan la distancia si no se estandariza), **sensibilidad a outliers** (distorsionan y desplazan centroides) y **alta dimensionalidad** (maldición de la dimensionalidad).
- **Recomendación práctica:** clusterizar sobre los datos originales, no sobre componentes de PCA — el PCA es una combinación lineal y sus centroides quedan en coordenadas artificiales, difíciles de interpretar frente a un tomador de decisiones. El PCA se reserva para visualizar los clusters ya formados.
- Las distancias más usadas son **Euclidiana** (línea recta), **Manhattan** (recorrido en cuadrícula) y **Chebyshev** (solo la diferencia máxima); no existe una distancia universalmente mejor — se elige la que mejor agrupe los datos de un problema específico (tema de evaluación de clustering, pendiente).
- El **clustering jerárquico** (contenido del PDF, no desarrollado oralmente) construye un **dendrograma** fusionando iterativamente los clusters más cercanos según un criterio de enlace (*linkage*): **salto mínimo**, **salto máximo**, **promedio** o **Ward** (el más usado en la práctica). Cada criterio produce dendrogramas y agrupaciones distintas sobre los mismos datos.
- Gráficos como boxplots, heatmaps, radar charts y silhouette plots son las herramientas estándar para interpretar y caracterizar los clusters obtenidos, complementando la visualización simple en 2D.

---

## Fuera del PDF — logística, quiz y metodología

- **Quiz 2** al inicio de la clase: 12 preguntas de repaso sobre **Análisis de Correspondencia Múltiple (ACM)**, **t-SNE** y **UMAP**, evaluado por velocidad de respuesta y por corrección — formato competitivo entre estudiantes.
- Aclaraciones surgidas durante la retroalimentación del quiz (no repetidas si ya están en resúmenes previos):
  - ACS y ACM son modelos **lineales**; t-SNE y UMAP no lo son, por lo que no existe relación de "extensión" entre ambos pares de métodos. El ACM es la extensión correcta del ACS (para manejar múltiples variables categóricas, no solo la matriz disyuntiva binaria de dos códigos que exige el ACS).
  - En un mapa factorial de ACM, dos categorías cercanas están asociadas frecuentemente; dos categorías alejadas presentan comportamientos diferentes (poca o nula asociación) — no pertenecen nunca a una misma dimensión, ya que cada categoría se lee en una dimensión distinta.
  - No hay un umbral fijo de probabilidad para decidir si dos puntos pertenecen al mismo cluster en UMAP: conviene comparar la probabilidad de conexión más alta contra las demás probabilidades de conexión de otros grupos antes de decidir.
  - Una "conexión fuerte" en UMAP indica vecinos cercanos en el **espacio original**, pero no necesariamente implica que pertenezcan a la misma **clase** (concepto de aprendizaje supervisado) — sí puede indicar que pertenecen al mismo **cluster**, y UMAP puede usarse en un esquema semi-supervisado.
- **No hay tarea esta semana**; la próxima entrega inicia hasta la semana siguiente.
- Encuesta de retroalimentación de medio ciclo: valorada positivamente por los estudiantes, especialmente el formato de quiz con retroalimentación inmediata.
- Sesión de tutoría opcional el viernes de 20:30 a 21:30/22:00 h, para repasar antes de continuar con Hopkins, VAT y clustering jerárquico la siguiente semana.
- Próxima clase: se retoma justo donde quedó pendiente — Índice de Hopkins, VAT, y el desarrollo completo de clustering jerárquico (dendrograma, algoritmo, linkage) con ejemplos prácticos.
