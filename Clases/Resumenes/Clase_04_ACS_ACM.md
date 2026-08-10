# Clase 4 — Análisis de Correspondencia Simple y Múltiple (ACS / ACM)

**Curso:** Minería de Datos · Lead University
**Docente:** Rachit Vargas (MBA | Engineer)
**PDF fuente:** `Clases/PDFs/Clase 4 - ACS ACM.pdf`

> Segundo bloque de algoritmos de reducción de dimensionalidad del curso. A diferencia del PCA, que trabaja sobre variables numéricas, el **Análisis de Correspondencia Simple (ACS)** y el **Análisis de Correspondencia Múltiple (ACM)** están diseñados para **variables cualitativas** representadas en tablas de contingencia. Comparten el mismo objetivo de fondo que el PCA —reducir dimensiones maximizando la información conservada— pero cambian la matemática de partida: en vez de covarianza, usan perfiles de frecuencia y distancia chi-cuadrado.

---

## 1. El problema de analizar asociaciones en datos categóricos

### 1.1 Tipos de datos y tabla de contingencia

Muchos estudios recogen información cualitativa en tablas de contingencia de $I$ individuos (filas) y $J$ categorías (columnas), donde $n_{ij}$ es la frecuencia del individuo $i$ en la categoría $j$.

**Ejemplo de clase.** Un dataset con columnas *ID*, *género* (hombre/mujer — 2 categorías) y *educación* (primaria, secundaria, bachillerato universitario, maestría, doctorado — 5 categorías). El género y la escolaridad son variables cualitativas: representan una clasificación o etiqueta, no un valor continuo como el salario. No tiene sentido decir que una persona tiene una escolaridad "3.5" entre bachillerato y maestría — a diferencia de una variable cuantitativa, que puede tomar cualquier valor continuo.

Para poder analizar la relación entre ambas variables se construye una **matriz (o tabla) de contingencia**: en Excel corresponde a lo mismo que resuelve una tabla dinámica (*pivot table*). En el ejemplo, la matriz cruza las 2 categorías de género contra las 5 de escolaridad y cuenta cuántas veces aparece cada combinación (p. ej., un hombre con escolaridad de bachillerato universitario suma un conteo en esa celda).

### 1.2 Qué queremos entender

Interesa estudiar si existen asociaciones entre las categorías y los individuos: qué categorías tienden a aparecer juntas en los mismos individuos, si hay grupos de individuos con perfiles similares, y qué categorías caracterizan a cada grupo. Por ejemplo, si las mujeres tienden a tener un nivel académico más alto que los hombres.

### 1.3 El problema central

Cuando $I$ y $J$ son grandes, la tabla vive en un espacio de dimensión $IJ$, lo que genera las mismas dificultades que motivaron el PCA:

- Es imposible visualizar directamente las asociaciones en ese espacio: solo se puede interpretar un gráfico sobre dos ejes (x, y), o como mucho tres dimensiones, y aun así de forma incómoda.
- Hay mucha información y redundancia — la reducción de dimensionalidad busca justamente quitar esa redundancia y las variables altamente correlacionadas que explican lo mismo.
- No es claro cómo identificar patrones, grupos u oposiciones entre filas y columnas sin un algoritmo dedicado: no existe otra forma directa de llegar a esa conclusión con un solo gráfico.

Con solo 2 categorías (hombre/mujer) la tabla es manejable, pero basta ampliar a una variable con 50 categorías, o agregar una tercera variable con algunas categorías adicionales, para pasar de un problema bidimensional a uno tridimensional o mayor — analizar eso simultáneamente se vuelve una tarea casi imposible con herramientas tradicionales (tablas dinámicas, dashboards con múltiples gráficos).

**En síntesis:** el problema es representar e interpretar las asociaciones entre filas y columnas de una tabla de contingencia de forma clara, intuitiva y en un espacio manejable — es decir, en $\mathbb{R}^2$.

### 1.4 Por qué es un problema importante

Los datos categóricos aparecen en muchos ámbitos: ciencias sociales (encuestas, estudios de opinión), marketing (preferencias, segmentación, satisfacción), biología (presencia/ausencia de especies, síntomas), educación (rendimiento por niveles, tipo de centro), entre otros.

El profesor Jorge Arce, quien anteriormente impartía este curso, utilizaba ACM combinado con análisis de datos simbólicos para analizar encuestas presidenciales, y con ese método detectó que Rodrigo Chaves sería el próximo presidente de Costa Rica antes de que las encuestas tradicionales (UCR y otras casas encuestadoras) lo reflejaran — un estudio con márgenes de error, pero más robusto que las encuestas convencionales.

---

## 2. Análisis de Correspondencia Simple (ACS)

### 2.1 Idea del modelo

El ACS busca representar filas (individuos) y columnas (categorías) en un **mapa conjunto de baja dimensión** de modo que las distancias reflejen sus asociaciones (desviaciones respecto a la independencia): **perfiles similares quedan cerca, perfiles distintos quedan lejos**.

- **Perfil de fila $i$:** cómo se distribuye ese individuo entre las categorías (columnas).
- **Perfil de columna $j$:** cómo se distribuyen los individuos en esa categoría.

El ACS funciona **solo con dos variables cualitativas** (una tabla de contingencia $I \times J$).

### 2.2 Métrica: distancia chi-cuadrado e inercia

La distancia entre dos perfiles de fila $i$ y $k$ se mide con la métrica **chi-cuadrado ($\chi^2$)**:

$$d^2(i,k) = \sum_{j=1}^{J} \frac{1}{c_j}(p_{ij} - p_{kj})^2$$

donde $p_{ij} = n_{ij}/r_i$ es el perfil de fila $i$, y $c_j$ es el total de la columna $j$ entre $n$.

La **inercia total** mide la variabilidad total respecto a la independencia:

$$I_{total} = \sum_{i=1}^{I}\sum_{j=1}^{J} \frac{(n_{ij} - r_ic_j/n)^2}{r_ic_j/n}$$

Cuanta más inercia explican las dimensiones elegidas, mejor la representación. **Objetivo del ACS:** encontrar ejes (direcciones) que expliquen la mayor inercia posible y proyectar filas y columnas sobre ellos — el primer vector maximiza la inercia explicada, y el segundo se traza de forma ortogonal al primero para eliminar la correlación con él, exactamente igual a la lógica de los eigenvectores en PCA.

### 2.3 Algoritmo paso a paso

| Paso | Qué hace |
|---|---|
| **1. Preparar datos** | Calcular totales de fila $r_i$, de columna $c_j$ y total $n$. Calcular proporciones $P = N/n$. Calcular perfiles de fila $p_{ij} = n_{ij}/r_i$ y de columna $p_{ej}^{(col)} = n_{ij}/c_j$. |
| **2. Construir la matriz de residuos tipificados** | Comparar lo observado con lo esperado bajo independencia ($r_ic_j/n$): $S = D_r^{-1/2}(P - rc^T)D_c^{-1/2}$. |
| **3. Descomponer (SVD)** | Aplicar descomposición en valores singulares: $S = U\Sigma V^T$, donde $U$ y $V$ son los vectores (direcciones) de filas y columnas, y $\Sigma$ los valores singulares que indican cuánta inercia explica cada dimensión. |
| **4. Elegir dimensión $K$** | Normalmente 2 o 3, las que expliquen la mayor parte de la inercia (típicamente 60–90 %). |
| **5. Calcular coordenadas** | Filas: $F_K = D_r^{-1/2}U_K\Sigma_K$. Columnas: $G_K = D_c^{-1/2}V_K\Sigma_K$. |
| **6. Representar e interpretar** | Dibujar filas (puntos azules) y columnas (puntos verdes) en el plano de las dos primeras dimensiones. |

**Lectura del mapa final:**

| Elemento | Interpretación |
|---|---|
| Proximidad fila–columna | Si un individuo está cerca de una categoría, tiene una probabilidad relativa alta de esa categoría. |
| Proximidad entre filas | Filas cercanas tienen perfiles de categorías similares. |
| Proximidad entre columnas | Columnas cercanas atraen a los mismos tipos de individuos. |
| Lados opuestos | Puntos en lados opuestos del origen tienden a estar asociados negativamente (perfiles opuestos). |
| Cerca del origen | Perfiles cercanos al promedio (poca información específica, comportamiento de independencia). |

> Este es exactamente el mismo mecanismo del PCA: en vez de sacar una matriz de covarianza (algo numérico), el ACS calcula probabilidades y frecuencias; sobre esas frecuencias calcula una distancia (chi-cuadrado en vez de correlación) y luego proyecta los resultados en un espacio de menor dimensión a partir de los vectores que maximizan la varianza (inercia) explicada y eliminan la redundancia.

### 2.4 Ejemplo de clase — género × preferencia de producto

Tabla de contingencia 2×3: género (hombre/mujer) contra preferencia de producto (A, B, C). Los hombres prefieren más el producto A sobre B y C; las mujeres prefieren más el producto B sobre A y C.

**Perfiles de fila** (frecuencia relativa del individuo dentro de su propia fila, deben sumar 100 %): para los hombres, el producto A pesa 62 %, el B 23 % y el C 15 %.

**Perfiles de columna** (frecuencia relativa dentro de cada columna, de forma vertical en vez de horizontal): sobre el producto A, los hombres representan un 67 % y las mujeres un 33 %.

**Distancia chi-cuadrado.** Con la fórmula $d^2(i,k) = \sum_j \frac{1}{c_j}(p_{ij}-p_{kj})^2$: se resta el perfil de fila de un individuo contra el de otro para la misma columna, se eleva al cuadrado, se divide entre el total de esa columna, y se suma sobre todas las columnas — el resultado (antes de la raíz) no tiene interpretación directa por sí solo. Al sumar sobre las tres categorías y sacar la raíz cuadrada se obtiene la distancia real entre hombres y mujeres (en el ejemplo, 0.74).

Con un ejemplo extendido a 4 categorías (hombre, mujer, joven, adulto) se obtiene una **matriz de distancias completa**: entre más corta la distancia, más parecidos son los perfiles de esos dos individuos (y viceversa). En el ejemplo trabajado en clase: hombre–joven tiene la distancia más corta (preferencias de producto muy similares, compran los mismos productos con frecuencia similar) y hombre–adulto la más larga (preferencias bastante distintas). Las mujeres se parecen más a los jóvenes que a los adultos.

El mismo cálculo de distancias se repite también para las columnas (A vs. B, A vs. C, B vs. C), para saber qué categorías están cerca o lejos entre sí.

**Biplot resultante.** Con solo 2 variables (producto y género) el resultado es un ACS. En el ejemplo, el hombre se asocia fuertemente al producto A; la mujer queda en un resultado contrario, entre los productos B y C, sin relación directa con A.

### 2.5 Gráficos que pueden obtenerse en un ACS

| Gráfico | Qué muestra |
|---|---|
| **Mapa factorial conjunto (biplot)** | Filas y columnas en el mismo plano — el más importante, el de mayor interés práctico. |
| **Mapa de filas** | Solo los individuos. |
| **Mapa de columnas** | Solo las categorías. |
| **Scree plot** | Porcentaje de inercia explicado por cada dimensión. |
| **Inercia acumulada** | Inercia acumulada al incluir más dimensiones — es lo mismo que el scree plot, presentado de forma acumulativa (tipo Pareto). |
| **Contribuciones de filas / columnas** | Qué filas o columnas contribuyen más a cada dimensión. |
| **Cos² (calidad de representación)** | Igual que en PCA: debe superar el umbral práctico de **60 %** para poder afirmar algo sobre ese individuo o categoría. |
| **Heatmap de residuos χ²** | Desviaciones respecto a la independencia (azul = menor a lo esperado, rojo = mayor a lo esperado). |
| **Perfiles (distribuciones)** | Gráfico de barras apiladas con la distribución de cada fila entre las categorías — es un gráfico descriptivo, no un resultado del modelo. |
| **Coordenadas factoriales** | Tabla numérica de coordenadas de filas y columnas en cada dimensión. |
| **Mapa factorial 3D** | Igual que el biplot pero con una tercera dimensión — agrega información, pero es más difícil de interpretar visualmente que uno en 2D. |
| **Cluster sobre mapa factorial** | Agrupamiento de filas sobre el mapa — el ACS/ACM también sirve para clusterizar: la clusterización se apoya sobre la distancia chi-cuadrado, no sobre el algoritmo en sí, y se profundiza en clases posteriores del curso. |
| **Trayectorias factoriales** | Evolución de puntos a lo largo del tiempo. |
| **Gráfico de masas** | Tamaño de los puntos proporcional a su masa (peso). |

**Sobre el heatmap de residuos y el cos² en la práctica:** el criterio de cos² funciona igual que en PCA — mide si un individuo o columna supera el punto de corte (60 %) y por tanto está bien representado; si un individuo cae cerca del centro del mapa pero su cos² es alto, no significa necesariamente pérdida de información, sino que puede representar un perfil promedio (no se caracteriza fuertemente por ninguna categoría en particular). Si en cambio el cos² es bajo, el modelo lo ubicó ahí por necesidad de graficarlo, pero esa posición no es representativa — la contribución no tiene un criterio de corte tan definido como el cos²: entre mayor, mejor.

> Sobre el heatmap de residuos ajustados en específico: el docente reconoció no tener una interpretación clara ni experiencia aplicándolo en la práctica, y lo dejó fuera del alcance central de la clase.

### 2.6 Ventajas y desventajas del ACS

| Ventajas | Desventajas |
|---|---|
| **Representación visual intuitiva:** ofrece un mapa de pocas dimensiones que facilita comprender asociaciones complejas entre categorías e individuos. | **Sensibilidad al tamaño muestral:** con muestras pequeñas, el mapa puede ser inestable y las interpretaciones poco fiables. |
| **Captura asociaciones (no solo correlaciones):** se basa en desviaciones respecto a la independencia, por lo que detecta cualquier tipo de asociación. | **Dependencia de la escala de perfiles:** las distancias se basan en perfiles; categorías muy raras pueden tener un peso exagerado y sesgar el resultado. |
| **Detección de patrones y estructuras:** identifica grupos de individuos, categorías similares, oposiciones y perfiles característicos. | **Posible sobre-interpretación:** no todo lo que aparece cerca en el mapa es "importante" — debe apoyarse en inercia, contribuciones y cos². |
| **Adecuado para datos categóricos** (posiblemente la ventaja más relevante): diseñado específicamente para tablas de contingencia, sin supuestos fuertes sobre las distribuciones. | **Solo asociaciones, no causalidad:** el ACS describe patrones relacionales, pero no implica relaciones causales — igual que la mayoría de algoritmos de minería de datos. |
| **Flexibilidad interpretativa:** permite analizar tanto filas como columnas en el mismo espacio o por separado, y estudiar su contribución y calidad. | **Dificultad con muchas categorías raras:** categorías con frecuencias muy bajas generan ruido y pueden distorsionar la estructura. |
| **Simplicidad computacional:** el algoritmo es relativamente sencillo (basado en SVD) y está implementado en la mayoría de paquetes estadísticos — posiblemente también converge rápido. | **Dependencia del número de dimensiones elegidas:** la interpretación puede cambiar según las dimensiones conservadas; elegir $K$ inadecuado puede ocultar estructura. |

**Sobre "categorías muy raras" y peso exagerado:** si en una tabla de contingencia un solo individuo concentra el 100 % de su participación en una única categoría (sin variabilidad frente al resto), ese caso es estadísticamente raro y puede sesgar el resultado — el mismo riesgo que enfrenta cualquier algoritmo de machine learning ante outliers.

**Conclusión clave del ACS:** transforma una tabla de contingencia en un mapa factorial que resume las asociaciones entre filas y columnas. Se pueden interpretar: asociaciones positivas (filas y columnas cercanas), asociaciones negativas (puntos en lados opuestos), perfiles promedio o de independencia (puntos cerca del origen), grupos de individuos o categorías con perfiles similares (nubes de puntos juntas). **Regla práctica:** usar siempre el mapa conjunto junto con la inercia explicada, las contribuciones y el cos² para interpretar — nunca el mapa solo.

### 2.7 ¿Cuándo usar el ACS?

**Sí usarlo cuando:**
- Se tiene una tabla de contingencia de dos variables categóricas (filas y columnas).
- Se busca explorar asociaciones sin imponer un modelo previo.
- Se busca segmentar individuos según sus perfiles de categorías.
- Se necesita identificar qué categorías caracterizan a cada grupo.
- Los datos provienen de encuestas, marketing, ciencias sociales, biología, educación, etc.

**No es la mejor opción cuando:**
- Existen muchas variables categóricas (usar ACM en su lugar).
- Se quiere predecir una variable respuesta (usar modelos supervisados).
- Las frecuencias esperadas son extremadamente bajas en casi todas las celdas (revisar el diseño de datos).

---

## 3. Análisis de Correspondencia Múltiple (ACM)

### 3.1 Extensión del ACS

El ACM generaliza el ACS cuando hay **más de dos variables categóricas**. La idea principal: transforma un conjunto de variables categóricas en un conjunto de variables binarias (0/1) mediante **codificación disyuntiva completa**, y luego aplica un ACS a esa tabla indicador.

| | ACS | ACM |
|---|---|---|
| **Tipo de datos** | Dos variables categóricas (tabla de contingencia) | Tres o más variables categóricas (conjunto de variables) |
| **Entrada** | Tabla $I \times J$ | Tabla de datos $n \times p$ categóricas |
| **Método** | Perfiles de filas/columnas + SVD | Codificación disyuntiva + ACS |
| **Salida** | Mapa filas y columnas | Mapa individuos y modalidades |
| **Uso típico** | Estudiar asociación entre dos variables | Explorar estructura de datos categóricos multivariados |

**Ejemplos de uso del ACM:** segmentación de clientes (sexo, edad, ingresos, preferencias, zona…), análisis de encuestas con muchas preguntas categóricas, estudios de percepción o satisfacción, análisis exploratorio de bases de datos cualitativas grandes.

En el ejemplo de clase: al pasar de género y escolaridad (ACS, 2 variables) a agregar sexo, edad, estudio y zona simultáneamente (ACM, 4 variables), la tabla de contingencia crece de $I \times J$ a $n \times p$, con muchas más categorías involucradas — un problema considerablemente más complejo de visualizar sin el algoritmo.

### 3.2 Qué datos usa: tabla disyuntiva completa (dummies)

El ACM parte de una **tabla disyuntiva completa $Z$ (0/1)**: individuos (filas) × modalidades de todas las variables (columnas). Cada fila es un individuo; cada columna, una modalidad de alguna variable; el valor es 1 si el individuo posee esa modalidad, 0 en caso contrario.

Esta codificación es la técnica conocida como **dummies** (variables ficticias) o **one-hot encoding**: convierte una columna cualitativa (por ejemplo, género con valores "masculino"/"femenino") en columnas binarias separadas por cada categoría — necesario porque los modelos de machine learning solo pueden procesar datos numéricos, no texto.

**Ejemplo de clase.** Género con 4 filas: masculino, femenino, masculino, femenino. Se crean dos columnas, `G.masculino` y `G.femenino`. La primera fila (masculino) queda como `1, 0`; la segunda (femenino) como `0, 1`; y así sucesivamente.

**Sobre eliminar columnas redundantes.** En variables binarias (como género, con solo 2 categorías), puede bastar con una sola columna, porque la segunda se infiere de la primera (si no es masculino, es femenino) — eliminarla reduce el tamaño del dataset. Esta práctica ayuda mucho en análisis supervisado: si se aplicara PCA sobre género con ambas columnas, el vector de "femenino" apuntaría a 180° del de "masculino" (correlación 100 % negativa), por lo que mantener ambas columnas es información redundante ("le está metiendo memoria" al modelo).

> No es una regla universal — depende del contexto. Puede haber casos donde un algoritmo no supervisado converja mejor o encuentre un patrón más definido incluso con esa redundancia (más raro, pero se puede dar). La lógica de eliminar una columna por inferencia aplica claramente a variables con solo 2 categorías con correlación negativa perfecta entre ellas; con 3 o más categorías (por ejemplo, producto A, B y C) esa correlación negativa perfecta entre pares ya no se cumple necesariamente.

### 3.3 Métrica, inercia y algoritmo

Se usa la misma **distancia chi-cuadrado** entre perfiles de filas (individuos) y de columnas (modalidades):

$$d^2(i,i') = \sum_{j=1}^{p}\frac{1}{c_j}(z_{ij}-z_{i'j})^2$$

donde $c_j = \frac{1}{n}\sum_{i=1}^n z_{ij}$ es el perfil de la modalidad $j$. La inercia total es $I_{total} = \sum_{j=1}^p \frac{c_j(1-c_j)}{n}$.

**Objetivo del ACM:** encontrar ejes (dimensiones) que expliquen la mayor inercia posible — mismo objetivo que el ACS y el PCA.

| Paso | Descripción |
|---|---|
| **1. Preparar datos** | Construir la tabla disyuntiva $Z$ ($n \times p$) con 0/1; calcular perfiles de filas $r_i$ y de columnas $c_j$. |
| **2. Residuos tipificados** | Eliminar el efecto de los márgenes (perfiles): $S = D_r^{-1/2}(Z - rc^T)D_c^{-1/2}$. |
| **3. Descomposición (SVD)** | $S = U\Sigma V^T$. |
| **4. Elegir dimensión $K$** | Las 2 o 3 primeras dimensiones que explican la mayor inercia: $\lambda_k = \sigma_k^2 / \sum_l \sigma_l^2$. |
| **5. Coordenadas principales** | Individuos: $F = D_r^{-1/2}U_K\Sigma_K$. Modalidades: $G = D_c^{-1/2}V_K\Sigma_K$. |

La intuición del algoritmo es idéntica a la del ACS — perfiles de filas, perfiles de columnas, distancias, proyección — solo que la tabla de contingencia es disyuntiva (0/1, no conteos) y el perfilamiento es diferente.

**Qué se obtiene:** mapa conjunto de individuos y modalidades. Perfiles similares quedan cerca (asociación positiva); perfiles opuestos quedan lejos, en direcciones contrarias (asociación negativa).

### 3.4 Interpretación práctica en el biplot ACS/ACM

La lectura del biplot es equivalente a la del círculo de correlación del PCA, pero en vez de variables numéricas se grafican **individuos y categorías/modalidades**:

- Dos categorías muy cercanas entre sí (p. ej., categoría B y D) tienen una correlación fuerte — se comportan igual.
- Una categoría en dirección opuesta (p. ej., categoría F frente a B y D) tiene una relación contraria.
- Un individuo cercano a un grupo de categorías se caracteriza por tener una tasa alta en esas categorías, y por tener niveles bajos en las categorías del lado opuesto — no se puede afirmar nada sobre categorías casi ortogonales a su dirección (no hay evidencia suficiente).
- Un individuo cercano al origen puede significar dos cosas: pérdida de información al reducir la dimensionalidad (cos² bajo, no representativo), o bien que su perfil es simplemente promedio: no se caracteriza marcadamente por ninguna categoría en particular (cos² alto, sí representativo, pero sin preferencia dominante). Conviene siempre revisar el cos² antes de concluir cuál de los dos casos aplica.

---

## 4. PCA vs. ACS/ACM — cómo decidir cuál usar

El PCA reduce dimensionalidad sobre variables **numéricas**; el ACS/ACM está pensado para variables **cualitativas**. Sin embargo, el PCA también puede aplicarse sobre variables cualitativas transformándolas primero a dummies — y en la práctica, a veces el PCA da mejores resultados que el ACS/ACM incluso en datasets mayoritariamente cualitativos (datos de encuestas, por ejemplo), aunque en teoría el ACS/ACM "debería" rendir mejor porque trabaja sobre el tipo de dato que le corresponde. No hay una regla fija: la recomendación es siempre **probar las herramientas correctas sobre los problemas correctos** y comparar resultados, en vez de asumir de antemano cuál algoritmo ganará.

**Criterio práctico de comparación:** correr ambos algoritmos sobre el mismo dataset y comparar el scree plot (inercia/varianza explicada acumulada) de cada uno. Si el PCA explica, por ejemplo, un 43 % acumulado entre sus dos primeras componentes, y el ACM explica un 50 % acumulado entre sus dos primeras dimensiones, se elige el algoritmo que conserve mayor información — en ese caso, el ACM. Es decir: **conviene aplicar siempre ambos algoritmos** y comparar antes de decidir cuál usar para la interpretación final.

> Sobre el criterio de Kaiser (tema traído a la clase anterior): un eigenvalor mayor a 1 indica que la componente principal correspondiente sí aporta suficiente información como para justificar conservarla; un eigenvalor menor a 1 significa que, al reducir la dimensionalidad con esa componente, se pierde más información de la que la variable original aportaba — no tiene sentido conservarla. El docente investigó el tema a partir de un comentario de una estudiante y lo consideró bastante práctico, aunque no lo conocía de antemano; queda como criterio adicional a los ya vistos (varianza acumulada, tolerancia del negocio) para decidir cuántas componentes conservar.

---

## Conceptos clave de la clase

- El **ACS** funciona sobre tablas de contingencia de **exactamente dos** variables cualitativas; el **ACM** generaliza el mismo mecanismo a **tres o más** variables cualitativas mediante codificación disyuntiva (dummies).
- Ambos comparten el objetivo del PCA (reducir dimensionalidad maximizando información conservada) pero no son una extensión matemática del PCA: cambian covarianza por **perfiles de frecuencia** y **distancia chi-cuadrado ($\chi^2$)**.
- El algoritmo, en esencia: tabla de contingencia → perfiles de fila y columna (frecuencias/probabilidades) → distancia chi-cuadrado entre perfiles → SVD sobre residuos tipificados → proyección en un espacio de baja dimensión ($K=2$ o 3).
- La interpretación del mapa/biplot es equivalente a la del PCA: proximidad = asociación positiva, lados opuestos = asociación negativa, cercanía al origen = perfil promedio o pérdida de información (revisar cos² para distinguir cuál caso aplica).
- **Cos² (umbral 60 %)** y **contribución** se interpretan igual que en PCA, aplicados ahora a individuos y a categorías/modalidades.
- El ACS y ACM también se usan para **clusterización**, apoyándose en la distancia chi-cuadrado — se profundiza en clases posteriores.
- **Dummies / one-hot encoding:** transforma cada categoría de una variable cualitativa en una columna binaria (0/1); en variables de solo dos categorías, una de las columnas puede eliminarse porque se infiere de la otra, reduciendo el tamaño del dataset sin perder información.
- **No existe una regla fija para elegir entre PCA y ACS/ACM** ante datos cualitativos: la recomendación práctica es correr ambos y comparar la inercia/varianza acumulada de cada uno antes de decidir cuál usar para interpretar.
- El ACS/ACM describe **asociaciones, no causalidad** — igual que la mayoría de algoritmos de minería de datos no supervisados.

---

## Fuera del PDF — logística, quiz y metodología

### Quiz de la clase
Primer quiz del curso, aplicado con **Mentimeter** (herramienta similar a Kahoot), con aproximadamente 12 preguntas sobre PCA y ACS/ACM. A diferencia de Kahoot, el profesor desactivó la puntuación por velocidad de respuesta: solo importa la respuesta correcta, no cuán rápido se responde, porque el objetivo es verificar comprensión de la teoría, no premiar velocidad. El quiz se retrasó para la semana siguiente por falta de tiempo al final de la clase, e incluirá todo el contenido hasta PCA inclusive.

> *"Si lo llenan de forma honesta, chicos, si no los que aprendemos somos nosotros."*

### Dudas sobre la tarea de PCA
Uno de los datasets de la tarea (`cars`) no incluye una columna de precio explícita, contrario a lo que algunos estudiantes asumieron. La recomendación fue omitir cualquier pregunta de la tarea que dependa de relacionar variables con el precio si el dataset no lo contiene, o bien usar como sustituto otra variable numérica disponible (por ejemplo, una columna de *rating*/*ranking* en escala de 1 a 10), como hizo un estudiante. Cada dataset de la tarea requiere graficar cuatro gráficos: scree plot, plano principal, círculo de correlación y biplot.

### Tangente sobre aprendizaje supervisado (adelanto del siguiente curso)
A raíz de una pregunta sobre por qué unas veces un algoritmo da mejores resultados que otro (PCA vs. ACM), se abrió una discusión sobre el **hyperparameter tuning**: en análisis supervisado existen muchos algoritmos (KNN, árboles de decisión, gradient boosting, SVM, redes neuronales, entre otros) y no hay forma de saber de antemano cuál rendirá mejor sobre un dataset específico — hay que probarlos todos y competir sus resultados.

- **KNN:** clasifica un punto nuevo según la clase mayoritaria de sus vecinos más cercanos ("dime con quién andas y te diré quién eres"). Es sensible a valores atípicos (outliers).
- **Árbol de decisión:** construye reglas secuenciales (ramas) sobre las variables para llegar a una clasificación; sí es interpretable — se puede explicar exactamente por qué llegó a una conclusión.
- **Gradient boosting:** más avanzado, usa un método de *bagging* (combinar varios modelos/"cerebros" en vez de uno solo) para mejorar el resultado.
- **Support vector machine:** traza vectores de soporte que maximizan el margen entre categorías.

**Criterios para elegir un algoritmo:** (1) precisión (*accuracy*) — el que se equivoque menos; (2) explicabilidad — qué tan fácil es justificar la decisión del modelo ante un tomador de decisiones. Las redes neuronales son una "caja negra" difícil de explicar, a diferencia de un árbol de decisión o una regresión lineal (donde los coeficientes dan una interpretación directa). Cuál criterio priorizar depende del negocio: en Fintech/procesamiento de tarjetas, la velocidad de respuesta es crítica; en un caso más delicado como un diagnóstico médico o una decisión financiera ante una junta directiva, la explicabilidad puede pesar más que la pura precisión — a veces conviene entregar ambos tipos de modelo (uno preciso y otro explicable) en vez de elegir uno solo.

> Ejemplo real citado por el docente: el gerente financiero de una empresa (ex-Pozuelo) expresó que, aunque la predictividad de los modelos de machine learning era muy buena, no podía presentar ante una junta directiva una cifra basada en un método que él mismo no comprendía.

### Referencias mencionadas
- **William Aguilar**, egresado de la misma universidad, realizó una maestría en computación cuántica y hoy da charlas explicativas sobre esa teoría (sin proponer investigación original) en Panamá, Suiza y Argentina — ejemplo de nivel de profundidad matemática necesario para crear nuevos algoritmos desde cero.

### Sesión de repaso y pendientes
Sesión de repaso confirmada para el sábado siguiente a las 9:00 a.m., grabada y compartida por WhatsApp para quienes no puedan asistir en vivo. En esa sesión se mostrará el código práctico de ACM (pendiente de esta clase) y también el de PCA. El repaso es opcional, no obligatorio, y no cubre materia nueva.

### Asistencia
Recordatorio de marcar asistencia en el campus virtual (semana 4, sesión del 2 de junio); ausencias justificadas por correo se registran igualmente como presente, no como ausencia justificada.
