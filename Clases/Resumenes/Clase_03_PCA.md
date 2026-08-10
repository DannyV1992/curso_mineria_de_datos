# Clase 3 — Reducción de dimensionalidad: PCA

**Curso:** Minería de Datos · Lead University
**Docente:** Rachit Vargas (MBA | Engineer)
**PDF fuente:** `Clases/PDFs/Clase 3 - PCA.pdf`

> Primer algoritmo de reducción de dimensionalidad del curso: **Análisis de Componentes Principales (PCA)**. Cubre el problema de la alta dimensionalidad, la intuición geométrica detrás de eigenvectores y eigenvalores, el algoritmo paso a paso y la interpretación de los resultados (varianza explicada, plano principal, círculo de correlación, biplot, coseno cuadrado y contribución) sobre el dataset `decathlon2`.

---

## 1. El problema de la alta dimensionalidad

### 1.1 Cuándo ocurre

Ocurre cuando un conjunto de datos tiene un número muy grande de variables (dimensiones). Hoy en día ya no es común encontrar datasets sencillos con dos o tres columnas —como precio de una casa y metros cuadrados—; la mayoría de datasets reales combinan decenas de variables: metros cuadrados, cantidad de habitaciones, si tiene piscina, tipo de vecindad, entre otras.

**Ejemplos de datos con muchísimas dimensiones:**
- Imágenes con miles de píxeles. Una foto en blanco y negro ya tiene un rango de 0 a 255 por píxel, es decir, cientos de variables solo para describir una imagen. Si la imagen es a color, esas columnas se multiplican por 3 (un canal por color).
- Datasets con múltiples mediciones: dictámenes médicos, mediciones deportivas.
- Sensores IoT con cientos de dimensiones — el auge del IoT y de la ciberseguridad ha disparado la cantidad de variables que hoy se recolectan por negocio.

### 1.2 Por qué visualizar es el problema central

El ojo humano solo puede interpretar hasta **3 dimensiones**, y ya un gráfico 3D es incómodo de leer — a la mayoría de las personas, especialmente gerentes que están acostumbrados a tablas y gráficos sencillos, no les gusta un gráfico de tercera dimensión. Con una matriz $X \in \mathbb{R}^{n \times m}$ de muchas columnas, es imposible representar visualmente el problema completo.

Un dashboard con varios gráficos (líneas, pastel, barras) puede visualizar varias dimensiones a la vez, pero cada gráfico solo cruza dos o tres variables entre sí. Para un problema de 20 dimensiones esto se vuelve inviable: se necesitarían decenas de gráficos y aun así no se vería el patrón conjunto de los datos. **El PCA no visualiza 100 dimensiones a la vez**; lo que hace es decidir, de esas 100 dimensiones, cuáles son las más importantes, y comprimirlas en dos o tres para poder graficarlas.

Los tests estadísticos vistos en la Clase 2 (ANOVA, matriz de correlación) solo permiten análisis **univariados o bivariados** — una variable, o como mucho dos, comparadas entre sí. El PCA es la herramienta que permite un análisis **multivariado**.

### 1.3 A mayor dimensión, los puntos están más lejos entre sí

**Ejemplo ilustrativo.** Dos personas con la misma edad (26 años) se ubican en el mismo punto sobre el eje *edad*: no hay variabilidad entre ellas. Al agregar la variable *vecindad* (una vive en Desamparados, la otra en Tibás), los puntos empiezan a separarse aunque comparten la misma edad. Cada variable nueva que se agrega —vecindad, gustos musicales, ingresos, profesión, países visitados— empuja los puntos cada vez más lejos entre sí, aunque en el fondo se esté describiendo a las mismas personas.

Con 100 dimensiones el espacio se vuelve tan disperso que prácticamente todos los puntos terminan alejados entre sí — el espacio se vuelve "vacío": en 2D está lleno, en 10D empieza a dispersarse y en 100D queda casi vacío.

### 1.4 Los tres problemas de la alta dimensionalidad

| Problema | Explicación |
|----------|-------------|
| **Complejidad computacional** | Entre más variables se le metan a un algoritmo, más pesado computacionalmente resulta: no es lo mismo procesar 2 variables que 60. |
| **Riesgo de sobreajuste (overfitting) en modelos supervisados** | Con demasiada variabilidad en los datos, el modelo aprende el ruido en lugar del patrón real. Es como un estudiante que memoriza las preguntas exactas de un examen en vez de entender el tema: si le repiten el mismo examen saca 100, pero es incapaz de resolver un examen nuevo. |
| **Dificultad para interpretar y visualizar** | Es el foco principal que ataca el PCA: es fundamentalmente una técnica de **visualización**. |

> El PCA no resuelve los tres problemas por completo, pero en su mayoría ayuda con cada uno de ellos.

---

## 2. El problema que resuelve el PCA

Dado un dataset $X \in \mathbb{R}^{n \times m}$ (n filas/individuos, m columnas/variables), el PCA sintetiza la información de $X$ en un conjunto más pequeño de variables sintéticas llamadas **componentes principales** ($C^1, C^2, \dots$), manteniendo la información esencial de $X$.

En la etapa 1 se encuentra la primera componente principal $C^1$, una **combinación lineal** de todas las variables originales $X^j$:

$$C^1 = a_{11}X^1 + \cdots + a_{1j}X^j + \cdots + a_{1m}X^m$$

Para el individuo $i$-ésimo:

$$C^1_i = a_{11}x_{i1} + \cdots + a_{1j}x_{ij} + \cdots + a_{1m}x_{im}$$

Generalmente $C^1$ solo no es suficiente para condensar toda la información contenida en $X$, por lo que se construye una segunda componente $C^2$, luego $C^3$, y así sucesivamente. En la etapa $k$:

$$C^k = a_{k1}X^1 + \cdots + a_{kj}X^j + \cdots + a_{km}X^m$$

Cada $a$ es un **eigenvalor**, resultado de la computación de la matriz de covarianza/correlación mediante álgebra lineal. La cantidad total de componentes principales que se pueden construir es igual a la cantidad de columnas del dataset original: si hay 20 variables, se pueden calcular 20 componentes principales.

**Reducir siempre implica un costo: pérdida de información.** Si de 20 variables se pasa a 2 componentes, se puede perder, por ejemplo, un 40 % de la información — las dos primeras componentes explicarían entonces el 60 % de las 20 variables originales.

> No existe un criterio universal de "cuánta información hay que conservar". Depende de la tolerancia del análisis y del negocio: si se define un umbral de conservar el 80 % de la información, puede que de 20 variables solo se logren eliminar 2 — la ganancia real depende del dataset.

### Multicolinealidad y redundancia — el ejemplo de las monedas

Explicar las ventas en dólares, en colones o en pesos colombianos es, en el fondo, explicar exactamente lo mismo: la correlación entre esas tres variables sería 1 (100 %), porque lo único que cambia es la moneda. Tener las tres dentro del dataset es **información redundante**. El PCA detecta esto: si tres variables dicen lo mismo, con una sola componente puede capturar el 100 % de esa información, porque agarrar una de ellas ya es suficiente.

Lo mismo ocurre con variables como población, cantidad de hombres y cantidad de mujeres: si $H + M = $ Población, las tres explican prácticamente lo mismo sobre las ventas finales. Pasar de 3 dimensiones a 1 puede conservar, por ejemplo, el 90 % de la información, precisamente porque esas variables están altamente correlacionadas.

### Eigenvectores, varianza y ortogonalidad

El primer eigenvector ($V_1$) siempre se traza en la dirección donde los datos **maximizan la varianza** — la mayor cantidad de información posible. El segundo eigenvector ($V_2$) se traza de forma **ortogonal** (90°) a $V_1$: va a la dirección donde se elimina al 100 % la correlación con $V_1$, capturando información que $V_1$ no explicaba.

| Ángulo entre vectores | Correlación |
|------------------------|-------------|
| 90° (ortogonal) | 0 % — no hay correlación |
| 0° (misma dirección) | Correlación positiva fuerte (cercana al 100 %) |
| 180° (dirección contraria) | Correlación 100 % negativa |

Si dos vectores apuntan en direcciones casi iguales, básicamente repiten la misma información — es justo lo que el PCA busca evitar. Por eso el segundo componente se traza de forma ortogonal al primero: para que $V_1$ contenga una información específica y $V_2$ contenga otra completamente distinta, sin redundancia.

> **Caso particular importante.** Dos componentes ($V_2$ y $V_3$) pueden ser ambos ortogonales a $V_1$ (correlación 0 % con $V_1$) y, sin embargo, estar **correlacionados entre sí en un 100 % negativo**, si sus direcciones son opuestas entre ellos. Ejemplo: vender más natilla implica vender menos mantequilla (productos sustitutos) — esa correlación negativa sigue aportando información no redundante con $V_1$, distinta de una correlación positiva o nula.

El algoritmo no entiende el significado de las variables (no sabe qué son dólares o colones): solo detecta qué datos se repiten, proyecta los eigenvectores y elimina la correlación entre ellos. Si de 100 dimensiones, 95 dicen exactamente lo mismo, con 5 variables puede bastar para explicar el patrón — las otras 95 son irrelevantes para el problema.

### PCA es 100 % lineal

El PCA resuelve **exclusivamente problemas lineales**, porque su mecanismo es una combinación lineal de las variables originales. En problemas no lineales el modelo no funciona bien — para eso existen **t-SNE** y **UMAP**, que se ven en la próxima clase.

> Aplicar clustering o PCA sin verificar primero, con tests estadísticos, si el problema realmente admite ser reducido o agrupado es un error común en la práctica profesional: puede llevar a resultados que parecen válidos pero que, al implementarse, no tienen relación real entre los datos de cada grupo — con el consiguiente costo de tiempo y dinero al tener que rehacer el análisis.

### Requisito de variables numéricas

El PCA es un algoritmo matemático: solo funciona con **variables numéricas**. Una variable categórica binaria (por ejemplo, género codificado como 0/1) no distorsiona el modelo, porque deja de ser una categoría y pasa a ser un problema puramente numérico. Sin embargo, cuando el dataset es **muy cualitativo** y hay que codificar demasiadas variables, el PCA empieza a distorsionarse: no está diseñado para eso. Para esos casos existen el **Análisis de Correspondencias Simple** y el **Análisis de Correspondencias Múltiple**.

Elegir la herramienta correcta según las características del problema —numérico vs. categórico, lineal vs. no lineal— es, como en un taller, saber qué herramienta usar para cada trabajo; de lo contrario el resultado será un análisis erróneo aunque se haya aplicado un algoritmo de moda.

### El dilema de la proyección — ejemplo de la taza

Al reducir un objeto en 3D a una proyección en 2D, la elección de qué componentes usar determina si la proyección es representativa o no. Usando los componentes principales 1 y 2 (los que más maximizan la información) se puede lograr una proyección reconocible como una taza. Si en cambio se usan el componente 1 y el componente 5 —donde el 5 aporta poca información—, la proyección puede parecer más bien un sartén. Usando el componente 1 y el componente 20 (información casi nula), la proyección puede degenerar en solo una figura sin relación clara con el objeto original.

Esto se traduce a nivel de individuos: puede ser que el individuo $i=1$ se proyecte muy bien (90 % de representación) en $C^1$ y $C^2$, mientras que el individuo $i=2$, con las mismas dos componentes, se proyecte mal (20 %). Esta discrepancia se mide con el **coseno cuadrado** (ver sección 4).

> Es un trade-off real y sin salida perfecta: puede que otro individuo se proyecte mejor usando, por ejemplo, $C^4$ y $C^{10}$, pero eso implicaría proyectar mal a la mayoría de los demás individuos. El algoritmo siempre prioriza la combinación de componentes que representa bien a la **mayoría**, aunque eso implique sacrificar la representación de algunos casos particulares.

---

## 3. Algoritmo PCA

### 3.1 Flujo general (5 pasos)

| Paso | Qué hace | Por qué |
|------|----------|---------|
| **1. Estandarizar (centrar)** | Restar la media a cada variable. | El PCA es un algoritmo sensible a la escala: no es lo mismo computar dólares que colones. Sin estandarizar, la variable con mayor magnitud domina el resultado. |
| **2. Covarianza** | Calcular la matriz de covarianza $S$ ($p \times p$). | Interesa saber qué variables están correlacionadas y en qué dirección hay más variación conjunta. |
| **3. Autovalores y autovectores** | Obtener $\lambda_j$ y $v_j$; ordenar por $\lambda_j$. | Los autovectores indican las direcciones principales; el autovalor indica cuánta varianza (información) hay en esa dirección. |
| **4. Elegir k componentes** | Seleccionar las $k$ direcciones con mayor varianza. | Se ordenan las direcciones de mayor a menor información. |
| **5. Proyectar** | Obtener $Z = \tilde{X}V_k$ (datos en $k$ dimensiones). | Representa los datos en menos dimensiones. |

### 3.2 Ejemplo — por qué estandarizar es crítico

| | Edad (años) | Ingreso ($) |
|---|---:|---:|
| | 20 | 20 000 |
| | 30 | 50 000 |
| | 40 | 80 000 |

$Var(\text{Edad}) \approx 66.67$, mientras que $Var(\text{Ingreso}) = 600\,000\,000$. Como $Var(\text{Edad}) \ll Var(\text{Ingreso})$, el **ingreso domina la varianza** por su mayor escala. Al aplicar PCA sin estandarizar, el resultado quedaría **dominado por la variable ingreso**.

Al estandarizar con $X' = \dfrac{X - \mu}{\sigma}$, ambas variables quedan con media 0 y desviación estándar 1 ($Var(\text{Edad}') = Var(\text{Ingreso}') = 1$): **PCA deja de estar sesgado por la escala de las variables**. Es equivalente a dejar de comparar "manzanas con peras" (edad vs. ingreso) y pasar a comparar "peras con peras" — ambas variables en la misma escala.

Este paso de estandarización lo realiza el propio cálculo del PCA de forma automática; no se hace manualmente antes de correr el modelo.

### 3.3 Algoritmo detallado (10 pasos)

**Entrada:** la tabla de datos $X \in M_{n \times m}$. **Salida:** la matriz de componentes principales $C \in M_{n \times m}$.

| Paso | Descripción |
|------|-------------|
| 1 | Centrar y reducir (estandarizar) la tabla de datos $X$. |
| 2 | Calcular la matriz de correlaciones $R \in M_{m \times m}$: $R = \frac{1}{n}X^TX$, o bien calculando todas las correlaciones a pie. |
| 3 | Calcular los vectores y valores propios de $R$. |
| 4 | Ordenar de mayor a menor los valores propios. |
| 5 | Con $\lambda_1, \dots, \lambda_m$ ordenados y $v_1, \dots, v_m$ sus vectores propios, construir $V = [v_1 \mid v_2 \mid \cdots \mid v_m] \in M_{m \times m}$. |
| 6 | Calcular la matriz de componentes principales: $C = X \cdot V$. |
| 7 | Calcular la matriz de **calidades de los individuos** (cosenos cuadrados) $Q \in M_{n \times m}$: $Q_{ir} = \dfrac{(C_{i,r})^2}{\sum_{j=1}^{m}(X_{ij})^2}$. |
| 8 | Calcular la matriz de **coordenadas de las variables** $T \in M_{m \times m}$, con entradas $\sqrt{\lambda_r}\,v_{j,r}$. |
| 9 | Calcular la matriz de **calidades de las variables** (cosenos cuadrados) $S \in M_{m \times m}$, con entradas $\lambda_r(v_{j,r})^2$. |
| 10 | Calcular el vector de **inercias de los ejes** $I \in M_{1 \times m}$: $I = \left(100 \cdot \frac{\lambda_1}{m}, \dots, 100 \cdot \frac{\lambda_m}{m}\right)$. |

### 3.4 Eigenvectores y eigenvalores (definición formal)

Los eigenvectores $v$ son vectores que, al aplicarles la matriz de covarianza $\Sigma$, solo cambian de escala, no de dirección ($\Sigma v = \lambda v$). Forman un sistema de coordenadas nuevo, ortogonal e independiente, que **elimina las correlaciones entre variables**: al proyectar los datos sobre estos ejes, cada componente captura información única, sin redundancia.

Cada eigenvalor $\lambda$ mide cuánta varianza existe a lo largo de su eigenvector asociado — en PCA, esto cuantifica cuánta "información" contiene cada componente principal. Al ordenar los eigenvalores de mayor a menor se identifican las direcciones más importantes. El PCA conserva los eigenvectores con mayores eigenvalores porque concentran la mayor parte de la varianza total, permitiendo reducir dimensiones con mínima pérdida de información.

---

## 4. Elementos que arroja el PCA y su interpretación

El PCA entrega principalmente cuatro elementos que hay que interpretar, más dos adicionales:

1. **Varianza explicada por componente principal** — cuánta información acumula cada componente ($C^1$, $C^2$, etc.).
2. **Plano principal** — los individuos proyectados sobre dos componentes.
3. **Círculo de correlación** — las variables proyectadas.
4. **Biplot** — combinación (sobreposición) del plano principal y el círculo de correlación en un mismo gráfico, mostrando tanto individuos como variables mediante vectores.
5. **Contribución de cada variable** al PCA.
6. **Coseno cuadrado** de cada individuo (y de cada variable).

### 4.1 Coseno cuadrado ($\cos^2$)

Mide la **calidad de representación** de una variable (o individuo) en un eje o plano principal.

$$\cos^2_{jk} = \frac{\text{cor}(X_j, Z_k)^2}{\sum_l \text{cor}(X_j, Z_l)^2}$$

En PCA con datos estandarizados: $\cos^2_{jk} = \text{cor}(X_j, Z_k)^2$.

- $\cos^2_{jk} \approx 1$: la variable (o individuo) está **bien representada** en el componente $k$.
- $\cos^2_{jk} \approx 0$: está **mal representada**.
- Propiedad: para cada variable/individuo, $\sum_{k=1}^{p} \cos^2_{jk} = 1$.
- Interpretación geométrica: $\cos^2_{jk} = \cos^2(\theta_{jk})$, el cuadrado del coseno del ángulo entre la variable y el componente.

**Umbral práctico: 60 %.** Si un individuo está representado con un coseno cuadrado **acumulado** (sumando las dimensiones del plano, no cada dimensión por separado) por encima del 60 %, la representación es buena y se puede interpretar con confianza. Por debajo de ese umbral, no es recomendable afirmar nada sobre ese individuo — puede que su información se haya perdido al reducir la dimensionalidad.

> El coseno cuadrado es conceptualmente parecido al p-valor de la regresión lineal: si es bajo, la representación de ese individuo en el plano no es concluyente. No implica que el individuo sea "malo" en una variable; implica que **falta información** para poder afirmarlo, porque esa información se perdió en la reducción de dimensionalidad.

Cuando el coseno cuadrado es 1 (100 %) para un individuo y la contribución de una variable también es 1, hay **certeza**: al reducir la dimensión no se perdió ninguna información. Si en cambio solo se logró contener, por ejemplo, el 20 %, ya no se tiene una probabilidad, sino una **incertidumbre**, porque el 80 % restante de la información se perdió en el proceso.

### 4.2 Contribución de una variable

Mide cuánto aporta una variable a la construcción de un eje (componente) $k$:

$$\text{ctr}_{jk} = \frac{\text{cor}(X_j, Z_k)^2}{\lambda_k}$$

**Propiedad clave:** para cada componente $k$, $\sum_{j=1}^{p} \text{ctr}_{jk} = 1$ (las contribuciones de todas las variables a un componente suman 1, o 100 %).

- $\text{ctr}_{jk}$ grande $\Rightarrow$ la variable $j$ contribuye mucho a formar el eje $k$.
- $\text{ctr}_{jk}$ pequeña $\Rightarrow$ contribuye poco.
- Relación con el coseno cuadrado: $\text{ctr}_{jk} = \dfrac{\cos^2_{jk}}{\lambda_k}$ — la contribución ajusta el coseno cuadrado por la importancia del componente (su eigenvalor).

**Ejemplo numérico** (4 variables, 2 componentes; $\lambda_1 = 2.50$, $\lambda_2 = 1.30$):

| Variable | $\cos^2_{j1}$ | $\cos^2_{j2}$ | Suma |
|----------|---:|---:|---:|
| $X_1$ | 0.70 | 0.20 | 0.90 |
| $X_2$ | 0.60 | 0.10 | 0.70 |
| $X_3$ | 0.15 | 0.60 | 0.75 |
| $X_4$ | 0.05 | 0.75 | 0.80 |

| Variable | $\text{ctr}_{j1} = \cos^2_{j1}/2.50$ | $\text{ctr}_{j2} = \cos^2_{j2}/1.30$ |
|----------|---:|---:|
| $X_1$ | 0.28 | 0.15 |
| $X_2$ | 0.24 | 0.08 |
| $X_3$ | 0.06 | 0.46 |
| $X_4$ | 0.02 | 0.58 |

Verificación: $\sum_j \text{ctr}_{j1} = 1.00$ y $\sum_j \text{ctr}_{j2} = 1.00$.

> Sobre si existe un criterio formal como el **criterio de Kaiser** (mencionado en clase para decidir cuántos componentes conservar, con valores propios mayores a 1) para elegir la cantidad de componentes en PCA: el docente no lo confirmó en el momento y quedó pendiente de verificar. La postura general del curso es que esa decisión depende del negocio y de cuánta pérdida de información se está dispuesto a aceptar, más que de un criterio matemático único.

---

## 5. Demostración — dataset `decathlon2`

Datos de rendimiento de **41 atletas** en competiciones de decatlón (Olímpicos y Decastar). Cada fila es un atleta; cada columna, una prueba o resultado. **10 pruebas deportivas** (100 m, salto largo, lanzamiento de bala, salto alto, 400 m, 110 m con vallas, lanzamiento de disco, salto con pértiga, lanzamiento de jabalina, 1500 m), más 2 resultados agregados (puntos totales, rango final) y 1 variable de información adicional (tipo de competencia). En total, 13 variables — de las cuales 10 numéricas son las que entran al PCA.

### 5.1 Varianza explicada (scree plot)

| Componente | % de varianza explicada |
|---|---:|
| PC1 | ≈ 41 % |
| PC2 | ≈ 18 % |
| **Acumulado PC1 + PC2** | **≈ 59-60 %** |

Con solo las dos primeras componentes se explica más de la mitad de un dataset que originalmente tenía 10 dimensiones — un buen resultado. El gráfico de varianza acumulada (tipo Pareto) puede dar la impresión de un crecimiento exponencial por la diferencia de tamaño entre la primera y la segunda barra, pero en este caso el crecimiento acumulado es más bien lineal.

### 5.2 Plano principal — individuos

Con solo el componente 1 y el componente 2 ya es posible visualizar el patrón de los 41 atletas, que antes vivían en un espacio de 10 dimensiones imposible de graficar. Atletas ubicados muy cerca entre sí en el plano son muy parecidos en su perfil de rendimiento; atletas en direcciones opuestas del plano tienen perfiles opuestos: donde uno es bueno, el otro es malo, y viceversa.

### 5.3 Círculo de correlación — variables

El círculo de correlación grafica las **variables** (no los individuos) como vectores. La lectura es la misma que la de los eigenvectores:

| Relación entre vectores | Ejemplo en el dataset | Interpretación |
|---|---|---|
| Dirección similar (correlación alta positiva) | Lanzamiento de disco y lanzamiento de bala | Ambas disciplinas van de la mano: quien es bueno en una, tiende a ser bueno en la otra. |
| Dirección opuesta ~180° (correlación fuerte negativa) | Lanzamiento de disco vs. carrera de 100 m | Quien es bueno lanzando disco tiende a ser malo en la carrera de 100 m, y viceversa. |
| Ángulo intermedio (correlación moderada) | High jump y long jump | Están relacionadas, pero no de forma perfecta — la correlación es baja, alrededor de un 20 %. |
| Casi ortogonal (correlación cercana a 0 %) | Lanzamiento de jabalina y salto alto | Se relacionan poco: ser bueno o malo en una no permite inferir nada sobre la otra. |

### 5.4 Biplot — lectura combinada

El biplot fusiona el plano principal y el círculo de correlación: los individuos que están alineados en la dirección de una variable se caracterizan por tener un valor alto en esa variable.

**Ejemplos de interpretación sobre atletas concretos:**
- **Bernard** se caracteriza por saltar alto (*high jump*), pero es malo en la carrera de 400 m — no se puede afirmar que sea malo en 100 m, porque esa variable está casi ortogonal (correlación cercana a cero) respecto a la dirección de Bernard.
- **Martínez** es bueno en la carrera de 100 m, y malo en jabalina, salto largo, disco y bala.
- **Sebrle** es bueno en 1500 m, jabalina y, en menor medida, salto largo.
- **Clay** es bueno en jabalina, salto largo, bala y disco, pero malo en la carrera de 100 m (y probablemente en 110 m con vallas, variable relacionada). No se puede afirmar si es bueno o malo en 400 m o en salto alto: son variables donde no hay suficiente correlación (ni evidencia) para concluir algo — el atleta queda en un nivel intermedio, ni tan bueno ni tan malo, simplemente porque **no hay datos suficientes para respaldar la afirmación**, no porque se sepa que es "regular".

---

## 6. Demostración en código — dataset `country data`

Aplicación práctica sobre `country_data`: indicadores de desarrollo por país (**9 variables**: tasa de mortalidad infantil, exportaciones, índice de salud, importaciones, ingresos, inflación, expectativa de vida, tasa de fertilidad, ingreso per cápita), con el objetivo de segmentar países según su necesidad de ayuda humanitaria.

**Notas de implementación:**
- El código base del curso usa la librería **Prince** para PCA (preferida por el docente sobre Scikit-learn para este flujo de trabajo).
- Es indispensable convertir la columna `country` en el **índice** del DataFrame (`df.set_index('country')`) antes de correr el modelo: como es una variable categórica (texto), pasarla a índice asegura que el modelo la use únicamente como **etiqueta** para identificar cada país en los gráficos, sin intentar computarla numéricamente.
- No hace falta programar de más: el resultado de PCA es el mismo sin importar la implementación (Prince, Scikit-learn, o generado con ayuda de un LLM) — lo que cambia y realmente importa es la **interpretación**.

### Interpretación sobre datos reales

- **Ingreso (income) vs. tasa de mortalidad infantil (child mortality):** correlación **negativa** — a mayor ingreso, menor mortalidad infantil.
- **Salud vs. exportaciones:** correlación prácticamente nula (vectores casi ortogonales).
- **Importaciones vs. exportaciones:** correlación fuerte positiva. Singapur, por ejemplo, se ubica en el extremo de ambas variables: exporta e importa mucho.
- **Haití:** alta mortalidad infantil, alta fertilidad, y en dirección opuesta (contrapuesta) a ingreso, ingreso per cápita, expectativa de vida y salud — es decir, bajo en todas esas variables.
- **Salud alta:** Japón (primer lugar), Estados Unidos, Portugal, Francia, Australia, Canadá — Costa Rica también aparece bien posicionada en esta variable.
- **Inflación alta:** Nigeria, Pakistán, Sudán, Uganda, y —dependiendo del año de captura de los datos— Argentina y Venezuela. La lectura de este tipo de variables macroeconómicas depende fuertemente del período histórico de los datos: por ejemplo, una política de dolarización fallida puede disparar temporalmente la inflación de un país, y el mismo país puede verse muy distinto una década después.

> Cuando varios países quedan agrupados cerca del centro del plano (con correlación casi nula respecto a los ejes), esto puede deberse a que **outliers muy extremos** (como Singapur o Luxemburgo, con exportaciones/importaciones fuera de rango) distorsionan la escala del gráfico y empujan a los países "normales" hacia el centro. Antes de concluir que esos países están en un nivel intermedio, conviene revisar su coseno cuadrado, porque puede que sí estén bien representados a pesar de estar cerca del origen.

### Coseno cuadrado con países reales

Países como **Singapur, España e Italia** están bien representados (coseno cuadrado combinado por encima del 60 %). Países como **Jamaica y Venezuela** están mal representados — no se puede afirmar nada concluyente sobre ellos a partir de este plano, porque se perdió su información al reducir la dimensionalidad.

Para decidir con cuáles individuos trabajar, conviene **filtrar primero los que están bien representados** (coseno cuadrado combinado por encima del umbral) y sacar las conclusiones únicamente sobre ese subconjunto, ignorando el resto.

---

## 7. Ventajas y desventajas del PCA

| Ventajas | Desventajas |
|----------|--------------|
| **Reducción de dimensionalidad:** representa los datos en un espacio de menor dimensión conservando la mayor variabilidad posible. | **Interpretabilidad limitada:** las componentes principales son combinaciones lineales de las variables originales, lo que puede dificultar su interpretación física o práctica. |
| **Eliminación de multicolinealidad:** al transformar variables correlacionadas en componentes no correlacionadas, PCA elimina la multicolinealidad perfecta. | **Sensible a la escala:** depende de la escala de las variables; por eso es necesario estandarizarlas para evitar que las de mayor magnitud dominen el análisis. |
| **Compresión de datos:** facilita almacenamiento, visualización y procesamiento al trabajar con menos variables. | **Supone relaciones lineales:** solo captura estructuras lineales; no es adecuado para relaciones no lineales. |
| **Minimiza la pérdida de información:** conserva la máxima varianza posible con el menor número de componentes. | **Sensible a valores atípicos:** los outliers pueden influir significativamente en las componentes principales y distorsionar los resultados. |
| **Útil en múltiples aplicaciones:** exploración de datos, reducción de ruido, reconocimiento de patrones, preprocesamiento. | **No considera la variable respuesta:** al ser no supervisado, no toma en cuenta información sobre la variable objetivo (puede no ser óptimo para predicción). |

**Multicolinealidad:** ocurre cuando existe una alta correlación entre múltiples variables del dataset (por ejemplo, 10 columnas que están todas correlacionadas entre sí). El PCA la elimina proyectando los eigenvectores de forma ortogonal entre ellos.

---

## 8. Usos de PCA

| Uso | Descripción |
|-----|-------------|
| **1. Reducción de dimensionalidad** | Reduce el número de variables originales conservando la mayor parte de la variabilidad. Se conservan los primeros $k$ componentes que explican la mayor proporción de la varianza total $\sum_{i=1}^{k}\lambda_i / \sum_{i=1}^{p}\lambda_i$. |
| **2. Visualización de datos** | Permite visualizar datos de alta dimensión en 2D o 3D usando los primeros dos o tres componentes, que capturan las direcciones de mayor variación. |
| **3. Reducción de ruido** | Al proyectar sobre los componentes dominantes y descartar los de eigenvalores pequeños, PCA filtra el ruido y preserva la estructura esencial. |
| **4. Extracción de características** | Genera nuevas características (componentes) no correlacionadas que capturan los patrones más importantes. |
| **5. Manejo de multicolinealidad** | Transforma variables correlacionadas en un conjunto ortogonal, mitigando problemas de multicolinealidad en regresiones y otros análisis. |
| **6. Compresión de datos** | Representa los datos de forma aproximada usando menos componentes para almacenamiento o transmisión, minimizando el error cuadrático medio de la reconstrucción. |
| **7. Preprocesamiento en aprendizaje automático** | Mejora el desempeño y la eficiencia de los algoritmos al eliminar redundancia y concentrarse en las direcciones más informativas. |
| **8. Detección de anomalías** | Observaciones con gran error de reconstrucción al proyectarse sobre los componentes dominantes pueden considerarse atípicas. |
| **9. Análisis exploratorio de datos (EDA)** | Ayuda a comprender las principales fuentes de variabilidad y la estructura de los datos mediante valores propios, vectores propios y cargas factoriales. |
| **10. Modelado predictivo y regresión** | Utiliza componentes principales como predictores para construir modelos más parsimoniosos, con menor riesgo de sobreajuste y mejor interpretabilidad. |

> PCA es una técnica lineal y no supervisada que transforma las variables originales $X \in \mathbb{R}^{n \times p}$ en componentes no correlacionadas $Z = XV$, donde $V = [v_1\ v_2\ \cdots\ v_p]$ es una matriz ortogonal de vectores propios de la matriz de covarianza $\Sigma = \frac{1}{n-1}X^TX$. Los valores propios $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0$ indican la varianza explicada por cada componente. La varianza total es $\sum_{i=1}^{p}\lambda_i = \text{tr}(\Sigma)$.

---

## Conceptos clave de la clase

- El PCA sintetiza una tabla $X \in \mathbb{R}^{n \times m}$ en componentes principales $C^1, \dots, C^k$, cada uno una **combinación lineal** de todas las variables originales.
- El primer eigenvector siempre se traza donde se **maximiza la varianza**; los siguientes se trazan **ortogonales** a los anteriores para eliminar redundancia — cada componente aporta información única.
- Ortogonalidad (90°) = correlación 0 %; misma dirección = correlación positiva fuerte; dirección opuesta (180°) = correlación 100 % negativa. Dos componentes ortogonales a un tercero pueden estar correlacionados negativamente entre sí sin dejar de ser válidos.
- **El PCA es completamente lineal y solo funciona con variables numéricas.** Para problemas no lineales: t-SNE, UMAP. Para variables muy cualitativas: Análisis de Correspondencias Simple/Múltiple.
- **La estandarización es obligatoria:** sin ella, la variable de mayor escala domina el resultado. El propio algoritmo la realiza internamente.
- Reducir dimensiones siempre implica **pérdida de información**; cuánta pérdida es aceptable depende del negocio, no de un criterio matemático único.
- **Varianza explicada** dice cuánta información aporta cada componente; el **plano principal** grafica individuos; el **círculo de correlación** grafica variables; el **biplot** combina ambos.
- **Coseno cuadrado:** mide qué tan bien representado está un individuo o variable en el plano proyectado. Umbral práctico: **60 %** acumulado. Por debajo, no hay evidencia suficiente para afirmar nada sobre ese caso — es incertidumbre, no una conclusión negativa.
- **Contribución:** mide cuánto aporta una variable a construir un eje específico; las contribuciones a un mismo componente siempre suman 1.
- Elegir mal las componentes a usar (por ejemplo, combinar $C^1$ con un componente de baja varianza) puede producir proyecciones irreconocibles o engañosas — es un trade-off entre representar bien a la mayoría o a un caso particular.
- Antes de aplicar PCA o clustering a un problema, conviene verificar con tests estadísticos si los datos realmente admiten ese tipo de análisis; aplicarlo sin esa verificación previa es una causa común de resultados que luego resultan no ser concluyentes.

---

## Fuera del PDF — logística, tarea y metodología

### Metodología de la clase
- El código de PCA usado en clase (librería **Prince**) no fue escrito por el docente, sino que proviene del material del Dr. Aldemar Rodríguez; se reutiliza directamente sin necesidad de reprogramarlo.
- El foco de la clase y de la tarea es la **interpretación de resultados**, no la programación: cualquier implementación (Prince, Scikit-learn, o generada con ayuda de una IA) llega al mismo resultado numérico. Lo que un gerente de negocio pide son conclusiones, no detalles de las librerías o funciones usadas.
- Existe un repositorio de GitHub del curso con una versión más avanzada del código en programación orientada a objetos, disponible para quienes quieran profundizar, pero no es la que se usa en la tarea.

### Tarea de la semana
Aplicar PCA sobre **tres datasets** ya subidos al campus (`country data`, uno de autos/`cars` y uno de análisis de clientes de telecomunicaciones/`Telecom Churn`), replicando los gráficos vistos en clase (scree plot, plano principal, círculo de correlación, biplot) y respondiendo preguntas de interpretación para cada uno.

Para el caso de `country data` puntualmente se pide:
- Identificar qué variables se asocian directamente con la mortalidad infantil (*child mortality*) y con el ingreso per cápita.
- Identificar qué países se ubican en los extremos de la mortalidad infantil según el biplot, y cuáles variables resultan opuestas a esa variable (ingreso, ingreso per cápita, expectativa de vida, salud, fertilidad total).

### Otros
- Se ofrecieron sesiones extra de repaso (viernes o sábado) para resolver dudas prácticas de la tarea; estas sesiones no cubren materia nueva, son repaso de lo ya visto en clase.
- Las fechas de entrega de tareas corren durante todo el cuatrimestre; el campus virtual puede marcar una entrega como "atrasada" por una configuración inicial, pero eso no afecta la nota.
- Recomendación general para las tareas: priorizar la **síntesis de las respuestas** e interpretación de negocio por encima de pulir el código o la presentación visual del notebook.
