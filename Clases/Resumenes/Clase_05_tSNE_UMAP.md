# Clase 5 — t-SNE y UMAP

**Curso:** Minería de Datos · Lead University
**Docente:** Rachit Vargas (MBA | Engineer)
**PDF fuente:** `Clases/PDFs/Clase 5 - t-SNE UMAP.pdf`

> Continuación de la reducción de dimensionalidad: dos algoritmos **no lineales**, t-SNE y UMAP, que resuelven lo que el PCA y el análisis de correspondencias no pueden. Se compara su comportamiento visual contra PCA sobre tres datasets (estudiantes, Iris, dígitos escritos a mano) y se explica el mecanismo interno de t-SNE paso a paso; UMAP queda introducido a nivel conceptual, con el detalle del algoritmo pendiente de repaso en la siguiente clase.

---

## 1. Métodos de reducción de dimensionalidad

Recordatorio: reducir dimensionalidad significa simplificar un conjunto de datos complejo (muchas variables o columnas) transformándolo en una versión más manejable —normalmente dos o tres ejes—, conservando la mayor información posible de los datos originales.

### 1.1 PCA (1901) vs t-SNE (2008) y UMAP (2018)

| Algoritmo | Año | Tipo |
|---|---:|---|
| PCA / ACP | 1901 | Lineal |
| t-SNE (t-distributed Stochastic Neighbor Embedding) | 2008 | No lineal |
| UMAP (Uniform Manifold Approximation and Projection) | 2018 | No lineal |

El PCA es un algoritmo de reducción **lineal**: genera combinaciones lineales de las variables originales. Muchas veces la información de un conjunto de variables no se puede extraer como una combinación lineal — en esas situaciones existen algoritmos de reducción **no lineales** como t-SNE y UMAP.

**Qué significa "no lineal".** Con dos variables $x_1$ (por ejemplo salario) y $x_2$ (años de experiencia), si al aumentar $x_2$ aumenta también $x_1$ (o disminuye, en una relación inversa), una sola línea recta explica el problema — eso es un problema **lineal**. Un problema es **no lineal** cuando los puntos no siguen ninguna estructura de línea recta: por ejemplo, separar dos clases (equis y círculos) que están mezcladas de tal forma que ninguna línea recta las separa; se necesitaría una curva que suba y baje para lograrlo. En la práctica, un porcentaje importante de los datasets reales guarda relaciones no lineales, y ante esos casos el PCA y el análisis de correspondencias no funcionan bien.

t-SNE y UMAP son algoritmos relativamente nuevos y todavía activos en investigación: incluso quedan dudas abiertas sobre ellos, como la falta de una métrica de varianza explicada equivalente a la del PCA.

**Origen de los nombres:**
- **t-SNE**: la *t* viene de la distribución **t-Student** que utiliza; *Stochastic* porque el algoritmo tiene un componente aleatorio (correr el algoritmo dos veces puede dar resultados distintos, aunque similares — a diferencia de un algoritmo **determinístico**, que siempre da el mismo resultado); *Neighbor Embedding* porque agrupa los puntos según qué tan "vecinos" son entre ellos (si dos individuos son cercanos en el espacio original, deben proyectarse cercanos en el mapa reducido).
- **UMAP**: es más complejo de explicar; se apoya en teoría de grafos y en un resultado matemático llamado **teorema del nervio**.

### Comparación visual: PCA vs t-SNE vs UMAP

**Dataset de estudiantes.** El PCA agrupa los puntos en dos nubes parcialmente separadas; t-SNE forma tres grupos claramente compactos y distantes entre sí; UMAP también separa en grupos, aunque con una forma más alargada y continua entre ellos.

**Dataset Iris** (4 variables explicativas — longitud y ancho de sépalo y pétalo— y una etiqueta de especie: setosa, versicolor, virginica). Con PCA, *setosa* se separa bien, pero *versicolor* y *virginica* quedan entrelazadas y se confunden. t-SNE separa las tres especies de forma casi perfecta — un resultado que impresiona por lo limpio de la agrupación. UMAP separa mejor que el PCA, pero no logra una separación tan nítida como t-SNE en este dataset.

**Dataset de dígitos escritos a mano** (imágenes representadas por píxeles, dígitos del 0 al 9). El PCA amontona casi todos los dígitos en el centro del gráfico, distinguiendo apenas algunos. t-SNE separa los dígitos en grupos mucho más definidos. UMAP los separa incluso mejor que t-SNE y que el PCA.

**La desventaja de t-SNE y UMAP: no calculan varianza explicada.** A diferencia del PCA, que siempre reporta cuánta información conserva cada componente, no existe (o al menos no es de conocimiento público) una métrica de inercia o varianza explicada para t-SNE y UMAP. Esto es un arma de doble filo: son algoritmos muy potentes para separar visualmente los individuos, pero no hay forma de cuantificar cuánta información se retuvo al reducir la dimensionalidad. Por esta razón, muchas personas evitan estos algoritmos cuando necesitan justificar cuantitativamente sus resultados. La explicación es que su objetivo no es estadístico como el del PCA, sino más bien cualitativo y visual: estudiar cómo se relacionan los individuos entre sí y qué agrupaciones (clusters) se forman en el plano de baja dimensión — por eso tanto t-SNE como UMAP solo trabajan con **individuos** y no proyectan variables como sí lo hace el PCA (que sí permite graficar variables mediante el círculo de correlación).

Precisamente por ese enfoque en agrupar individuos, t-SNE y UMAP son las herramientas de reducción de dimensionalidad más útiles como paso previo al **clustering** (tema de la siguiente clase): visualizar en dos dimensiones cómo se agrupan los individuos de un negocio (por ejemplo, tiendas de una cadena según su patrón de ventas) tiene mucho más sentido con estos algoritmos que con el PCA o el análisis de correspondencias.

---

## 2. t-SNE (t-distributed Stochastic Neighbor Embedding)

### 2.1 Origen y definición

t-SNE se basa en el método "Stochastic Neighbor Embedding", desarrollado originalmente por Sam Roweis y Geoffrey Hinton; Laurens van der Maaten propuso después la variante que usa la distribución t de Student (la versión que se usa hoy). t-SNE mide la distancia entre **todas las observaciones** (individuos) del conjunto de datos —calcula la matriz de distancias— y luego aleatoriza esas observaciones en, generalmente, dos nuevos ejes.

### 2.2 Paso 1 — matriz de distancias

La distancia entre dos individuos se calcula con la fórmula euclidiana:

$$d(x, y) = \sqrt{\sum_{i=1}^{n} (y_i - x_i)^2}$$

**Ejemplo numérico.** Sobre un plano $x_1, x_2$ se ubican tres puntos: $A = (1, 2)$, $B = (2, 3)$, $C = (5, 5)$. Visualmente, $A$ y $B$ están más cerca entre sí que $A$ y $C$. Aplicando la fórmula:

$$d(A, B) = \sqrt{(1-2)^2 + (2-3)^2} = \sqrt{2} \approx 1.41$$

$$d(A, C) = \sqrt{(1-5)^2 + (2-5)^2} = \sqrt{25} = 5$$

La distancia confirma la lectura visual: $A$ y $B$ están mucho más cerca ($1.41$) que $A$ y $C$ ($5$).

t-SNE repite este cálculo para **todos los pares de individuos** y arma una matriz de distancias completa (individuo contra individuo).

### 2.3 De distancias a probabilidades de vecindad

Para cada punto $x_i$, t-SNE calcula la distancia a cada otro punto $x_j$, y luego **escala** esas distancias para cuantificar la probabilidad de que $x_j$ sea su vecino. El proceso:

1. Se obtienen las distancias sin escalar (por ejemplo: 0.300, 0.380, 0.380, 0.020, 0.010, 0.009).
2. Se suman todas ($1.099$ en el ejemplo).
3. Se divide cada distancia entre esa suma total, obteniendo probabilidades escaladas que sí suman 1 (0.273, 0.346, 0.346, 0.018, 0.009, 0.008).

Formalmente, la probabilidad condicional de que $x_j$ sea vecino de $x_i$ usa una distribución gaussiana:

$$p_{j|i} = \frac{\exp\left(-\dfrac{\lVert x_i - x_j \rVert^2}{2\sigma_i^2}\right)}{\sum_{k \neq i} \exp\left(-\dfrac{\lVert x_i - x_k \rVert^2}{2\sigma_i^2}\right)}$$

La probabilidad $p_{j|i}$ es alta si $x_j$ está cerca de $x_i$ en el espacio original. Repitiendo este cálculo para cada individuo contra todos los demás, se obtiene una matriz de distancias/probabilidades completa, en donde cada fila guarda una distribución gaussiana de probabilidades de vecindad.

### 2.4 Paso 2 — mapa aleatorio en baja dimensión y distribución t-Student

El segundo paso del algoritmo es **aleatorizar** los individuos a lo largo de, generalmente, dos nuevos ejes — de aquí proviene la palabra "Stochastic" del nombre. Se colocan los puntos $y_i \sim \mathcal{N}(0, e^2 I_d)$ en $\mathbb{R}^d$ ($d = 2$ o $3$) de forma aleatoria.

t-SNE calcula las distancias entre los individuos en este nuevo espacio aleatorio y las convierte en probabilidades igual que antes, pero con una diferencia clave: en lugar de usar la distribución normal (gaussiana), usa la distribución **t de Student** con 1 grado de libertad:

$$q_{ij} = \frac{\left(1 + \lVert y_i - y_j \rVert^2\right)^{-1}}{\sum_{k \neq \ell} \left(1 + \lVert y_k - y_\ell \rVert^2\right)^{-1}}, \quad q_{ii} = 0$$

### 2.5 Por qué t-Student y no gaussiana

En espacios de muchas dimensiones, muchos puntos pueden estar "moderadamente lejos" entre sí. Al intentar meter todo eso en solo 2D, una distribución normal hace que esos puntos moderadamente lejanos reciban probabilidades extremadamente pequeñas — el algoritmo termina empujando demasiados puntos hacia el centro, es decir, todo se amontona (el mismo problema que le ocurre al PCA cuando reduce muchas variables a dos).

La distribución t-Student tiene **colas más pesadas y alargadas** que la normal: los puntos lejanos siguen teniendo una probabilidad relativamente importante. Esas colas más largas permiten que puntos moderadamente lejanos ejerzan fuerzas repulsivas más fuertes entre sí, y por lo tanto el algoritmo puede separarlos mejor en el mapa 2D — es la explicación de por qué t-SNE logra clusters mucho más nítidos que el PCA.

### 2.6 Paso 5 — comparar y ajustar (proceso iterativo)

El objetivo de t-SNE es que el mapa en baja dimensión ($Q$) reproduzca lo mejor posible las relaciones de similitud (distancia) del espacio original ($P$). Para lograrlo, el algoritmo repite un ciclo hasta que $P$ y $Q$ sean muy parecidas, es decir, hasta que la **divergencia KL** (Kullback-Leibler) sea pequeña:

$$KL(P \parallel Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

- **KL grande** → mapa malo (las relaciones de distancia originales no se preservaron).
- **KL pequeña** → mapa bueno (los individuos que estaban cerca en el espacio original siguen cerca en el mapa reducido).

El ciclo iterativo:
1. **Comparar** — medir qué tan diferentes son $P$ (original) y $Q$ (mapa 2D) mediante KL.
2. **Ajustar posiciones** — mover los puntos en el mapa 2D: acercar los que en $P$ eran similares (alta probabilidad) y alejar los que en $P$ eran diferentes (baja probabilidad).
3. **Repetir** — volver a calcular $Q$, comparar con $P$ de nuevo, y ajustar, hasta que cada iteración reduzca la divergencia KL y el algoritmo converja.

El proceso es conceptualmente similar al entrenamiento de una red neuronal (parte de valores aleatorios, itera, corrige, converge a un error mínimo); la diferencia es que en t-SNE hay trazabilidad completa porque se aplica una sola fórmula matemática explícita en cada paso, sin las capas de una red neuronal.

El mapa final preserva la **estructura local**: los puntos que eran cercanos en el espacio original permanecen cerca, y los clusters se hacen visibles.

---

## 3. UMAP (Uniform Manifold Approximation and Projection)

> El desarrollo del algoritmo de UMAP quedó pendiente de repasar con más detalle en la siguiente clase; el tiempo de la sesión solo permitió introducir sus características principales, sin llegar a explicar en profundidad los pasos matemáticos.

UMAP es un algoritmo de reducción de dimensión no lineal cuyo objetivo es mejorar algunas de las limitaciones de t-SNE. Funciona de forma similar (encuentra distancias en un espacio con muchas variables y luego intenta reproducirlas en baja dimensión), pero difiere, entre otras cosas, en cómo mide esas distancias. UMAP es considerablemente más rápido que t-SNE.

**Diferencias clave frente a t-SNE:**

| Característica | t-SNE | UMAP |
|---|---|---|
| Naturaleza | Estocástico (resultado puede variar entre corridas) | **Determinístico** (misma entrada → misma salida) |
| Proyectar nuevos individuos | No | **Sí** — puede usarse incluso como método supervisado |
| Estructura preservada | Principalmente local | Local **y global** |
| Velocidad | Más lento | Más rápido, escalable y estable |

Al preservar tanto la estructura local como la global, UMAP permite interpretar individuos y clústeres que están cerca en baja dimensión como similares también en las dimensiones altas originales — algo que t-SNE, enfocado casi exclusivamente en relaciones locales, no garantiza (puede incluso separar artificialmente grupos que en realidad estaban relacionados).

### Los 7 pasos del algoritmo (resumen conceptual)

1. **Datos en muchas dimensiones** — cada punto es un dato con varias características.
2. **Buscar vecinos cercanos** — para cada punto, UMAP identifica sus vecinos y mide qué tan fuerte es su relación (similitud), usando una noción de **probabilidad de cercanía**, no solo distancia "bruta":
   $$p_{ij} = \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right)$$
   donde $\rho_i$ es el radio de cercanía y $\sigma_i$ controla la escala total.
3. **Construir el grafo global** — se simetrizan las probabilidades entre cada par de puntos mediante una unión difusa ("fuzzy union"):
   $$p_{ij}^{(sym)} = p_{ij} + p_{ji} - p_{ij}\,p_{ji}$$
   Repitiendo esto para todos los puntos se obtiene un grafo que representa la estructura completa del conjunto de datos.
4. **Proyección inicial en 2D** — se coloca cada punto en un plano 2D de forma aleatoria.
5. **Optimización: acercar y alejar** — UMAP mueve los puntos para acercar los vecinos que estaban cerca en el grafo original y alejar los que no lo eran.
6. **Resultado intermedio** — tras varias iteraciones, la estructura empieza a tomar forma: clusters y relaciones cada vez mejor representados.
7. **Proyección 2D final** — UMAP entrega la representación final que preserva, en la medida de lo posible, la estructura del grafo original: vecinos cercanos en el grafo quedan cercanos en 2D (probabilidad alta), y los que no son vecinos quedan lejanos (probabilidad baja).

En síntesis: UMAP construye un grafo probabilístico de conexiones locales y luego optimiza iterativamente la posición de los puntos en baja dimensión hasta reproducir lo mejor posible la estructura original de los datos.

---

## Conceptos clave de la clase

- **PCA es lineal; t-SNE y UMAP son no lineales.** Un problema no lineal es aquel que no puede explicarse trazando una sola línea recta entre las variables.
- t-SNE (2008) y UMAP (2018) son algoritmos recientes que **solo trabajan con individuos**, no con variables — a diferencia del PCA, que analiza ambos.
- Ninguno de los dos calcula una **varianza explicada** equivalente a la del PCA: son algoritmos orientados a visualizar agrupaciones (clusters), no a cuantificar cuánta información se retuvo.
- **t-SNE**: calcula una matriz de distancias euclidianas entre todos los individuos, las convierte en probabilidades de vecindad (distribución gaussiana), coloca los puntos aleatoriamente en 2D, calcula ahí las distancias con una distribución **t-Student** (colas pesadas, evita que todo se amontone en el centro), y ajusta iterativamente las posiciones minimizando la **divergencia KL** entre las probabilidades originales y las del mapa reducido.
- **Estocástico** = el resultado puede variar entre corridas (t-SNE); **determinístico** = misma entrada siempre da la misma salida (UMAP).
- **UMAP** mejora las limitaciones de t-SNE: es determinístico, más rápido, permite proyectar nuevos individuos (uso supervisado) y preserva tanto la estructura local como la global mediante un grafo de probabilidades de cercanía.
- En datasets con relaciones no lineales (Iris, dígitos escritos a mano), t-SNE y UMAP separan los grupos con mucho más nitidez que el PCA, que tiende a amontonar los individuos en el centro del plano.
- Estos dos algoritmos son la base natural para el tema de **clustering** de la siguiente unidad, porque su objetivo principal es visualizar cómo se agrupan los individuos.

---

## Fuera del PDF — repaso de código, quiz y logística

### Repaso de código: PCA vs MCA sobre datos de churn

Repaso del código de Análisis de Correspondencia Múltiple (MCA) pendiente de la semana anterior, usando la librería **Prince** (desarrollada por Max Halford), que agrupa varios algoritmos de reducción de dimensionalidad (PCA, análisis de correspondencia simple y múltiple, y un método para datos mixtos —cuantitativos y cualitativos— llamado *Factor Analysis of Mixed Data*). Prince no incluye t-SNE ni UMAP; para esos se usan otras librerías.

**Dataset:** `Telco Customer Churn` — datos de clientes de una compañía de servicios (telefonía, internet, streaming, soporte técnico) con **21 columnas**, en su mayoría variables **cualitativas** (partner, dependents, tipo de contrato, servicios contratados), más algunas cuantitativas (total de la factura). Es un dataset de aprendizaje supervisado (variable *Churn*: si el cliente sale o no de la compañía), reutilizado aquí para ilustrar reducción de dimensionalidad no supervisada.

- Se instancia la clase pasándole el DataFrame y el nombre de la columna índice (`CustomerID`), y se ajusta pidiendo dos componentes.
- **Variables dummy:** `pd.get_dummies` convierte cada columna categórica en columnas binarias (0/1), una por cada categoría. Una variable con 2 categorías (hombre/mujer) genera 2 columnas; una con 3 categorías (rojo/azul/verde) genera 3. El método de la librería ya aplica esta conversión internamente.
- **Varianza explicada:** PCA conservó 24.14 % (dim. 1) y 13.27 % (dim. 2), acumulado 37.41 %. MCA conservó 23.36 % y 11.87 %, acumulado 35.23 %. Con este criterio, el PCA sería la opción a elegir por conservar más información — aun tratándose de un dataset mayormente cualitativo, para el que en teoría el MCA está mejor diseñado.
- Un 37 % de información conservada es un valor bajo, pero la ventaja de reducir a solo 2 dimensiones (de 21 originales) es que, aunque se pierda mucha información, sí es posible visualizar e interpretar la parte que se logró rescatar; la desventaja es que separar 21 variables en solo 2 ejes inevitablemente sacrifica gran parte del detalle.
- **Biplot — insights sobre los clientes que no salen** (`Churn = No`): se caracterizan por contrato a 1 o 2 años, ser *partner*, tener contratados varios servicios (multilíneas, protección de dispositivo, streaming) y mayor tiempo de permanencia en la compañía. **Clientes que sí salen** (`Churn = Yes`): contrato mes a mes, poca permanencia (1 a 8 meses), pocos servicios contratados y facturas más bajas. El patrón es prácticamente el mismo tanto en PCA como en MCA, aunque el MCA resultó ligeramente más legible visualmente pese a explicar algo menos de varianza.
- **Coseno cuadrado en un contexto de negocio:** conviene priorizar la interpretación de los individuos mejor representados (coseno cuadrado alto) y, dentro de esos, priorizar a los clientes de mayor peso económico (por ejemplo cuentas corporativas con facturas de miles de dólares) por sobre clientes individuales de bajo consumo — graficar cómo se comporta una cuenta como Walmart o PriceSmart es más relevante para el negocio que graficar una pulpería local.
- **Contribución vs. coseno cuadrado — el dilema del criterio de corte:** el coseno cuadrado sí tiene un criterio práctico de corte (60 % acumulado, visto en la Clase 3) para decidir si un individuo está bien representado. La **contribución**, en cambio, no tiene un punto de corte conocido: solo permite comparar una variable contra otra ("esta contribuye más que aquella"), pero no establece un umbral objetivo para decidir cuándo excluir una variable. Es una decisión que recae en el juicio y experiencia del científico de datos, no en un criterio estadístico único.
  - Analogía visual: si el vector de un componente (por ejemplo *high jump*) es más largo que el de otro (*jabalina*), el primero tiene mayor contribución. Si un vector es extremadamente corto en comparación con los demás, ahí sí se puede argumentar con más seguridad que su contribución es demasiado baja como para ser relevante.
  - Analogía con el error cuadrático medio (RMSE) en series de tiempo: un valor de RMSE aislado (por ejemplo 0.65) tampoco tiene interpretación por sí solo — solo cobra sentido al compararlo contra el RMSE de otro modelo (por ejemplo 0.69) para decidir cuál se equivoca menos. Es el mismo principio que la contribución en PCA/MCA: una métrica comparativa, no absoluta.

### Quiz de repaso (Wooclap/Mentimeter) — PCA y aprendizaje no supervisado

Puntos reforzados durante la corrección del quiz, más allá de lo ya cubierto en la Clase 3:

- Los componentes principales son **combinaciones lineales** de las variables originales, nunca variables originales reescaladas ni medidas de correlación entre variables — de hecho, el objetivo del PCA es justamente **eliminar** la correlación entre sus componentes (ortogonalidad = independencia = correlación 0).
- El primer componente principal es, por definición, el que **maximiza** la varianza explicada — nunca el de menor varianza.
- La estandarización nunca es "innecesaria por defecto": es indispensable cuando las variables tienen escalas distintas (dólares, libras, colones), y el propio modelo de PCA la aplica automáticamente. Para variables **categóricas** no es necesaria, porque su escala (0/1) no genera el mismo tipo de distorsión que una escala numérica amplia.
- La estandarización también es relevante en varios algoritmos de **aprendizaje supervisado** basados en distancias (KNN) o sensibles a la escala (redes neuronales) — no solo en no supervisados. Algoritmos basados en árboles (Random Forest, Gradient Boosting, AdaBoost) no la necesitan.
- **Aprendizaje no supervisado** se define por la ausencia de una variable objetivo (etiqueta): el algoritmo no busca pronosticar nada, sino descubrir estructura, patrones o agrupaciones naturales en los datos — es una fase exploratoria. Ejemplo de aplicación de negocio: bancos, hoteles y plataformas como Netflix agrupan (clusterizan) a sus clientes según comportamiento de consumo para luego recomendar productos o contenido a un cliente basándose en lo que consumieron otros clientes de su mismo grupo — sin necesidad de predecir nada puntual sobre ese individuo.
- **Aprendizaje semi-supervisado** (mencionado a partir de una pregunta de clase): combina ambos enfoques. El algoritmo **Isolation Forest**, usado en detección de anomalías, es un ejemplo: no parte de etiquetas históricas de "esto es una anomalía", pero las va deduciendo él mismo a medida que procesa los datos — de ahí que se considere semi-supervisado.
- Una **hipótesis causal** no se prueba con aprendizaje no supervisado, sino con **tests estadísticos** (t-test, ANOVA/MANOVA, A/B testing): son las herramientas que permiten aceptar o rechazar si un efecto es causal o producto del azar.

### Logística y metodología del curso

- Tarea 4 (la que correspondía a análisis de correspondencia simple y múltiple) se entrega esta semana, dado que el código completo se terminó de explicar hasta esta clase.
- El docente ofreció una sesión adicional de repaso sobre tests estadísticos (t-test, MANOVA, etc.) para el viernes, de forma virtual y grabada.
- Recomendación explícita de no usar IA para resolver el quiz de clase: se relató que en otro grupo, al pedir a los estudiantes que justificaran oralmente sus respuestas después de un quiz con alta tasa de aciertos, quedó en evidencia que habían usado IA sin entender el contenido. La nota del quiz depende de la participación, no de la cantidad de respuestas correctas — su propósito es diagnóstico, para que el docente sepa qué temas repasar.
- Se mencionó una actividad opcional organizada por la universidad junto con Microsoft Costa Rica sobre la herramienta Microsoft Fabric (análisis de datos), recomendada como buena oportunidad de *networking*, aunque sin certificado de participación.
