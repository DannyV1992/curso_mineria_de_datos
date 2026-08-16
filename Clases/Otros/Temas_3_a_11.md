# Explicación por tema — Clase 3 a Clase 11 (Minería de Datos)

Documento complementario al [Resumen_Examen_Clase3_a_11.md](Resumen_Examen_Clase3_a_11.md). Aquí cada clase se desglosa en sus temas, con una explicación en lenguaje simple de qué es y por qué importa, seguida de los puntos principales a recordar.

---

## Clase 3 — PCA (Análisis de Componentes Principales)

### Tema 1: El problema de la alta dimensionalidad
Cuando un dataset tiene muchas variables (decenas o cientos), pasan tres cosas malas: se vuelve pesado de procesar, los modelos supervisados corren riesgo de sobreajuste, y —el problema que ataca el PCA— se vuelve imposible de visualizar e interpretar, porque el ojo humano solo entiende hasta 3 dimensiones. Además, entre más variables se agregan, más se dispersan los puntos entre sí (el espacio se vuelve "vacío"), lo que dificulta ver patrones.

**Puntos principales:**
- Tres problemas: complejidad computacional, riesgo de overfitting, dificultad de interpretar/visualizar.
- El PCA ataca principalmente el tercer problema.
- A mayor dimensión, los puntos se alejan más entre sí (espacio disperso).

### Tema 2: Qué resuelve el PCA — componentes principales
El PCA sintetiza muchas variables originales en pocas "variables sintéticas" (componentes principales), donde cada componente es una combinación lineal de todas las variables originales. La cantidad máxima de componentes que se puede construir es igual al número de columnas originales, pero normalmente solo se usan las primeras 2 o 3, porque son las que concentran más información (varianza).

**Puntos principales:**
- $C^k = a_{k1}X^1 + \cdots + a_{km}X^m$: cada componente es combinación lineal de todas las variables.
- Reducir siempre implica pérdida de información; cuánta pérdida aceptar depende del negocio.
- El PCA detecta y elimina **multicolinealidad**: variables que dicen lo mismo (ej. ventas en dólares/colones) se comprimen en una sola componente.
- El primer eigenvector maximiza la varianza; los siguientes son ortogonales (90°) entre sí para no repetir información.
- Ortogonalidad = correlación 0%; misma dirección = correlación positiva fuerte; dirección opuesta = correlación 100% negativa.
- El PCA es **100% lineal** y solo funciona con variables **numéricas**.

### Tema 3: El algoritmo PCA paso a paso
El algoritmo tiene 5 pasos esenciales: estandarizar los datos (para que ninguna variable domine por su escala), calcular la matriz de covarianza/correlación, obtener sus eigenvalores y eigenvectores, elegir cuántos componentes conservar (los de mayor eigenvalor) y proyectar los datos sobre esos componentes.

**Puntos principales:**
- Estandarizar es obligatorio: sin ello, la variable de mayor magnitud domina el resultado. El propio algoritmo lo hace internamente.
- Eigenvector = dirección; eigenvalor = cuánta varianza (información) hay en esa dirección.
- Se ordenan los eigenvalores de mayor a menor para decidir qué componentes conservar.

### Tema 4: Qué arroja el PCA (interpretación de resultados)
El PCA entrega varios elementos gráficos y numéricos que hay que saber leer juntos, nunca por separado: la varianza explicada (cuánta información aporta cada componente), el plano principal (individuos proyectados), el círculo de correlación (variables proyectadas), el biplot (ambos superpuestos), el coseno cuadrado y la contribución.

**Puntos principales:**
- **Coseno cuadrado (cos²):** calidad de representación de un individuo/variable en el plano; umbral práctico 60% acumulado.
- **Contribución:** cuánto aporta una variable a construir un eje; suma 1 por componente; sin umbral de corte fijo.
- Un cos² bajo no significa que el individuo sea "malo" en algo: significa que falta información porque se perdió al reducir dimensiones.
- El biplot combina variables e individuos: permite decir "este individuo se caracteriza por..." leyendo la dirección de las variables cercanas.

### Tema 5: Ventajas, desventajas y usos
El PCA es útil para reducir dimensionalidad, eliminar multicolinealidad, comprimir datos, reducir ruido y hacer EDA; pero pierde interpretabilidad directa (las componentes son combinaciones lineales abstractas), es sensible a escala y a outliers, y no sirve para relaciones no lineales.

**Puntos principales:**
- Ventaja clave: elimina multicolinealidad proyectando eigenvectores ortogonales.
- Desventaja clave: baja interpretabilidad física de las componentes.
- No es supervisado: no considera ninguna variable objetivo.

---

## Clase 4 — ACS y ACM (Análisis de Correspondencias Simple y Múltiple)

### Tema 1: El problema de los datos categóricos
Cuando los datos son cualitativos (género, nivel educativo, tipo de producto), no se puede calcular varianza ni covarianza como en PCA. Para estudiar la asociación entre dos variables categóricas se construye una **tabla de contingencia** (como una tabla pivote de Excel), que cruza las categorías de ambas variables y cuenta cuántas veces aparece cada combinación.

**Puntos principales:**
- Variable cualitativa = etiqueta/categoría, no puede tomar valores intermedios (no existe una escolaridad "3.5").
- Con muchas categorías, la tabla de contingencia crece y se vuelve imposible de leer directamente — mismo problema de alta dimensionalidad que motivó el PCA.

### Tema 2: ACS (Análisis de Correspondencias Simple)
El ACS funciona sobre exactamente dos variables cualitativas. En vez de covarianza, usa **perfiles de frecuencia** (cómo se distribuye cada fila o columna) y una **distancia chi-cuadrado** para medir qué tan parecidos son esos perfiles. El objetivo es el mismo que en PCA: encontrar los ejes que expliquen la mayor inercia (información) posible y proyectar filas y columnas en un mapa de baja dimensión.

**Puntos principales:**
- Perfil de fila: cómo se distribuye un individuo entre las categorías; perfil de columna: cómo se distribuyen los individuos en una categoría.
- Distancia chi-cuadrado en vez de correlación; inercia en vez de varianza.
- El algoritmo usa SVD (descomposición en valores singulares) sobre residuos tipificados.
- Lectura del mapa: proximidad fila-columna = asociación positiva; lados opuestos = asociación negativa; cerca del origen = perfil promedio.
- Se debe interpretar siempre junto con inercia explicada, contribuciones y cos² — nunca el mapa solo.

### Tema 3: ACM (Análisis de Correspondencias Múltiple)
El ACM generaliza el ACS cuando hay **tres o más** variables categóricas. Para lograrlo, primero convierte todas las variables en una tabla disyuntiva completa (dummies / one-hot encoding: cada categoría se vuelve una columna binaria 0/1), y luego aplica sobre esa tabla la misma lógica del ACS.

**Puntos principales:**
- Tabla disyuntiva = codificación dummy de todas las variables categóricas juntas.
- En variables binarias (2 categorías) se puede eliminar una columna porque se infiere de la otra, evitando redundancia perfecta.
- El ACM no es una extensión del PCA, sino del ACS (mismo mecanismo, más variables).
- La interpretación del biplot es equivalente a la del círculo de correlación del PCA, pero con categorías/modalidades en vez de variables numéricas.

### Tema 4: PCA vs. ACS/ACM — cómo decidir
No hay una regla fija sobre cuál algoritmo usar ante datos cualitativos: en teoría el ACS/ACM debería rendir mejor (trabaja con el tipo de dato correcto), pero en la práctica a veces el PCA (aplicado sobre dummies) da mejores resultados. La recomendación es correr ambos y comparar la varianza/inercia acumulada.

**Puntos principales:**
- Criterio práctico: correr ambos algoritmos y elegir el que explique más información acumulada en sus primeras dimensiones.
- El criterio de Kaiser (eigenvalor > 1) es un criterio adicional para decidir cuántos componentes/dimensiones conservar.

---

## Clase 5 — t-SNE y UMAP

### Tema 1: Por qué se necesitan algoritmos no lineales
El PCA solo puede capturar relaciones lineales entre variables. Muchos datasets reales tienen relaciones no lineales (grupos que no se separan con una línea recta), y ahí el PCA falla, amontonando los individuos en el centro del plano. t-SNE (2008) y UMAP (2018) son algoritmos no lineales diseñados justamente para separar visualmente esos grupos.

**Puntos principales:**
- Problema lineal: se explica con una línea recta; no lineal: se necesita una curva o frontera irregular.
- t-SNE y UMAP solo trabajan con **individuos**, no con variables (no ofrecen biplot).
- Ninguno de los dos calcula varianza explicada — su objetivo es visual/cualitativo, no estadístico.

### Tema 2: t-SNE — mecanismo
t-SNE calcula la distancia entre todos los pares de individuos en el espacio original y las convierte en probabilidades de vecindad (distribución gaussiana). Luego coloca los puntos aleatoriamente en un espacio 2D y calcula ahí también probabilidades de vecindad, pero usando una distribución t-Student (de colas más pesadas). El algoritmo ajusta iterativamente las posiciones para que las probabilidades del mapa 2D (Q) se parezcan lo más posible a las del espacio original (P), minimizando la divergencia KL.

**Puntos principales:**
- "Stochastic" = el resultado puede variar entre corridas (no determinístico).
- La distribución t-Student evita que todo colapse al centro, permitiendo separar mejor los clusters.
- Divergencia KL grande = mapa malo; KL pequeña = mapa bueno.
- Preserva la estructura **local** (vecinos cercanos siguen cercanos), no necesariamente la global.

### Tema 3: UMAP — mecanismo
UMAP construye un grafo de probabilidades de cercanía entre puntos (basado en vecinos, no solo en distancia bruta), lo simetriza mediante una "unión difusa" (fuzzy union) para resolver el problema de que las probabilidades de conexión no son simétricas por defecto, y luego optimiza iterativamente las posiciones en baja dimensión para reproducir ese grafo.

**Puntos principales:**
- Es determinístico (misma entrada, misma salida) y más rápido que t-SNE.
- Preserva estructura local **y global**.
- Puede proyectar nuevos individuos (útil en un esquema semi-supervisado); t-SNE no puede.
- Fuzzy union: $p_{ij}^{(sym)} = p_{ij} + p_{ji} - p_{ij}\,p_{ji}$, necesaria para tener una matriz de distancias/probabilidades consistente.

---

## Clase 6 — Cierre de t-SNE/UMAP e introducción a clustering

### Tema 1: Recomendación práctica para clustering
Cuando se combine reducción de dimensionalidad con clustering, se recomienda usar t-SNE o UMAP en vez de PCA, porque separan mejor los grupos visualmente. El PCA sigue siendo útil cuando se necesita explicar *qué caracteriza* a cada grupo (vía biplot), algo que t-SNE/UMAP no ofrecen por sí solos.

**Puntos principales:**
- t-SNE/UMAP > PCA para separar clusters visualmente en datasets no lineales.
- PCA > t-SNE/UMAP para explicar qué variables caracterizan cada cluster (biplot).
- Ambos enfoques se complementan: reducir para visualizar, clustering formal para caracterizar.

### Tema 2: Definición de clustering
Clustering es una técnica de aprendizaje no supervisado que busca identificar grupos naturales de observaciones, de modo que los elementos de un mismo grupo sean más parecidos entre sí que con los de otros grupos. Es una técnica exploratoria, no predictiva.

**Puntos principales:**
- No supervisado: no hay variable objetivo/etiqueta.
- El resultado depende directamente de cómo se defina la distancia/similitud.
- Permite resumir muchos datos con pocos centroides representativos.

### Tema 3: Puntos importantes antes de clusterizar
Antes de aplicar cualquier algoritmo de clustering hay que resolver tres problemas: variables en escalas muy distintas (dominan la distancia si no se estandarizan), sensibilidad a outliers (distorsionan los centroides) y alta dimensionalidad (maldición de la dimensionalidad: las distancias pierden poder discriminativo).

**Puntos principales:**
- Escalas distintas → estandarizar/normalizar antes de clusterizar.
- Outliers → pueden arrastrar el centroide hacia una zona poco representativa.
- Alta dimensionalidad → las distancias se vuelven parecidas entre todos los puntos, dificultando distinguir grupos.

### Tema 4: Matriz de distancias y tipos de distancia
Toda técnica de clustering necesita calcular una distancia entre cada par de observaciones y organizarla en una matriz. Las distancias más comunes son euclidiana (línea recta), Manhattan (recorrido en cuadrícula) y Chebyshev (solo la diferencia máxima); existen otras para casos particulares (coseno, Jaccard, Hamming, Minkowski).

**Puntos principales:**
- Propiedades de toda matriz de distancias válida: no negativa, $d_{ii}=0$, simétrica, cumple desigualdad triangular.
- Euclidiana: $\sqrt{\sum(x_i-y_i)^2}$. Manhattan: $\sum|x_i-y_i|$. Chebyshev: $\max_i|x_i-y_i|$.
- No existe una distancia universalmente mejor; se elige según cuál agrupe mejor los datos del problema.

### Tema 5: Hopkins Statistic y VAT (evaluación previa)
Antes de clusterizar hay que verificar si los datos realmente tienen tendencia a agruparse. El Índice de Hopkins compara distancias entre puntos reales y puntos aleatorios generados en el mismo espacio; el VAT hace lo mismo de forma visual, reordenando la matriz de distancias en un heatmap.

**Puntos principales:**
- $H = \frac{\sum y_i}{\sum x_i + \sum y_i}$; H cercano a 1 = tendencia fuerte a clusterizar; H ≈ 0.5 = datos aleatorios.
- VAT: bloques oscuros sobre la diagonal = estructura de clusters; patrón disperso = no hay estructura.

---

## Clase 7 — Fundamentos de clustering (con ejemplos en vivo)

### Tema 1: Centroide
El centroide es el promedio de todas las observaciones de un cluster; resume el comportamiento típico del grupo en un solo valor por variable. Es más manejable decir "el centroide del cluster de países ricos exporta X" que describir cada país individualmente.

**Puntos principales:**
- El centroide se recalcula (y se desplaza) cada vez que se agrega un nuevo punto al cluster.
- Un outlier arrastra el centroide hacia una región poco representativa de la mayoría.

### Tema 2: La similitud determina la agrupación
Como existen muchas funciones de distancia y cada una puede llegar a un resultado distinto, el agrupamiento final depende directamente de qué distancia se elija — usar euclidiana no necesariamente da el mismo resultado que usar Manhattan sobre los mismos datos.

**Puntos principales:**
- No todos los datos son clusterizables: presentar clusters inventados a una gerencia es un error grave.
- El Índice de Hopkins es la herramienta formal para verificar esto antes de aplicar cualquier algoritmo.

### Tema 3: ¿Clusterizar sobre datos originales o sobre componentes reducidos?
Se recomienda clusterizar siempre sobre los datos originales, no sobre las componentes de un PCA, porque el PCA es una combinación lineal y los centroides resultantes quedarían en coordenadas artificiales, difíciles de explicar a un tomador de decisiones. El flujo correcto es: clusterizar sobre datos originales, usar PCA solo para visualizar los clusters ya formados.

**Puntos principales:**
- Clusterizar en datos originales preserva la interpretabilidad de negocio.
- El PCA (o t-SNE/UMAP) se reserva para la etapa de visualización posterior.

### Tema 4: Introducción al clustering jerárquico
El clustering jerárquico organiza los datos en un árbol (dendrograma) fusionando iterativamente los puntos o clusters más cercanos (enfoque aglomerativo, bottom-up) hasta llegar a un único cluster. La altura de cada fusión en el dendrograma representa la distancia a la que ocurrió.

**Puntos principales:**
- Algoritmo aglomerativo: cada punto empieza siendo su propio cluster; se fusionan de dos en dos.
- Criterio de parada: se detiene cuando todos los puntos quedan en un único cluster.
- El criterio de enlace (linkage) define cómo se mide la distancia entre dos clusters ya formados (desarrollo completo en Clase 8).

---

## Clase 8 — Hopkins, VAT y Clustering Jerárquico completo

### Tema 1: Índice de Hopkins (formal)
Es la técnica estadística formal para confirmar tendencia al agrupamiento: se generan puntos aleatorios en el mismo espacio de los datos reales, se calculan las distancias al vecino más cercano de ambos conjuntos, y se combina en el estadístico H.

**Puntos principales:**
- $H \leq 0.5$: datos aleatorios, no clusterizables. $H > 0.7$: evidencia de estructura. $H \to 1$: fuerte tendencia.
- Es un indicador estadístico "objetivo", análogo en robustez a un p-value.

### Tema 2: VAT (Visual Assessment of Tendency)
Es el complemento visual (no estadístico) del Índice de Hopkins: se calcula la matriz de distancias, se reordena de menor a mayor y se construye un heatmap. Bloques oscuros grandes sobre la diagonal indican fuerte tendencia a agrupar; si el bloque oscuro es muy pequeño, solo una minoría de los datos tiene estructura y no vale la pena clusterizar el conjunto completo.

**Puntos principales:**
- Reordenar es la clave del método: agrupa visualmente las distancias pequeñas cerca de la diagonal.
- Cuanta más superficie oscura, más fuerte la tendencia a clusterizar en la mayoría de los datos.

### Tema 3: Objetivos de todo algoritmo de clustering (criterio de Fisher)
Todo algoritmo de clustering busca dos cosas a la vez: minimizar la distancia intraclase (puntos de un mismo cluster compactos junto a su centroide) y maximizar la distancia interclase (centroides de clusters distintos bien separados). Cuando dos centroides quedan cerca, aparecen individuos "fronterizos" mal clasificados, que requieren criterio manual del analista.

**Puntos principales:**
- Intraclase: distancia dentro de un mismo cluster (buscar que sea pequeña).
- Interclase: distancia entre centroides de clusters distintos (buscar que sea grande).
- Puntos fronterizos: su asignación puede requerir decisión manual del analista, no solo del algoritmo.

### Tema 4: Clustering jerárquico vs. K-Means
El jerárquico es determinístico y recursivo, da buenos resultados pero es lento en datasets grandes (mejor para datasets pequeños). K-Means es estocástico, más rápido, mejor para datasets grandes, pero no garantiza el óptimo global por su aleatoriedad inicial.

**Puntos principales:**
- Jerárquico: determinístico, recomendado en datasets pequeños.
- K-Means: estocástico, recomendado en datasets grandes.
- Ambos buscan lo mismo (clusters), pero por caminos matemáticos distintos.

### Tema 5: Hipersegmentación
Al decidir cuántos clusters formar hay que balancear generalidad y especificidad: muy pocos clusters vuelve la campaña de marketing demasiado genérica; demasiados clusters "inventa" segmentos que no existen realmente en los datos, con el consiguiente costo de dirigir estrategias distintas al mismo tipo de cliente.

**Puntos principales:**
- Hipersegmentación = crear más clusters de los que existen realmente.
- Tiene un costo de negocio directo (presupuesto de campañas, políticas de precio).

### Tema 6: Algoritmo aglomerativo y dónde cortar el dendrograma
El algoritmo siempre termina fusionando todo en un único gran cluster, sin importar cuántos clusters se quieran al final; la cantidad de clusters se decide después, trazando una línea horizontal sobre el dendrograma. El número de intersecciones que corta esa línea es el número de clusters resultante.

**Puntos principales:**
- Nunca se fusionan tres puntos a la vez: siempre de dos en dos.
- Línea más arriba = menos clusters; línea más abajo = más clusters.
- El codo de Jambú y el índice de silueta ayudan a objetivar dónde cortar (en vez de una decisión puramente visual).

### Tema 7: Criterios de enlace (linkage)
Al fusionar clusters (no puntos individuales) hay que decidir cómo medir la distancia entre ellos. Existen cuatro criterios: salto mínimo (par más cercano, forma cadenas), salto máximo (par más lejano, clusters compactos), promedio (compromiso entre ambos) y Ward (mínima varianza, el más usado en la práctica).

**Puntos principales:**
- Single linkage: tiende a encadenar (clusters alargados).
- Complete linkage: clusters compactos y de tamaño similar.
- Average linkage: compromiso entre single y complete.
- Ward: minimiza el incremento de varianza intra-cluster; es el más usado.
- Cada criterio produce un dendrograma distinto sobre los mismos datos — no hay uno universalmente mejor.

### Tema 8: Gráficos para interpretar clusters
Una vez formados los clusters, se usan gráficos como boxplots, heatmaps, radar charts, barplots de medias y silhouette plots para caracterizar qué variables definen a cada grupo. Con audiencias no técnicas, barplots y radar charts son los más fáciles de comunicar.

**Puntos principales:**
- Boxplots/heatmaps: identifican qué variables diferencian los grupos.
- Radar chart: perfil comparativo de todos los clusters en todas las variables a la vez.
- Elección del gráfico depende del tipo de audiencia (técnica vs. gerencial).

---

## Clase 9 — Código de clustering jerárquico y K-Means

### Tema 1: Implementación de clustering jerárquico en Python
En la práctica, `pdist` calcula la matriz de distancias, `linkage` aplica el criterio de enlace elegido (ward, single, complete, average) y `fcluster` asigna la etiqueta final de cluster a cada individuo, dado un número de clusters `k` deseado.

**Puntos principales:**
- Los individuos deben ir como índice del DataFrame, nunca como columna.
- Los números de cluster son arbitrarios entre ejecuciones (la composición del grupo se mantiene, la etiqueta numérica no).
- El PCA sirve para verificar visualmente por qué se separan ciertos individuos, aunque puede no reflejar el 100% de las distancias reales (es una aproximación).

### Tema 2: Criterio de Fisher (inercia)
Combina cohesión interna y separación externa en un solo índice a maximizar: $J = L/d$, donde $L$ es la distancia intercluster y $d$ la distancia intracluster. Un valor de J aislado no significa nada por sí solo — solo sirve para comparar configuraciones entre sí.

**Puntos principales:**
- $D_{\text{intra}}$: dispersión de un cluster respecto a su centroide (buscar que sea baja).
- $D_{\text{inter}}$: separación entre centroides de clusters distintos (buscar que sea alta).
- J más alto = mejor configuración, pero solo en términos comparativos, no absolutos.

### Tema 3: Codo de Jambú
Técnica para decidir cuántos clusters K son adecuados: se grafica la inercia $W(K)$ (qué tan compactos son los clusters) contra distintos valores de K, y se elige el punto donde la curva forma un "codo" (deja de bajar de forma pronunciada). Se asocia principalmente a K-Means porque ahí K es obligatorio (en jerárquico no lo es).

**Puntos principales:**
- Inercia alta con K=1 (todo en un solo grupo); baja progresivamente al aumentar K.
- Más allá del codo, aumentar K ya no reduce significativamente la inercia → riesgo de hipersegmentación.

### Tema 4: K-Means — definición y algoritmo
K-Means divide los datos en K grupos, iterando entre dos pasos: asignar cada punto al centroide más cercano, y recalcular cada centroide como el promedio de los puntos asignados. Se repite hasta que los centroides dejan de cambiar (convergencia). Es estocástico porque los centroides iniciales se colocan aleatoriamente.

**Puntos principales:**
- K debe definirse **antes** de correr el algoritmo (a diferencia del jerárquico).
- Estocástico: distintos centroides iniciales pueden dar resultados distintos.
- El criterio de parada de iteraciones también se basa en cuándo el índice de Fisher deja de mejorar significativamente.

### Tema 5: Normalización en K-Means
Igual que todos los algoritmos de clustering, K-Means es sensible a la escala de las variables y a outliers. Estandarizar (`StandardScaler`) o normalizar (`MinMaxScaler`) antes de aplicar el algoritmo produce clusters notablemente mejor formados.

**Puntos principales:**
- Sin normalizar, la variable de mayor magnitud domina la distancia y distorsiona los clusters.
- `StandardScaler` estandariza (media 0, desviación 1); `MinMaxScaler` normaliza (rango fijo, ej. 0-1).

### Tema 6: Ventajas y desventajas de K-Means
Es simple, rápido, escalable y de fácil interpretación; pero requiere definir K de antemano, es sensible a outliers, a la inicialización aleatoria de centroides y a la escala, y no funciona bien con formas de cluster no esféricas.

**Puntos principales:**
- Ventaja clave: rapidez y escalabilidad en datasets grandes.
- Desventaja clave: no garantiza el óptimo global (depende de la inicialización).

---

## Clase 10 — DBSCAN

### Tema 1: Qué es DBSCAN y en qué se diferencia
DBSCAN agrupa observaciones según **densidad**, no según distancia a un centroide (K-Means) ni según fusiones jerárquicas. Va "conquistando" regiones densas del espacio y las convierte en clusters; los puntos alejados de cualquier región poblada se marcan como ruido/outliers. No requiere conocer el número de clusters de antemano.

**Puntos principales:**
- Densidad = qué tan poblada está una región del espacio.
- No requiere K previo (como el jerárquico), pero sí dos hiperparámetros: ε y MinPts.
- Detecta outliers de forma nativa — su principal ventaja práctica.

### Tema 2: Tipos de puntos
Cada punto se clasifica según cuántos vecinos tiene dentro de su radio ε: **punto central** (≥ MinPts vecinos), **punto de frontera** (no cumple MinPts, pero está dentro del vecindario de un punto central) y **punto de ruido** (ni central ni frontera).

**Puntos principales:**
- Un punto de frontera no es outlier: sí pertenece a un cluster, aunque no expanda la búsqueda desde él.
- El punto inicial de exploración se elige de forma 100% aleatoria (por eso DBSCAN es estocástico).

### Tema 3: Algoritmo paso a paso
Se elige un punto no visitado al azar, se construye su vecindario ε, y si es central se crea un nuevo cluster que va expandiéndose a través de los puntos centrales conectados entre sí; los puntos no centrales del vecindario se agregan como frontera. Al terminar de explorar una región, se pasa al siguiente punto no visitado del dataset.

**Puntos principales:**
- Proceso secuencial (nunca paralelo): cada paso depende del resultado del anterior.
- Puntos de borde ambiguos: pueden quedar asignados a un cluster u otro según el orden en que se procesaron.
- Al final, todo punto sin asignar queda etiquetado como ruido.

### Tema 4: Los dos hiperparámetros — ε y MinPts
ε define el tamaño del radio de vecindad; MinPts define cuántos vecinos se necesitan dentro de ese radio para considerar densidad suficiente. Ambos afectan directamente el resultado: ε muy chico genera mucho ruido y clusters diminutos; ε muy grande fusiona clusters distintos entre sí.

**Puntos principales:**
- ε se elige con el gráfico k-distance (codo "invertido").
- MinPts se elige con la regla $MinPts \geq D+1$, recomendado $MinPts \approx 2D$ (D = número de dimensiones).
- El valor adecuado está en un punto intermedio entre sobre-fragmentar y fusionar todo.

### Tema 5: DBSCAN es 100% multivariado
Al igual que K-Means, DBSCAN debe correrse sobre todas las variables originales, sin reducir dimensionalidad antes, salvo redundancia extrema entre variables — reducir primero implica perder información que puede impedir una clusterización óptima.

**Puntos principales:**
- La distancia entre vecinos debe calcularse con el vector completo de variables.
- Reducir dimensiones antes solo se justifica si hay variables muy redundantes.

### Tema 6: Ventajas, desventajas y cuándo usarlo
DBSCAN destaca en detección de outliers/anomalías y descubre clusters de forma arbitraria (no solo convexa), pero es sensible a la elección de ε/MinPts, pierde sentido en alta dimensionalidad, y no siempre asigna todos los puntos a un cluster (deja ruido fuera de cualquier estrategia).

**Puntos principales:**
- Usar DBSCAN cuando el objetivo es detectar anomalías (fraude, defectos).
- Usar K-Means cuando el objetivo es asignar una estrategia a **todos** los clientes/puntos, incluidos los atípicos.

---

## Clase 11 — Evaluación de algoritmos de clustering

### Tema 1: ¿Tiene sentido clusterizar estos datos?
Antes de aplicar cualquier algoritmo, hay que verificar si el dataset realmente tiene estructura agrupable; de lo contrario, el algoritmo va a "inventar" clusters artificiales, y actuar sobre ese hallazgo falso (en marketing, finanzas, manufactura) puede ser costoso. Esta evaluación es una etapa más del EDA.

**Puntos principales:**
- Métodos disponibles: Hopkins, VAT, Gap Statistic, análisis visual (PCA/UMAP).
- Evaluar clusterabilidad ahorra tiempo de cómputo y evita decisiones de negocio erróneas.

### Tema 2: Estadístico de Hopkins (repaso aplicado)
Compara la distancia de puntos reales contra puntos generados aleatoriamente en el mismo espacio: $H = \frac{\sum y_i}{\sum x_i + \sum y_i}$. H cercano a 1 = fuerte tendencia a clusterizar; H ≈ 0.5 = datos aleatorios, no vale la pena clusterizar.

**Puntos principales:**
- Regla práctica: $H \leq 0.5$ → no clusterizar.
- Es la versión estadística/objetiva de lo que el VAT muestra visualmente.

### Tema 3: Silhouette Score (índice de silueta)
Mide, para cada punto, qué tan bien asignado está a su cluster combinando distancia intracluster (a) e intercluster (b): $s = (b-a)/\max(a,b)$. Rango de −1 a 1; valores negativos indican que el punto está más cerca de otro cluster que del suyo.

**Puntos principales:**
- Umbral práctico de preocupación: **0.4**.
- Permite detectar individuos mal clasificados en zonas fronterizas entre clusters.
- Fusionar clusters muy pegados suele subir la silueta general por encima del umbral.
- Individuos mal clasificados: se resuelven re-clusterizando o reasignando manualmente con criterio de negocio.

### Tema 4: VAT y otros objetos visuales
El VAT persigue el mismo objetivo que Hopkins pero de forma visual: heatmap de la matriz de distancias reordenada, con bloques oscuros en la diagonal indicando estructura de clusters. El dendrograma y el gráfico de PCA con elipses de cluster cumplen una función de diagnóstico similar.

**Puntos principales:**
- VAT, dendrograma y PCA con elipses son complementos visuales a las métricas estadísticas (Hopkins, silueta).

### Tema 5: Calibración de hiperparámetros (hyperparameter tuning)
Ajustar los parámetros de un algoritmo de clustering (K, métrica de distancia, criterio de enlace, ε/MinPts) para obtener grupos compactos, separados e interpretables. A diferencia del aprendizaje supervisado, no existen librerías automatizadas (Grid Search) porque no hay una métrica de accuracy objetiva contra la cual comparar — solo se dispone de la silueta y las distancias intra/intercluster.

**Puntos principales:**
- En clustering, el tuning se hace con bucles manuales (fuerza bruta) probando combinaciones y maximizando la silueta.
- K-Means en scikit-learn no tiene hiperparámetro de métrica de distancia (siempre euclidiana).
- El enlace `ward` en clustering jerárquico solo es compatible con distancia euclidiana.
- La calibración correcta es crítica en aplicaciones de alto impacto (segmentación de clientes, medicina, detección de fraude).
