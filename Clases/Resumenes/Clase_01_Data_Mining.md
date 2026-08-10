# Clase 1 — Minería de datos aplicada a los negocios

**Curso:** Minería de Datos · Lead University
**Docente:** Rachit Vargas (MBA | Engineer)
**PDF fuente:** `Clases/PDFs/Clase 1 - Data Mining.pdf`

> Clase introductoria, sin programación. Fija qué es minería de datos, a qué problemas se aplica y con qué metodología se trabaja.

---

## 1. La analogía del minero

El proceso de minería de datos se estructura como la minería física (metáfora del video previo del **Dr. Aldemar Rodríguez**):

| Minería física | Minería de datos |
|---------------|------------------|
| Túnel / yacimiento | **Data Cleaning and Integration** — base de datos cruda |
| Excavadora | **Data Selection** — se extraen los datos objetivo (*target data*) |
| Mineral en bruto en la mano | **Data Transformation** — datos preprocesados y transformados |
| Oro refinado en barras | **Data Mining** — se obtienen los patrones |
| Joyería en vitrina | **Knowledge** — evaluación, presentación y conocimiento accionable |

### Flujo canónico (KDD)

```
Database → Data Cleaning and Integration → Target Data → Data Selection
→ Pre-processed data → Data Transformation → Transformed Data
→ Data Mining → Patterns → Evaluation & Presentation → Knowledge
```

El valor no está en los datos crudos sino en el conocimiento refinado al final: cada etapa descarta material sin valor y concentra el que sí lo tiene.

### El proceso en la práctica

**1. Entrar a la mina y decidir si hay valor.** El primer paso no es modelar, es identificar los *sources*: bases de datos, tablas que hay que unir con *joins*, y muchas veces Excel sin estructura. Las fuentes son sucias — columnas inconsistentes, duplicados, datos por validar. Es la tarea menos interesante y la más complicada, pero es la que determina si el problema es resoluble.

**2. Extracción y selección.** Solo si hay valor se extrae. No todas las variables son necesarias; elegir cuáles consumir es parte del trabajo.

**3. Transformación.** *Feature engineering*, tratamiento de nulos (descartar, imputar, o rellenar con algoritmos de ML) y **escalado**. El escalado importa porque *edad* e *ingreso* no viven en la misma escala y el modelo no debe "comparar peras con manzanas".

**4. Minería y conocimiento.** Se obtienen los patrones y se convierten en algo accionable.

### La última etapa es venta

*Evaluation & Presentation → Knowledge* — la joyería en la vitrina — no es adorno del diagrama. El resultado hay que **venderlo** a Junta Directiva o nivel C, con narrativa que entiendan perfiles técnicos y no técnicos.

Hay dinero real en juego: un modelo de fraude que no funciona significa clientes defraudados → pérdida de confianza → fuga de clientes → caída de participación de mercado → caída del EBITDA. Un hallazgo valioso mal comunicado no sirve de nada.

---

## 2. Minería de datos vs. estadística tradicional

| Aspecto | Estadística tradicional | Minería de datos |
|---------|------------------------|------------------|
| **Enfoque** | Explicar fenómenos, probar hipótesis, validar teorías | Descubrir patrones ocultos y predecir comportamientos |
| **Punto de partida** | Hipótesis formuladas previamente | Datos en bruto (exploración automática) |
| **Tipo de datos** | Muestras pequeñas y bien estructuradas | Grandes volúmenes, ruidosos / no estructurados |
| **Métodos** | Regresión, ANOVA, pruebas de hipótesis, intervalos | Árboles de decisión, clustering, redes neuronales, reglas de asociación |
| **Orientación** | Interpretación, causalidad, significancia estadística | Predicción, exactitud, utilidad práctica |
| **Herramientas** | R, SPSS, SAS, STATA | Python, R, SQL, Spark, Weka, TensorFlow, RapidMiner |

Matices sobre la tabla:

- **Punto de partida:** la estadística arranca de una hipótesis; la minería arranca de los datos brutos y usa **todos** los datos disponibles, no una muestra. Es el giro central entre ambas.
- **Métodos:** la **regresión es transversal** — se comparte entre estadística, minería y ML. Lo distintivo de la columna izquierda son las pruebas de hipótesis (t de Student, ANOVA, MANOVA) y los intervalos de confianza.
- **Herramientas:** **RapidMiner no se usa en el curso**, aunque figure en la tabla. El enfoque de ciencia de datos no es usar software ya hecho tipo SAS o RapidMiner (licencias caras, menos flexible), sino **programar**.

### Ubicación disciplinar

Diagrama de Venn: **Data Science** y **Artificial Intelligence** se solapan; **Data Mining** vive principalmente dentro de Data Science extendiéndose hacia IA, y **Machine Learning** ocupa la intersección de ambos círculos.

La diferencia práctica entre minería de datos y machine learning es menor — la minería es solo un poco más estadística — y los algoritmos son los mismos: PCA y K-means se usan en ambas ramas.

---

## 3. Casos de negocio

### VISA & Mastercard — detección de fraude

- **Problema:** pérdidas millonarias por transacciones fraudulentas.
- **Método:** modelos de **clasificación** que detectan patrones anómalos en segundos.
- **Resultado:** bloqueo casi en tiempo real de operaciones sospechosas.
- **Impacto:** reducción de fraudes y aumento de confianza de los clientes.

La confianza es lo que sostiene el volumen que mueve un banco o fintech — los ingresos que luego se colocan en préstamos. Menos fraude → más clientes → mejor margen.

### Marriott — segmentación de huéspedes

- **Problema:** personalizar la experiencia del cliente en miles de hoteles.
- **Método:** **clusterización** de reservas y consumos (duración de estadía, gasto en restaurantes/spa, tipo de habitación, motivo del viaje).
- **Resultado:** clústeres tipo viajeros de negocios, familias vacacionales, parejas de lujo, viajeros frecuentes de bajo costo.
- **Impacto:** marketing y programas de lealtad más efectivos → más ingresos por cliente.

La pregunta que responde el clustering: **¿cuál es el comportamiento de mis clientes y qué características comparten?** A partir de ahí se arma la oferta — a quien le gustan los masajes, estadía + spa; a quien le gustan los tours, hospedaje + tours a precio especial.

Las **tarjetas de crédito** aplican la misma lógica sobre el consumo: quien viaja y consume en el exterior es candidato a Visa Infinite, LifeMiles o Conect Miles; quien consume en Walmart, a la tarjeta de Walmart para acumular puntos. Muchas empresas **compran datos en el mercado** para enriquecer ese perfilamiento. Amazon y Netflix hacen lo mismo con vistas y clics: recomiendan según a qué otros usuarios registrados se parece cada persona.

### Coca-Cola — atribución de marketing

- **Problema:** justificar el gasto millonario en publicidad.
- **Método (estadística, no minería):** **regresión múltiple** sobre inversión en anuncios, promociones en punto de venta y variación de ventas por región. Canales evaluados: televisión, vallas publicitarias e influencers.
- **Resultado:** la publicidad televisiva es significativa en el **corto plazo**; las promociones en tienda impactan más en el **largo plazo**.
- **Impacto:** optimización del presupuesto → mejor ROI.

La regresión es el algoritmo indicado porque **explica el impacto de cada variable sobre la objetivo**: cuánto mueve la aguja pasar de 10 a 20 millones de inversión en un canal concreto. La decisión deja de basarse en intuición o números crudos y se apoya en **p-valor** y **R²**, que dicen si las conclusiones son significativas.

### Dilema predicción vs. explicabilidad

Los algoritmos se dividen en dos perfiles y hay que elegir:

- **Predictivos:** redes neuronales y modelos complejos. Precisos, pero **cajas negras** — es muy difícil saber cómo llegaron a su conclusión.
- **Explicativos:** regresión lineal y árboles de decisión. Muestran el paso a paso hacia el resultado. Aunque pronostiquen peor, son los que prefiere el nivel C.

> *"Si algo sale mal, nosotros no podemos dejarle la responsabilidad a la IA."*

La responsabilidad recae sobre quien implementó el modelo, no sobre el modelo.

> **Contraste deliberado del PDF:** los dos primeros casos son minería de datos (descubrimiento y predicción); el tercero es estadística clásica (explicación y causalidad). Aterriza la tabla de la sección 2.

---

## 4. CRISP-DM

Marco de trabajo estándar para proyectos de ciencia de datos. Es **cíclico**, no lineal: las flechas van en ambos sentidos entre fases adyacentes y el ciclo se retroalimenta.

```
Business Understanding ⇄ Data Understanding
                              ↓
                        Data Preparation ⇄ Modeling
                                              ↓
                                          Evaluation
                                              ↓
                                          Deployment
                                              ↺ (vuelve a Business Understanding)
```

### 1. Business Understanding

> *"The main objective is to ask the right questions, while the choice of the right techniques for a specific case is secondary"*

Lo primero es **hacer las preguntas correctas del negocio**; la técnica tiene menos prioridad. En un problema de fraude, la pregunta previa es qué interesa detectar y cuáles son los requerimientos reales.

**Entregables:** Business Objectives · Assess Situation (recursos, requerimientos, riesgos, costos y beneficios) · Data Mining Goals y sus criterios de éxito · Project Plan.

### 2. Data Understanding

> *"This phase aims to familiarize the data analyst with the data and to ensure that the data is suitable for the intended analysis"*

Su función real es determinar si el proyecto **es ejecutable o no**: con qué datos se cuenta, qué variables hay y cuáles importan. Si las variables no son concluyentes para el problema, **la inversión completa se pierde** — equipo, máquinas, ingenieros estandarizando variables. Caso típico: clientes que piden *forecasting* sin datos históricos.

Aquí la **ingeniería de datos** (data lakes, Databricks, arquitectura Medallion) funciona como **filtro de factibilidad**.

**Entregables:** Initial Data Collection Report · Data Description Report · Data Exploration Report · Data Quality Report.

### 3. Data Preparation

> *"Data preparation is often a lengthy undertaking […] but it is essential as a prerequisite to put data in context in order to turn it into insights and eliminate bias resulting from poor data quality"*

Objetivo único: **asegurar datos de calidad** — eliminar sesgo, tratar faltantes, homogeneizar escalas. Es la fase más costosa en tiempo.

**Entregables:** Rationale for Inclusion/Exclusion · Data Cleaning Report · Derived Attributes · Merged Data · Reformatted Data.

### 4. Modeling

> *"This phase selects and fine-tunes appropriate techniques for the best model […] Model accuracy is important, but understanding feature impact is also necessary for decision-making"*

Es la fase que descubre **algo que nadie más en la empresa había visto**. Ejemplo: marketing asumía **6 clústeres** de clientes; al correr K-means resultaron **3**. Se invirtió presupuesto publicitario en atraer segmentos que no existían.

Condición de entrada: **solo entra data limpia y valiosa**.

> **GIGO — *garbage in, garbage out*:** *"Si yo a un modelo le meto basura, me va a tirar basura."*

Ninguna supercomputadora ni meses de entrenamiento arreglan datos malos. De ahí que importen los tests estadísticos y algoritmos como regresión lineal y árboles de decisión: **indican qué variables son realmente importantes**.

**Entregables:** Modeling Technique y sus supuestos · Test Design · Parameter Settings y Models · Model Assessment.

### 5. Evaluation

Las métricas dependen del tipo de análisis:
- **No supervisado / clustering:** Silhouette y similares.
- **Supervisado:** precisión, F1-score, AUC.

**Entregables:** Assessment of Results w.r.t. Business Success Criteria · Review of Process · Next Steps.

### 6. Deployment

El modelo queda alojado en un sistema, integrado operacionalmente con los sistemas de la compañía y recibiendo datos automáticamente vía API. Es el objetivo último de todo el trabajo previo.

> **~70 % de los proyectos de machine learning nunca llega a deployment.**

Causas: sistemas incompatibles, restricciones de seguridad, problemas de *setup*. Pero la más frecuente es que **los datos se quedan cortos frente al problema**: el proyecto llega hasta *Evaluation*, los resultados no son buenos y ahí se detiene. La salida es volver a minar datos y explorar nuevos *sources*, lo que hace determinante la **comunicación entre ingeniero de datos e ingeniero de ML**.

**Entregables:** Deployment Plan · Monitoring and Maintenance Plan · Final Report y Presentation · Experience Documentation.

> **Puntos clave:** (a) el negocio manda, la técnica es secundaria; (b) *Data Preparation* es la fase más costosa; (c) el proceso itera, no termina en Deployment.

---

## 5. Caso: discriminación de género (UC Berkeley)

Caso clásico de admisiones de posgrado en UC Berkeley. La universidad publicó la lista de admitidos de una facultad y varios periodistas notaron que se admitían muchos más hombres que mujeres. La acusación pública fue **discriminación de género**.

| Departamento | Hombres solicitantes | % Admitidos H | Mujeres solicitantes | % Admitidas M |
|--------------|---------------------:|--------------:|---------------------:|--------------:|
| A | 825 | 62 % | 108 | 82 % |
| B | 560 | 63 % | 25 | 68 % |
| C | 325 | 37 % | 593 | 34 % |
| D | 417 | 33 % | 375 | 35 % |
| E | 191 | 28 % | 393 | 24 % |
| F | 373 | 6 % | 341 | 7 % |
| **Total** | **2 691** | **45 %** | **1 835** | **30 %** |

### El hallazgo

- **Agregado:** 45 % de admisión para hombres vs. 30 % para mujeres. Mirando solo esos dos números, parece discriminación evidente.
- **Por departamento:** en 4 de 6 (A, B, D, F) las mujeres tienen tasa **igual o mayor** que los hombres. Solo en C y E los hombres superan, y por 3-4 puntos porcentuales.

### La explicación — Paradoja de Simpson

La agregación invierte el signo de la relación:

- **A y B son poco competitivos** (62-63 % de admisión) y reciben **casi solo hombres** (825 y 560 vs. 108 y 25).
- **C, E y F son muy competitivos** (6-37 %) y reciben **mayoría de mujeres** (593, 393 y 341).

Las mujeres se postulaban desproporcionadamente a los departamentos **más difíciles de entrar**. El *confounder* es el **departamento**, que determina a la vez el género del postulante típico y la probabilidad de admisión.

Los **pesos relativos** son el aspecto crítico: no es lo mismo calcular sobre 825 personas que sobre 108 — cada individuo impacta el porcentaje total de forma muy distinta, así que el agregado está dominado por los departamentos con más solicitantes de un género.

### Lección metodológica

> Un resultado agregado puede contradecir por completo los resultados de cada subgrupo. **Nunca concluir causalidad desde datos agregados sin desglosar por las variables de estratificación relevantes.**

Conecta con **Data Understanding** de CRISP-DM: explorar y verificar la estructura de los datos antes de modelar o concluir.

---

## 6. Fuera del PDF — logística, tareas y hoja de ruta

### 6.1 Metodología del curso

- Exclusivo para **Ciencia de Datos**. Existe un curso paralelo para Ciencias Empresariales, con contenido distinto y sin programación.
- **Aula invertida (modelo INCAE):** el material se revisa *antes* de la sesión. Hay un solo recurso por tema en el campus, deliberadamente, para que sea leíble.
- **15 clases, un tema por clase.** No se mezclan dos temas pesados en una sesión — PCA y t-SNE nunca van juntos porque exigen matemática distinta. Habrá clases más cortas que 3 horas.
- **~90 % de las clases involucran programación.** Solo las dos primeras funcionan con video previo (~46 min) más sesión en vivo.
- **Python, no R.** Se explota el trabajo con clases y métodos: **saber POO es fundamental** para aprovechar la parte algorítmica.

### 6.2 Evaluación

| Componente | Peso | Detalle |
|------------|-----:|---------|
| **Tareas cortas** | 40 % | 13 tareas, ≈ una por clase. Sin fecha límite estricta ni penalización por atraso; entregables hasta final de cuatrimestre. No conviene acumularlas — algunas son voluminosas (PCA tiene muchos apartados). |
| **Quizes** | 10 % | 5 quizes aleatorios, **al final** de una sesión; pueden cubrir clases anteriores. |
| **Participación** | 30 % | Preguntas y aportes. |
| **Examen final** | 20 % | **Formato entrevista**: ~5 preguntas al estilo de entrevista de trabajo real ("qué hace el PCA", "cómo se usa el K-means"), sin apoyo de herramientas. |

**IA generativa:** uso **mínimo**. El requisito real es poder **entender y explicar** el código entregado; si se detecta uso intensivo sin comprensión, la tarea puede anularse.

### 6.3 Hoja de ruta

Introducción a métodos no supervisados y tests estadísticos → **PCA** → **ACS/ACM** → **t-SNE y UMAP** → introducción a clustering → **clustering jerárquico** → **K-means** → **DBSCAN** y modelos de mezcla → **evaluación de clustering** → **detección de anomalías** con árboles de decisión → **reglas de asociación** → **cadenas de Markov** → examen final.

> **Minería de Datos 2** cubre aprendizaje **supervisado** y series de tiempo: Holt-Winters, SARIMA, SARIMAX, Facebook Prophet.

### 6.4 Adelantos conceptuales

Temas introducidos al recorrer el syllabus, sin diapositiva propia en este PDF.

**Supervisado vs. no supervisado.** El curso es 100 % no supervisado.
- *Supervisado:* existe una **etiqueta** o variable objetivo que hay que explicar. Ejemplo: predecir el precio de venta de una casa a partir de habitaciones, metros cuadrados, vecindario, terraza.
- *No supervisado:* no hay etiqueta ni nada que predecir. Ejemplo: entender el comportamiento de compra de distintas personas — visualizar, explorar y agrupar para ver quiénes comparten características. Es esencialmente EDA extendido.

**Para qué sirven los tests estadísticos.** Deciden si una variable es "de peso" **antes** de modelar. En fraude, el salario (una transacción de 100 000 dólares con salario de 1 000) y la zona horaria son discriminantes; el "gusto musical" no lo es, aunque muchas empresas lo recolecten. En problemas complejos hacen falta **t de Student, ANOVA, MANOVA** y **chi-cuadrado**. Cargar un modelo con variables sin poder discriminante solo hace que aprenda ruido.

> "Correlación" aplica **solo entre variables numéricas**. Para asociación entre dos variables categóricas el término y la herramienta correctos son **chi-cuadrado**.

**Por qué reducir dimensiones.** La visión humana llega a **3 dimensiones**; con decenas o cientos de variables es imposible graficar un panorama completo. Hay que bajar a 2-3 dimensiones **sin perder la mayor cantidad de información** para ver patrones, clústeres y qué individuos se parecen.
- **PCA:** variables **numéricas**, problemas **lineales**. Sirve además para datasets con **multicolinealidad**.
- **ACS / ACM:** lo mismo para variables **categóricas**.
- **t-SNE y UMAP:** problemas **no lineales**, que son la mayoría de los casos reales. La ventaja profesional está ahí — mucha gente aplica PCA sin verificar normalidad ni linealidad.

**Sobre clustering.** Clustering jerárquico y K-means persiguen el mismo objetivo con métodos distintos (distancias euclidiana, Manhattan, Chebyshev; saltos mínimos y máximos; centroides); DBSCAN lo hace por densidad. Punto crítico: **no todos los datos son clusterizables**, y hoy mucha gente clusteriza sin comprobar con tests si tiene sentido.

### 6.5 Tarea 1

El caso de Berkeley (sección 5) es la **Tarea 1**, deliberadamente sencilla y sin matemática del curso.

- **Herramienta libre:** Excel, Python, notebooks, calculadora gráfica. *"No trates de matar a una hormiga con una bazuca"* — se resuelve con promedios e interpretación, sin ML ni IA generativa.
- **Formato:** **en prosa**. Se pueden agregar números, tablas o análisis de apoyo, pero la respuesta va redactada.
- El enunciado del campus incluye una **segunda tabla**: departamento, solicitudes totales, tamaño del cupo, **relación de demanda** y **punto de corte**.

**Preguntas:**
1. El agregado de la tabla 1 sugiere desventaja para las mujeres. **¿Se sostiene la conclusión al controlar por departamento?**
2. Con la tabla 2 (relación de demanda y punto de corte) y los patrones de postulación por género, **¿qué hipótesis explica la brecha agregada?**
3. En el rol de **Laura Méndez**, **¿qué intervenciones propondría** en atracción, orientación y asignación de entrevistas para equilibrar el mix de aplicaciones **sin sacrificar estándares?**

---

## Conceptos clave

- **KDD:** limpieza → selección → transformación → minería → patrones → conocimiento. Cada etapa descarta lo que no vale.
- **Minar datos es ensuciarse las manos:** el trabajo empieza en fuentes sucias y no estructuradas, no en el modelo.
- **Minería ≠ estadística tradicional:** parte de los datos y no de la hipótesis, usa todos los datos y no una muestra, prioriza predicción sobre explicación.
- **Minería ≈ machine learning:** comparten algoritmos; la minería es solo un poco más estadística.
- **CRISP-DM:** 6 fases cíclicas, con el entendimiento del negocio como punto de partida obligatorio.
- **La técnica es secundaria; la pregunta correcta es lo primero.**
- **Data Understanding decide la factibilidad:** si los datos no alcanzan, el proyecto entero es inversión perdida.
- **GIGO:** datos basura producen modelos basura; ninguna capacidad de cómputo lo compensa.
- **El cuello de botella es el deployment:** ~70 % de los proyectos no llega, casi siempre porque los datos se quedan cortos.
- **Predicción vs. explicabilidad:** se suele preferir un modelo que explique sobre uno que solo acierte, porque la responsabilidad no se delega a la IA.
- **El resultado hay que venderlo:** sin narrativa para nivel C, un hallazgo valioso no se convierte en decisión.
- **Paradoja de Simpson:** el agregado puede invertir el signo de todos los subgrupos; estratificar antes de concluir.

1:01:23
1 hora 1 minuto 23 segundos
Lo que sí nos permiten a nosotros es saber si algo real o es algo producto al azar o k trabaja con pequeñas muestras, muchachos, eso tal vez es una desventaja, claramente, pues hoy por hoy trabajamos con Big Data, no trabajamos con datos pequeños. Entonces esto es parte de una desventaja y es un test paramétrico. Chicos, esto también es bastante importante. ¿Alguien sabe qué es paramétrico y no paramétrico, muchachos o K?
1:01:50
1 hora 1 minuto 50 segundos
Chicos, esta parte paramétrico y no paramétrico. Esto siempre lo van a ver en muchos de los modelos algorítmicos de de de Machine Learning. Todos los modelos siempre van a decir eso. Es un modelo que trata sobre datos paramétricos y hay otros. Hay otros algoritmos de Machine Learning que funciona muy bien con datos que no son paramétricos. Y esto es chicos, los datos paramétricos son datos que siguen una distribución normal y datos no paramétricos son.
1:02:19
1 hora 2 minutos 19 segundos
Datos que no siguen para nada. Ninguna distribución normal puede ser una distribución sesgada. Lo ven entonces ustedes van a escuchar, por ejemplo chicos, que la regresión lineal leaner agreation es un es un algoritmo paramétrico, modelo paramétrico y que depende de que los datos sigan una distribución normal. Si no, los resultados que ellos vayan a dar no van a ser como los mejores. O sea, ellos se van a desempeñar bastante bien.
1:02:49
1 hora 2 minutos 49 segundos
Cuando los datos son completamente siguen una distribución normal o k esa es la diferencia entre paramétricos y no paramétricos chicos y aquí es ahora sí donde donde entra, pues el test estadístico o el test student se lo oye muchachos, nosotros vamos a tratar de cuantificar, bueno esto ahorita este este valor, perdón este valor.
1:03:17
1 hora 3 minutos 17 segundos
Bueno, esta esta IGX, solamente hice valor, pero este puede ser cualquier cosa, chicos, esto puede ser salario, esto puede ser, no sé tipo de cambio, esto puede ser margen bruto, esto puede ser, no sé, conversión de clics, etcétera, etcétera. En este caso solamente hice valor chicos, entonces esto es algo más genérico o k.
1:03:45
1 hora 3 minutos 45 segundos
Entonces muchachos, lo que nosotros ocupamos a ver de acá es que por ejemplo para estos grupos, nosotros debemos saber si estos grupos son diferentes entre ellos, entonces vean que acá nos dice que el Grupo B es esta distribución que tenemos por acá y el Grupo a es esta como como marrón, como roja, algo así. Entonces chicos, si ustedes ven acá las medias son bastante similares. Bueno, este p value.
1:04:14
1 hora 4 minutos 14 segundos
Me dice 0.75, y si ustedes vieron en estadística, chicos casi, que la mayoría de esos tests unir variados y el regresión y demás el criterio de selección es que sean menor a 0.05 o K entonces vean que acá 0.75 el test student está confirmando de que estas medias son completamente iguales.
1:04:40
1 hora 4 minutos 40 segundos
Aquí no hay separación alguna. Esos estos dos grupos no los separa para nada. Esta variable que se llama valor son iguales, eso es lo que nos quiere decir eso, pero vean chicos que acá el Grupo CY el Grupo D Sí están separadas. Ven gráficamente vemos que el Grupo C, que es esta que tenemos por acá, está un poquito más separada que el Grupo DY el test estadístico de este Studio nos está diciendo que el P value es 0.000 menor a 0.5.
1:05:10
1 hora 5 minutos 10 segundos
Cero, bueno, 0.05, entonces chicos, eso va a ser crucial porque va a ser crucial, muchachos, porque por ejemplo, algo bastante simple, no sé, vamos a poner la relación entre cáncer y no cáncer si alguien tiene cáncer o no cáncer y por ejemplo, vamos a poner dos grupos, personas que familiarmente o sea que tienen una herencia.
1:05:41
1 hora 5 minutos 41 segundos
En su familia, personas que han tenido cáncer y no herencia, entonces vean que acá el Grupo D pueden ser personas que tengan cáncer y aquí en el Grupo C pueden ser personas que no tengan cáncer y el valor puede ser la cantidad de familias bueno y cantidad de familiares claramente aquí 60 y eso no tiene sentido, vamos a poner entre 0 y 5, entonces vean que acá el Grupo D.
1:06:05
1 hora 6 minutos 5 segundos
Entre más personas, entre entre más la cantidad de personas que ellos tengan familiares que tienen cáncer o que han tenido cáncer, es más probable que ellos tengan cáncer, que por ejemplo, las personas que no tienen tantos familiares con ese problema y que no tengan cáncer, lo hay. Entonces las medidas están completamente separadas, entonces ellos son, o sea, esta variable.
1:06:29
1 hora 6 minutos 29 segundos
Esta esta parte de la herencia, la variable de de de saber si tienen una una herencia familiar o no con respecto a esta enfermedad es crucial para saber si el día mañana una persona va a sufrir de cáncer o no va a sufrir de cáncer. Entonces, por ejemplo, Braulio y Daniel, ustedes que son ingenieros de datos, ya saben que por ejemplo, cuando están extrayendo datos y un científico de datos les dice, mira, dame datos que sean valiosos, por ejemplo, esta puede ser una data valiosa y se están trayendo.
1:06:57
1 hora 6 minutos 57 segundos
¿Datos en relación a a cáncer o no cáncer, verdad? ¿Entonces, por qué se los digo a esto chicos? Porque a veces muchos de stakeholders les les piden a ustedes hacer un modelo predictivo, pero eso es un modelo predictivo. Todas las variables con respecto a lo que están tratando predecir se comportan de esta forma. Entonces no hay ninguna variable que sea significativamente estadística en donde ustedes se puedan separar. ¿Qué es lo que provoca que una persona tenga cáncer o no tenga cáncer?
1:07:26
1 hora 7 minutos 26 segundos
O qué es lo que provoca que una persona vaya a caer en un problema de mora o no mora con respecto al Banco o k entonces chicos, todas estas pruebas estadísticas univariadas porque esto es completamente univariadas, es decir, solamente estoy analizando una variable con respecto a un grupo, son bastante importantes muchachos, para que ustedes apliquen 1 e d a y sepan cuáles son las variables que tienen impacto y cuáles no. Recuerden chicos garbaching, garbach o sea lo que yo le estoy metiendo al modelo si es basura.

1:07:56
1 hora 7 minutos 56 segundos
El modelo más basura o k o k chicos luego está bueno. Esta es la parte de la fórmula matemática. Tenemos el promedio x 1 con respecto al X dos y aquí luego tenemos las desviaciones estándar. Bueno, esa es una parte matemática más, pero el test student se basa sobre esta parte matemática y como les digo o sea yo agarro la media de la distribución 1. La media de la distribución dos.
1:08:25
1 hora 8 minutos 25 segundos
Y luego pues agarro las desvisiones combinadas para saber pues si es algo producto de la Lateridad, pues sí, definitivamente es algo estadístico y vean chicos que también agarro el tamaño de la muestra muy importante, pero bueno, eso es la parte de la parte más de fórmula matemática, pero lo que a nosotros nos interesa saber o nos interesa interpretar es este, este privalio, este privalio y claramente pues estos gráficos porque la gente claramente las personas que no.
1:08:52
1 hora 8 minutos 52 segundos
¿Que no son estadísticos o matemáticos van a entender más este gráfico que por ejemplo, un pvali? Daniel Bueno, entonces, si usted pudiera como decirlo de una forma simplificada, el test touch se utilizaría. ¿Para qué? O sea, sí, sí estoy entendiendo todo, pero como para poder sintetizar en qué escenarios es que esto es es bien aplicable. ¿Cómo lo podríamos sintetizar todo esto ahí, Daniel?
1:09:20
1 hora 9 minutos 20 segundos
¿El test toint tiene varias aplicaciones, verdad? Vamos a partir de ahí. O sea, eso depende de qué, en qué contexto querés utilizarlo. Pero para nuestro contexto, Daniel, el test student y todas estas test estadísticos que que que vamos a nosotros empezar a aplicar, nos va a servir a nosotros para saber si hay variables que están fuertemente bueno tanto correlacionadas o que tienen impacto sobre alguna otra verdad, entonces.
1:09:47
1 hora 9 minutos 47 segundos
¿Si, como te digo, si nosotros estamos tratando de evaluar si una persona, por ejemplo, tiene cáncer o no, tenemos que encontrar cuáles son aquellas variables que explican que esa persona tenga cáncer o k por qué? Porque si nosotros, por ejemplo, le metemos, no sé, imagínate que tengamos 10 variables y esas variables nosotros se lo metemos al modelo a ojos cerrados. Nosotros no sabemos si esas variables son significativamente estadísticas con respecto a esa variable objetivo y nosotros estamos metiéndole absolutamente todo. Y nosotros decimos, mira, es que qué raro si el modelo.
1:10:16
1 hora 10 minutos 16 segundos
Le metí 10 variables y el modelo de la curras y me dio, no sé un 0.04. O sea, esto no sirve para nada, el modelo no es capaz de predecir nada y cuando nosotros, por ejemplo, vamos a analizar esto, vamos a ver que la mayoría de los de los de las variables lucen, por ejemplo, como esta distribución en donde la media de las diferentes variables con respecto a juzgar si una persona tiene cáncer o no se comportan completamente igual.
1:10:45
1 hora 10 minutos 45 segundos
¿Entonces el modelo no tiene la capacidad para poder entender cuándo clasificar, si una persona tiene cáncer o no tiene cáncer o K entonces lo que me permite eso? Daniel, aunque este curso de aprendizaje no supervisado, vamos a ver temas de exploración de datos. Y esos tests estadísticos son parte de la exploración de datos. Es parte de la estadística que esto funciona más claramente para efectos de aprendizaje supervisado, porque vamos a tener los algoritmos de Machine Learning.
1:11:14
1 hora 11 minutos 14 segundos
KNN, árboles de decisión, catboosting, entre otros. Pero nosotros vamos a partir de aquí, vamos a partir. Mira, yo tengo esta variable, esas variables que definitivamente sí son, digamos, significativamente estadísticas para poder saber si una una clase, por ejemplo, es bastante buena, que me que me que me hace una separación entre las clases. Cáncer no, cáncer.
1:11:40
1 hora 11 minutos 40 segundos
O definitivamente es una variable que no importa absolutamente nada en el problema o K entonces Daniel, Esa, digamos, es como como como la respuesta. O sea, nosotros estamos utilizando esto para saber si estas variables me explican algo. No me están explicando absolutamente nada con respecto a una variable objetivo perfecto, listo o k.
1:12:02
1 hora 12 minutos 2 segundos
¿Entonces chicos no solamente bueno vamos a ver esto es esto es como se utiliza en Python, pero ya ahorita lo vamos a ver chicos esto no, no se preocupen esto vamos a ver esto son como ventas y queremos saber si las ventas a y las ventas B son iguales, entonces ven este es el Grupo 1 ventas a tengo diferentes ventas, esto puede ser en dólares, no sé, vamos a poner que eso son dólares y tengo ventas B utilizo saipi en Python, utilizo test test Int aquí la puedo, la pongo en práctica, lo ven?
1:12:31
1 hora 12 minutos 31 segundos
El Ty el P value, esa es la prueba Ty ese es el P value de la prueba t llamo a este módulo test Int le paso ventas a y ventas B que son los dos grupos, y lo que hago checos es solamente imprimir el valor y ponerle este esta condicional. Si el P value es 0.05, entonces hay diferencia estadística. La estrategia B mejoró las ventas y el y el.
1:12:56
1 hora 12 minutos 56 segundos
Y el dos, o sea, si es mayor a a 0. 5, no hay diferencias estadística entre las variables. Entonces chicos, vean que acá el test estudio me me dio resultado. el P value es 0.001. ¿Entonces quiere decir que sí hay una diferencia estadística? ¿La estrategia B es mejor que la estrategia a o K? Vamos a ver.
1:13:19
1 hora 13 minutos 19 segundos
O K chicos, esto para explicarles un poco tal vez esto, esto es más una prueba, una prueba. ¿AB testing, eso parece una una prueba invitesting por el, por el, por el, digamos por por cómo se llama esto, por por la comparativa que está haciendo la comparativa, sí, pero cómo se llama esto en Python? El Print, la impresión y máquina, la impresión del resultado o k ya ahorita vamos a ver eso chicos, pero eso es lo que quiero que ustedes vean.
1:13:48
1 hora 13 minutos 48 segundos
Ese test anteriormente que yo hice fue para poder detectar dos prototipos, que esto normalmente se llama la prueba IV Testing. No sé si la han escuchado, o sea las han pedido en sus empresas. Yo sí la yo sí lo he escuchado, pero pero más que nada para partes de marketing. Ajá. Marketing exacto, se utiliza mucho para marketing, entonces chicos, por ejemplo, vean que digamos si ustedes son diseñadores web y como dice Daniel son son parte de un equipo de marketing.
1:14:18
1 hora 14 minutos 18 segundos
A ustedes les piden diseñar dos tipos de páginas webs, el diseño a y el diseño b bueno, entonces en el en el diseño a prueban con 500 usuarios, o sea durante un mes, por ejemplo, y el y el diseño b prueban con otros 500 usuarios en Ah, bueno, son 3 semanas, perdón, por ejemplo, durante 3 semanas prueban con 1000,10 1000 usuarios lo parten y que y que objetivamente.

1:14:44
1 hora 14 minutos 44 segundos
5000 usuarios van a consumir el diseño a y 5000 usuarios van a diseñar también el van a van a consumir el diseño B o k entonces lo que nosotros queremos saber es cuál diseño tiene una mejor conversión en clics o la conversión de bueno si el ratio de la conversión.
1:15:06
1 hora 15 minutos 6 segundos
Entonces chicos ellos posiblemente lo que vayan a hacer es durante 3 semanas ponen a correr ese diseño b con estos con estos 505000 usuarios y los resultados posiblemente les vayan a decir que solamente lograron convertir un 2% o KY posiblemente ya la próxima. Los próximos 3 semanas ponen a correr el diseño b igual con con 5000 usuarios y la conversión les dió un 3%.
1:15:35
1 hora 15 minutos 35 segundos
Entonces saca un test estadístico para saber si ese 1, ese 1. Porcentual, es estadísticamente significativo para decir mira, Este diseño b es mejor que el diseño a o k entonces chicos, toda esta parte del testing también se las pruebas de de hipótesis y test estadísticos se utilizan también para Ivy testing. Eso es lo que les estaba mencionando antes, chicos, eso queda claro.
1:16:07
1 hora 16 minutos 7 segundos
Sí, profe o k chicos. Anova vamos a poner bueno, vamos a contextualizar Anova. Anova dice que permite comparar 3 o más medias. A diferencia de test student test Student solamente permite dos medias. Si lo vieron, solamente tenía dos.
1:16:32
1 hora 16 minutos 32 segundos
Dos distribuciones solamente tenía dos. Vean que acá la nueva me permite 3 variables o 3 grupos usando varianzas. Vean que aquí en vez de utilizar la mail utiliza la varianza para determinar si al menos 1 de ellas es significativamente diferente o K aquí se permite saber si la diferencia observada y real a los productos las aves es exactamente lo mismo.
1:16:56
1 hora 16 minutos 56 segundos
¿Y el test student? Vean aquí vean aquí chicos es la mayor, la la mayor diferencia. El test student compara medias y el la nueva compara varianzas entre las medias 1 compara 1 usa medias y el otro utiliza varianzas. Es un test panorámico paramétrico. También chicos y vean que acá estoy utilizando una variable ventas promedios, pero vean que aquí viene la diferencia. Estoy utilizando 3 grupos.
1:17:25
1 hora 17 minutos 25 segundos
El test estudio no me permite eso. Entonces chicos, vean que por ejemplo acá la diferencia es significativa. O sea, si ustedes ven el gráfico, la estrategia B tiene por por por defecto. O sea, si usted lo ve en el en el gráfico, esta estrategia B tiene mayor ventas promedio. ¿Lo ven? La estrategia a no tiene tantas ventas.
1:17:52
1 hora 17 minutos 52 segundos
¿Y la estrategia B? Digamos que están como en el en el medio esas dos estrategias. Entonces, chicos con un test test de Anova, nos puede decir a nosotros si dentro de esas 3 estrategias hay diferencias o no hay diferencias o K es exactamente lo mismo, pero vean que estoy utilizando más grupos.
1:18:22
1 hora 18 minutos 22 segundos
O K chicos, esto es bastante similar a lo que les estaba mencionando antes, en vez de utilizar textudent, vean que ahora igual con saipi perdón igual con saipi utilizó F On White hago 3 grupos distintos, a, B, c implemento igual. De la misma forma ese ese test, ese test estadístico y el criterio evaluación es exactamente el mismo, si está por cero está por debajo de 0.05.
1:18:52
1 hora 18 minutos 52 segundos
Hay diferencias estadísticas entre las estrategias de ventas, sino no hay diferencias siempre chicos, hermanova, muchachos, hermanova es una extensión de la Nova, pero a a diferencia de solamente ver una variable, si ustedes vieron si vieron antes muchachos, solamente estamos viendo ventas promedio. Ven, estamos viendo solamente una variable con varios grupos.
1:19:20
1 hora 19 minutos 20 segundos
Y seguimos antes también con el Anova, con el test student. Disculpen el test student también solamente miraba un valor, que era el bueno. El valor se llama con dos grupos, solamente una variable a la vez. ¿Hermanova Chicus voy a ver si lo puedo adelantar desde acá Hermanova chicus vamos a ver aquí hermanoa la diferencia de esto?
1:19:48
1 hora 19 minutos 48 segundos
¿Es que p me permite a mí analizar más de 2 o más de una? Sí, digamos que más de una digamos que dos o más variables cuantitativas, esa es la diferencia y más grupos también. Entonces muchachos, esa es la la la diferencia más grande mano a solamente una variable, osea solo 1 x 1 mano a x 1 x 2 hasta XN, sólo que me permite a Nova Manova.
1:20:17
1 hora 20 minutos 17 segundos
Es un test fono arámico también sensible a violaciones de normalidad. Bueno, eso es lo mismo que un test fono paramétrico nos quiere decir exactamente lo mismo. ¿Y aquí chicos? Bueno, aquí me salió primero el Código, pero lo que le quería demostrar es el gráfico. Vean chicos que aquí yo tengo dos categorías, droga X droga y pero vean que yo tengo acá 3 variables.
1:20:46
1 hora 20 minutos 46 segundos
La presión de la sangre, el colesterol y el herrated. Bueno, tal vez está como al revés del gráfico. Me parece como que está al revés, porque antes lo que habíamos visto es por medio de la etiqueta, los grupos, el grupo XY, el Grupo y, pero aquí está como acomodado de forma diferente en vez de de de de graficarlo por Grupo se ve como como como por cada variable. ¿Cómo se comporta el Grupo entonces? ¿Chicos, lo que ahora hace el manova?

1:21:16
1 hora 21 minutos 16 segundos
Es como saber, es como en vez de hacerlo uno por uno. Lo que hace el Manova es saber, por ejemplo, si estas 3 variables con respecto a los grupos XYY alguna de esas 3 variables, el colesterol, la presión de la sangre o el h, r a son significativamente diferentes en algún punto o k él puede decir mira, es que el colesterol es una variable que es muy diferente entre los grupos, es decir, el Grupo EX.
1:21:44
1 hora 21 minutos 44 segundos
Se caracteriza más por tener colesterol alto que por ejemplo, la droga. Y él solamente les va a decir, mira, entre esos 3 grupos hay una diferencia significativa, hay una diferencia ahí que es importante. O K profe, pero él él te dice que hay una diferencia en y te dice en cuál variable o simplemente te dice que hay una diferencia significativa.
1:22:12
1 hora 22 minutos 12 segundos
Entre los grupos solamente te dice una variable, solamente te dice que hay una diferencia de Daniel, eso es eso, es muy buen punto chicos, porque como o sea como lo estaba viendo, tal vez yo estoy señalando que los colesterol es la variable que que diferencia entre ellos. Pero como dice Daniel, la única forma de poder detectar a estos chicos es visualizándolo la prueba testudent, perdón, la prueba en mano que es esta que está por acá, solamente nos va a dar este resultado, nos va a decir.
1:22:42
1 hora 22 minutos 42 segundos
Por ejemplo, vean el valor F, el valor value, el número de observaciones entre otros, y solamente nos va a dar el el, el P value y la varianza y demás para interpretar, pero no nos va a decir en qué variable está la diferencia. Eso es muy, muy buena percepción. Daniel, eso es muy, muy eso está esa. Esa observación es bastante buena.
1:23:07
1 hora 23 minutos 7 segundos
O K chicos y luego está el chi cuadrado aquí voy un poquito rápido muchachos, porque creo que estoy sobre la hora bueno, no tanto, pero igualmente todavía hace falta ver. Todavía hace falta que veamos el código chicos y luego vean que nosotros solamente hemos estado viendo test estadísticos cuantitativos, solamente numéricos. Qué es lo que pasa chicos, si por ejemplo.
1:23:38
1 hora 23 minutos 38 segundos
Yo tengo dups dan que vimos por ejemplo ventas con respecto al Grupo a y el Grupo y el Grupo e, pero ventas es una variable cuantitativa con respecto a sus grupos, que pasa chicos, si por ejemplo.
1:24:06
1 hora 24 minutos 6 segundos
Vamos a poner los grupos que compraron el grupo a y el Grupo que no compraron, por ejemplo, y estoy poniendo la cantidad. Bueno, vamos a poner solamente una categoría. ¿Hizo clic o no hizo clic? Bueno, esto es como una especie de tabla dinámica en donde yo tengo el Grupo a.
1:24:32
1 hora 24 minutos 32 segundos
Las personas que hicieron clic, no sé, vamos a poner 10, las personas que no hicieron clic vamos a poner dos, las las personas que hicieron clic vamos a poner dos y las personas que no hicieron clic vamos a poner 50, por ejemplo. Bueno, entonces vean que esto es una asociación entre 2 variables categóricas, hizo clic, no hizo clic y el grupo a y el Grupo B el Grupo a son los que compraron. El Grupo a son los que no compraron si estuviésemos hablando en ecommerce.
1:25:02
1 hora 25 minutos 2 segundos
¿O K, entonces chicos, por ejemplo, acá la relación es clara, verdad? ¿Las personas que hacen clic son las personas que están más asociadas a las personas que compran y las personas que no hacen clic son las personas que están asociadas a que no compran, algo bastante obvio, verdad? Pero vean que son es una relación de 2 variables categóricas, cualitativas, no cuantitativas. ¿Cómo cuantificamos eso? ¿Cómo cuantificamos si hay una relación entre esas dos categorías? Esa es la pregunta.
1:25:32
1 hora 25 minutos 32 segundos
El chi cuadrado es el test estadístico que nos permite eso o K entonces dos variables categóricas están relacionados o no están o no son o son independientes. ¿Por ejemplo, muchachos, si yo tengo una variable variable a, variable b, variable c, variable a, variable b, variable c Yo aquí por ejemplo puedo hacer una, puedo hacer un gráfico de correlación?
1:25:59
1 hora 25 minutos 59 segundos
¿Verdad? ¿En donde este meiga bueno, aquí me hace falta 1, este con a, con a siempre es 1, este es siempre es 1, este es siempre es 1, verdad? Entonces el B me puede decir el B con el a puede tener una correlación del 9 del 0.9, el a con el C Puede tener una correlación del 0.5, el B me puede decir con respecto a a es lo mismo el 0.9, el C con el a vamos a poner que 0.4 y aquí el B con el C Es exactamente.
1:26:29
1 hora 26 minutos 29 segundos
Ah, no, aquí no le he puesto 0.4 y 0.4, digamos aquí hay un valor que se me está yendo, es este 0.5 le puse acá 0. ¿ entonces chicos, vean que esto es un gráfico de correlación, pero ese gráfico de correlación, cuál es el requisito para poder hacerlo? Que solamente las que las variables que estoy analizando sean numéricas, cierto, o sea un gráfico de correlación, una matriz de correlación solamente me funciona con variables numéricas.
1:26:58
1 hora 26 minutos 58 segundos
¿Si estoy tratando de de cuantificar variables categóricas, osea si estoy tratando de de de saber si hay una relación entre variables categóricas, no puedo hacerlo con un gráfico de correlación, verdad? No tiene sentido entonces el chi cuadrado si scare es el que me permite llegar a esa conclusión, no un gráfico de correlación, entonces chicos, aquí.
1:27:25
1 hora 27 minutos 25 segundos
Vamos a verlo luego esto es un poquito de matemática dura y ya son las 9:30. H, entonces chicos, esto lo vamos a dejar un poquito para después, porque lo que me interesa claramente es que ustedes sepan cuando utilizar un chico cuadrado y luego vamos a ver la matemática, la matemática, para que tal vez les quede claro de dónde viene todo esto, cómo se construye. Pero muchachos, tenemos una clase que se llama el análisis de correspondencia simple y el análisis de correspondencia múltiple.
1:27:54
1 hora 27 minutos 54 segundos
¿En donde utilizamos este chi cuadrado? Entonces vamos a vamos a ponerle un poquito de redundancia a la clase. Entonces para no entrar en eso y tal vez adelantar un poco de clase, no vamos a entrar aquí, vamos a saltarnos de eso y vamos a ver el código chicos, voy a para ver acá.
1:28:28
1 hora 28 minutos 28 segundos
Les presento muchachos, dudas hasta el momento, chicos o k perfecto. ¿Entonces vamos a ver, voy a presentarles a cada chicos igualmente muchachos, esto lo van a tener que hacer en la tarea entonces?
1:28:51
1 hora 28 minutos 51 segundos
Van a ver que por medio de la práctica van a contextualizar todo esto que les estoy diciendo y van a ver que que poco a poco chicos por medio del código van a van a entender todo esto. Bueno, esta era la tarea lleva muchachos y en la semana dos, en la semana dos, yo les dejé el código de clase. Bueno, se van a ir a este código y este código muchacho es un.
1:29:20
1 hora 29 minutos 20 segundos
Es un repositorio en github, en donde chicos lo que necesito es que ustedes abran este que dice edad demo execured. Este es el que tiene los resultados del modelo. Bueno, los resultados de de la de de una de las partes de la tarea. ¿Chicos, no es la no es la resolución de la tarea que ustedes van a hacer, pero utiliza los datos de la tarea, entonces van a utilizar este mismo código?

1:29:52
1 hora 29 minutos 52 segundos
Para poder solucionar lo que les estoy pidiendo. Entonces ya les voy a explicar, voy a agarrar esta parte de las scripts para que ustedes vean chicos. Todos hemos trabajado en en colab, en Notebooks. Hay alguien que no, sí o k genial o k si hay alguien, no, chicos, me avisan, por favor.
1:30:22
1 hora 30 minutos 22 segundos
O K chicos, esta data para ponerles un contexto, lo que vamos a hacer y por ende pues explicarles un poco el código. De hecho les voy a abrir, les voy a les voy a abrir 1 segundo la data para que la vean se llama Personality.
1:30:53
1 hora 30 minutos 53 segundos
¿Sintético, no me la ha estado leyendo para así la coloca así como está, aquí está una preguntilla y mientras busca eso para lo que son las entregas de las tareas, usted prefiere que se la entregue con el notebook o yo le puedo pasar el link del Repo?
1:31:20
1 hora 31 minutos 20 segundos
¿Como quieras, Daniel, como como quieras, digamos si me lo pones en HTML, creo que puede ser mejor, pero igualmente en una, en un ref o no hay ningún problema listo, chicos, ahí están viendo el Excel, verdad? El los datos sí o k muchachos, ésta es una data que que que es una data que ustedes van a ver en la tarea, van a utilizarla para para efectos de la tarea. Es una data que me dice a mí.
1:31:50
1 hora 31 minutos 50 segundos
Por personalidad vean que hay, creo que hay 3 categorías, está el introvertido, el introvertido y el que está como en el medio de esos 2 o K entonces ustedes tienen las diferentes personalidades y tiene acá la energía social tiene aquí cómo se desempeña, hablando las reflexiones de la persona, las habilidades de de escucha, la empatía, la creatividad, la organización, el liderazgo.
1:32:18
1 hora 32 minutos 18 segundos
Si es una persona adversa, perdón, si es una persona conservadora. El riesgo, curiosidad, bueno, o sea, hay un montón de índices chicos que tal vez acá por medio de un experimento. ¿La persona bueno era sintética, verdad? Pero asumamos que es un experimento de Psicología en donde hubieron varias personas que fueron categorizadas como introvertidos, extrovertidos y demás y les fueron poniendo un índice.
1:32:47
1 hora 32 minutos 47 segundos
A cada una de esas variables en relación a las a las habilidades sociales que tiene la persona o k esa, digamos es como la data. Entonces chicos, vamos a trabajar sobre esa data, vamos a hacer un poquito de analítica descriptiva, ya ustedes tienen en en la tarea lo que les estoy pidiendo yo a ustedes. Entonces pues acá no es exactamente lo que vamos a hacer con la tarea, no es lo mismo la tarea.
1:33:16
1 hora 33 minutos 16 segundos
Pero es muy parecido. Entonces muchachos, vean que acá y aquí ven algo muy importante, voy a bajar solamente el código para que ustedes vean algo, vean, vean este apartado de gráficos cuantitativos, vean esto, vean chicos que para poder graficar un hit map solamente estoy, estoy codificando solamente dos líneas de código para graficar un scaner plop.
1:33:46
1 hora 33 minutos 46 segundos
¿Lo ven que está aquí abajo? Solamente estoy graficando también estoy codificando dos líneas de código nada más Button pariplot lo ven también solamente dos líneas de códigos para que me hagan este gráfico o k entonces chicos, este código ustedes lo van a poder utilizar para su tarea también, pero vean que acá viene.
1:34:14
1 hora 34 minutos 14 segundos
La ventaja de la programación orientada a objetos como yo les estaba mencionando, vean que aquí yo utilizo varios import import wine, y eso digamos solamente es para configuración de para que no me salgan alertas, el maplobble panda cyborne para para para verificación de datos. Pero vean chicos que aquí yo utilizo scripts. Utilizo 111 clase, un método que se bueno una sí, digamos una clase que se llama Scripts y ese scripts yo le paso.
1:34:42
1 hora 34 minutos 42 segundos
Mis clases que yo programé gráficos cuantitativos, gráficos cualitativos, test estadísticos, regresión lineal y regresión logística. Eso es parte de la programación orientada a objetos. ¿Dónde están estos scripts o dónde están esos auxiliares que a mí me permiten graficar esas líneas sin tanto código? Están en esta, en esta carpeta, en otro notebook. Lo ven chicos. Entonces por ejemplo, yo tengo aquí gráficos cualitativos.
1:35:11
1 hora 35 minutos 11 segundos
Entonces yo aquí, yo aquí programé los diferentes gráficos cualitativos que yo voy a utilizar. Por ejemplo, la barra de frecuencias, gráficos de pay, barras apiladas, hit, MAP de contingencia, entre otros. Entonces chicos, yo lo que hago es no rellenar el notebook que va directamente para mis stakeholders. Yo a ellos no les voy a pasar tanto código, o sea algo que se vea.

1:35:38
1 hora 35 minutos 38 segundos
¿Humanamente casi que que confuso, pues verlo, sino más bien yo utilizo la programación orientada a objetos que me automatiza toda esta parte, verdad? ¿Por qué les estoy diciendo esto, chicos? Porque en minería de datos y de aquí en adelante, la mayoría de los entregados que ustedes me vayan a dar van a ser modelos que van a tener sus clases completamente por aparte. ¿Qué quiere decir esto? ¿Que si ustedes, por ejemplo, me van a desarrollar un algoritmo en K means?
1:36:07
1 hora 36 minutos 7 segundos
Por otra parte, por otro módulo van a tener la verificación del método de Code jambou. No me lo van a, o sea, pueden meterlo claramente dentro de notebook, pero pues no están aprovechando 100% la ventaja de Python. Por esa razón mejor nos quedamos en Jordan o K muy importante, Jordan.
1:36:32
1 hora 36 minutos 32 segundos
Los notebooks son como cuartos de trabajo, cuartos de trabajo que por ejemplo, si has utilizado cola, bueno me has dicho que no has utilizado notebook, pero a diferencia de scripts. Pay, los notebooks son auto reproducibles. ¿Qué quiere decir esto? Que por ejemplo, si vos ves yo tengo una línea de código acá y si yo le ejecuto esa línea de código, me me me da los resultados inmediatamente de los gráficos que que estoy que estoy programando.
1:37:01
1 hora 37 minutos 1 segundo
Entonces, a diferencia de un de un archivo. PAI esto vos no lo puedes hacer por por por por línea, o sea, ves que aquí yo tengo una celda EI por celda, pues yo no puedo, yo no puedo hacer eso. Teniendo por ejemplo un archivo. PAI, yo tengo que ejecutar todo el archivo. PAI para poder ver los resultados, y eso digamos si vos se lo mandas a un stakeholder o lo que sea a alguien de tu equipo, no es como tan amigable.
1:37:29
1 hora 37 minutos 29 segundos
Usualmente las personas que programamos programamos en ciencia, datos y todo lo demás. Hacemos un usuario notebooks para experimentación rápida, YYY, recolección de resultados, entonces, Jordan, los notebooks nos permiten a nosotros tener resultados inmediatos. Celdas por celdas es un cuaderno del trabajo. Básicamente eso es lo que lo que lo que es un notebook. No sé si has trabajado con R, no con R Studio.
1:37:59
1 hora 37 minutos 59 segundos
Profe, la semana pasada sí trabajé con colap, pero no es como mucho, mucho, mucho, mucho. Ah, OK, pero no es tanto colap, es un lo que yo le estaba preguntando. ¿Era como el otro caso, un ejemplo como le cuando usted mandó el código de clase con cómo se llama esta aplicación con con github? Yo nunca he trabajado porque.
1:38:24
1 hora 38 minutos 24 segundos
Profe Mario, lo que yo hacía era mejor montarlo en un tipo colap chiquitito y se lo enviaba así en vez de hacer todas estas carpetas o K pero vamos a ver, Jordan es es, es lo mismo, entonces no te no te preocupes digamos y trabajos con con colap es lo mismo, por ejemplo es que yo aquí lo tengo en un en un github para que ustedes lo vean, para que ustedes todos ustedes lo vean, solamente es un repositorio, pero Jordan, si vos vamos a ver.
1:38:54
1 hora 38 minutos 54 segundos
Si recuerdas acá voy a volver a a presentar un segundito chicos, porque esto que menciona esto que pregunta Jordan es muy importante. ¿Están viendo la presentación, verdad? Chicos, sí, sí, esto que yo les estaba mencionando esta estructura del proyecto. Si recuerdas Jordan.
1:39:23
1 hora 39 minutos 23 segundos
Yo les estaba mencionando que, por ejemplo, nosotros íbamos a tener los notebooks, o sea lo los colabs. Por ejemplo, acá en notebooks, pero en source vas a tener los diferentes scripts que vas a utilizar o que te van a ayudar dentro de esos colaps. ¿Entonces, qué significa eso? Por ejemplo, yo si yo tengo una celda, claramente yo puedo llegar a este mismo resultado, o sea a este mismo head MAP agarrando todo este código.
1:39:53
1 hora 39 minutos 53 segundos
¿Ahora a ver agarrando todo este código de vamos a ver dónde está esta parte gráfico de correlación por ejemplo y pegárselo acá, yo lo puedo agarrar y pegárselo acá, pero si ves son demasiados gráficos, son demasiados gráficos y si yo sigo bajando hay más y más y más gráficos y más test estadísticos, lo ves?
1:40:23
1 hora 40 minutos 23 segundos
Entonces, si yo le meto todo esto al notebook, va a ser un poquito pesado leerlo. Digamos que no tan posiblemente no tan pesado, pero estaríamos desaprovechando el paradigma de El paradigma de programación que tiene Python, que es la programación orientada a objetos, a diferencia de R.
1:40:47
1 hora 40 minutos 47 segundos
Que si ustedes han programado en R chick, pues toda la codificación se hace de forma lineal o K entonces Jordan ahí la respuesta ante esto es que Python o sea lo que estoy haciendo yo acá estoy grabando un notebook principal, pero por aparte estoy programando. Es un colap que se está auxiliando de otros scrips.
1:41:17
1 hora 41 minutos 17 segundos
Otro código que está en otro archivo y que permite y que me permite a mí. O sea, que nos permite a nosotros automatizar el uso de estas de estos, de esta de estos de estos método sin escribir tanto código en el en el notebook principal o k Daniel, pues pues chequear yo solamente una vez se me hubiera usado cola, porque me se me hace demasiado tieso trabajar en esa cosa.
1:41:46
1 hora 41 minutos 46 segundos
Pero en collabon no puede, digamos yo puedo trabajar, estar trabajando en un notebook y llamar a otro notebook ahí mismo, a otro notebook no, pero a un archivo. Pay, sí, Ah, entonces todos los chiquillos aquí van a tener que trabajar en visual Studio para eso no tarea, no, no, no, no, seriamente, no necesariamente. Sí, digamos, es una de las mejores opciones, pero si ves bueno, aquí están viendo a collab verdad, chicos, sí.
1:42:16
1 hora 42 minutos 16 segundos
Por ejemplo, o sea, si yo no sé voy a dar import de scripts, from scripts, scripts, por ejemplo, yo puedo aquí subir un código, o sea yo puedo subir acá un archivo. PAI que yo programé y ese archivo. PAI lo subo acá, entonces ese archivo. PAI yo ya lo puedo utilizar acá pero tiene que ser punto PAI no puede ser punto IP.
1:42:45
1 hora 42 minutos 45 segundos
¿No tiene, no, no tiene que ser igual que está por acá, lo ves? O sea, todos estos archivos, entonces, exacto, entonces chicos, por ejemplo, acá vamos a ver si profe, entonces todos los scripts los subimos ahí, en en el en el Drive, pero en la carpeta esa no. Santiago, no, no es necesario, o sea con que lo hagas en Colap.
1:43:14
1 hora 43 minutos 14 segundos
Es suficiente. O sea, si ustedes por ejemplo, no quieren mandarlo en visual Studio Code no es no es no, no es digamos obligatorio. Si ustedes quieren hacerlo excelente, no hay ningún problema ahí. Ahí vamos a ver para para para darme a entender aquí chicos todo esto vamos a ver si visual Studio es que yo aquí tengo para darme a entender.
1:43:49
1 hora 43 minutos 49 segundos
Y ya vemos que yo acá esto que pasó acá chicos estamos segundo que aquí esto se me trabó y no sé por qué voy a volver a compartir de esto número dos, si yo acá utilizo visual Studio Code, por ejemplo, acá me está cargando.
1:44:14
1 hora 44 minutos 14 segundos
¿Eso ya ya depende, chicos de cómo ustedes quieran trabajar la tarea, verdad? Pero lo normal sería, por ejemplo, que ustedes agreguen una nueva carpeta que sea, no sé, tarea número dos, la abran chicos y acá creen la misma estructura de la presentación en donde ustedes estén aquí datos, ustedes suman sus datos los datos de.
1:44:40
1 hora 44 minutos 40 segundos
De personalidad sintética, YYY la de la, la, la de la nutrición, luego ven acá a scripts, luego ven acá notebooks, ven entonces aquí en scripts chicos, ustedes, por ejemplo, pueden agregar 1 nuevo que sea métodos cuantitativos, por ejemplo. Pay.
1:45:06
1 hora 45 minutos 6 segundos
Entonces lo único que hacen chicos es solamente copiar esto o k o sea llegan acá métodos, bueno aquí me confundí es métodos por te voy a agarrar metos cuantitativos, copio todo y lo pego acá o KY luego chicos, aquí agrego main. IPNIBY aquí pues ya empiezo a programar, entonces yo hago acá import for.
1:45:34
1 hora 45 minutos 34 segundos
Métodos cuantitativos ahí la única gran diferencia para los que van a trabajar, como los notebooks, es que ya traen muchas librerías instaladas y cosillas. Sé aquí y aquí le toca a 1 venir YY provisionar el ambiente, instalar las librerías que vaya 1 a utilizar y todo.

1:45:55
1 hora 45 minutos 55 segundos
Sí, Daniel, porque lo que pasa es que, por ejemplo, o sea, ves que aquí esto es un método tuyo, o sea, esto es un método que nosotros estamos creando desde cero. O sea, aquí esto no lo vas a encontrar por ningún lado. Aquí lo que estás haciendo es que estás estás como creando una librería desde cero, en el sentido de que aquí métodos cuantitativos. Si vos importás esta librería, ya tenés todos los métodos cuantitativos, o sea, todos los gráficos que vos quieras utilizar para.
1:46:25
1 hora 46 minutos 25 segundos
Para graficar tus datos, entonces es como utilizar un cycle learning, es como utilizar lo que sea, o sea es básicamente eso, es lo que puedes hacer. Entonces eso chicos, ustedes así lo pueden trabajar, como les digo, pueden trabajarlo también en un colap en donde este va a ser claramente el archivo principal, pero aquí chicos ustedes van a tener que subir, no sé, vamos a poner un ejemplo, vean que acá.
1:46:54
1 hora 46 minutos 54 segundos
Y aquí ya se nos está yendo el tiempo, chicos, pero no hay problema. Si yo abro un archivo de texto y copio los valores, o sea, copio o copio el Código, eso también yo lo puedo Guardar como métodos, punto, perdón, métodos cuantitativos, punto pie. Acá vamos a ver si no me lo está leyendo. Bueno, esto no me lo está leyendo.
1:47:24
1 hora 47 minutos 24 segundos
¿Y dónde me los estoy guardando para ver ubicación? Lo voy a guardar en descargas, por ejemplo, ya lo guardé en descargas, voy a irme a descargas un segundito porque parece que desde ahí no me está, no me lo está transformando, pero acá yo lo renombro, le Quito eso que no me estaba, no me está dejando y ahora sí, ahora sí es un archivo punto pie.
1:47:52
1 hora 47 minutos 52 segundos
Entonces era ese archivo. Pi, ahora sí lo subo acá, entonces yo métodos cuantitativos ya ya los tengo acá, entonces Imports scripts. Bueno, aquí métodos cuantitativos se llamaba entonces from métodos cuantitativos import y aquí pues importo aquí importo la clase que se llama.
1:48:19
1 hora 48 minutos 19 segundos
Gráficos cuantitativos y luego ahí yo ya puedo empezar a utilizar todo esto que tenemos acá todos los diferentes métodos, scarplot, line, plot, histogramas, entre otros, igual chicos para la tarea, muchachos y todavía no se acostumbran a la programación orientada a objetos.
1:48:46
1 hora 48 minutos 46 segundos
Pueden entregármelo sin esto, o sea, en el sentido de que pueden programarme todo mientras nos acostumbramos de forma tradicional. Entonces, si no se me, si no se sienten cómodos con la primera tarea, pueden hacerlo así. Pero chicos, como les digo, o sea, minería de datos, ingeniería software. Ahí Braulio y Daniel me corrigen, pero básicamente se programan muchos en programación orientado a objetos.
1:49:13
1 hora 49 minutos 13 segundos
Yo tengo por ejemplo un o sea si estamos trabajando 1 e t L por ejemplo, un digamos, no sé un painant. Por ejemplo, yo tengo un archivo que me extrae, tengo otro archivo que me procesa y tengo otro archivo que me carga. Eso se llama E t L o e L T si estamos trabajando por ejemplo en data en data lakes, con la la metrología de Medalum, entonces digo, si estamos trabajando en nube, si estamos trabajando en lo que sea. Normalmente la estructura de los proyectos es así. Yo tengo un archivo.
1:49:41
1 hora 49 minutos 41 segundos
¿Que me cumple una misión específica? No tengo, por ejemplo, ahí me confirmará Braulio y Daniel, no tengo, por ejemplo, un notebook o un archivo. Pay que me hace todo el código solamente en un archivo yo libre. ¿Yo le llego esto a mi, a mi manager o a mi p o y me despide, verdad, entonces muchachos, básicamente así es como como se programa en en, en el software real o en El Mundo real, verdad? ¿Ahí me conformarán Braulio y Daniel, chicos, y si estoy diciendo mentiras, verdad?
1:50:13
1 hora 50 minutos 13 segundos
Muchachos, nos hacen falta 10 minutos, voy a voy a explicar así rápido por acá tengo un mensaje, voy a voy a voy a explicar esto un poquito rápido chicos, pero igualmente aquí les queda el código no para eso sí podemos utilizar ch G, P t o gemina y lo que sea para que les explique.
1:50:36
1 hora 50 minutos 36 segundos
Detalladamente, si no los estoy, no me da chance a explicar detalladamente lo que hace cada código, chicos. Pero lo más importante que quería mencionarles acá era precisamente esto, esta discusión que ya teníamos. O sea, teníamos los diferentes métodos gráficos, cuantitativos, cualitativos, test estadísticos, loeim, regresión lineal y regresión logística para poder implementarlos en esta en este data SET.
1:51:01
1 hora 51 minutos 1 segundo
Entonces chicos, vean que acá yo bueno cargo los datos, vean DF. Reads, CLB acá, personalidad, personalidad, lo lo convierte en categoría y algunas otras cosas, como por ejemplo la alta energía, si esa alta energía es mayor a la mediana, lo califico que si tiene una alta energía con respecto a la mediana o no tiene una alta energía.
1:51:27
1 hora 51 minutos 27 segundos
Luego chicos, este este, bueno este ahorita lo vemos, esto no, no lo utilizo tan no es tan crítico, pero luego chicos, utilizo el print. SHAPE para saber cuáles son las dimensiones de mis datos. Aquí ven que yo tengo 20000 líneas con 31 columnas, tengo aquí la la cuantificación de cada del conteo de cada una de esas personalidades extrovertidos. Tengo 6857.
1:51:56
1 hora 51 minutos 56 segundos
Introvertidos tengo 6570 y este que es entre medio de los dos tengo 6573. Entonces aquí yo hice un poquito de de conteos para saber con qué me estoy enfrentando. Luego hago un hit Maps, hago un scare plot, un box, plot gráfico, violín, un Nine plot y un par plot para cualquier, para para, para obtener algunos resultados. Ya lo vamos a interpretar.
1:52:21
1 hora 52 minutos 21 segundos
Pero chicos, vean que acá yo tengo un gráfico de correlación. Yo veo digamos hay una correlación alta. Por ejemplo, vea que aquí esta variable que está por acá está muy correlacionada con social Energy, tiene una correlación del 71 del 71%, digamos otra que puede tener una correlación alta, por ejemplo esta. Ah, bueno, también es la misma Party LinkedIn, bueno, LinkedIn, perdón con la habilidad para para hablar.
1:52:51
1 hora 52 minutos 51 segundos
Vean que la correlación del 70% también learship. Vean, hay mucha correlación entre varias variables y tal vez alguna correlación significativa negativa puede ser, no sé, empatía con respecto al learship. Bueno, esto es muy interesante, la empatía con respecto al liderazgo, que sea una bueno, no es tan fuerte la correlación negativa, pero hay una correlación ahí. Si lo ven chicos, aquí tenemos.
1:53:19
1 hora 53 minutos 19 segundos
¿Aquí tenemos un scare plot, ya nos faltan 8 minutos chicos, vamos a ver si nos da chance, aquí tengo un scare plot, muchachos en donde vean que acá las personas que son extrovertidas lo ven acá en este cuadrante son las personas que tienen mayor cantidad de de de energía social y también son las personas que se desempeñan bastante bien hablando, lo ven? Por otro lado, el introvertido es completamente al revés.

1:53:47
1 hora 53 minutos 47 segundos
Son las personas que no tienen tanta energía social y también las personas que no son muy buenas. Pues hablando en público, así es como lo estoy leyendo. Sí, sí. ¿Y el que está en el medio? ¿Pues claramente está en el medio, verdad? Aquí los extrovertidos chicos vamos a ver extrovertidos. Vean que aquí tengo tengo la personalidad extrovertidos, introvertidos.
1:54:13
1 hora 54 minutos 13 segundos
Tengo esta variable que estoy analizando chicos. ¿Y qué son estas bolitas que yo tengo aquí fuera de los boxflot? ¿Alguien sabe? ¿Son los outliers, son los outliers correcto, son las personas que se me están saliendo de de Del comportamiento atípico y esto no es normal, verdad que esto no es normal? Entonces muchachos vean que por ejemplo acá hay algunos extravertivos que inclusive tienen una nota bastante baja en esta variable, entonces por eso es que son bastante raros.
1:54:40
1 hora 54 minutos 40 segundos
¿A pesar de que son extrovertidos, esta variable no es tan, tan, no es tan grande para ellos, lo ven? Gráfico violín es muy similar al al gráfico de de Boxpro, Bueno, pero yo prefiero el boxprot, ese nunca lo había visto. ¿Yo profe, este violín sí es la primera vez que veo ese gráfico, este este gráfico, digamos que que es bastante bueno en el sentido de que además de de dar una dispersión, también nos da como una distribución, ves?
1:55:10
1 hora 55 minutos 10 segundos
Por ejemplo, aquí me está diciendo, bueno, es que aquí ya nos ya pasamos de de esta variable a empatía, pero la parte del EXTROVERTIDO, además de darnos la distribución, también nos está diciendo si es una distribución normal o no. Bueno, las 3 distribuciones normales, pero por ejemplo, esta vez que es o sea todas son distribuciones normales claramente, pero posiblemente.
1:55:36
1 hora 55 minutos 36 segundos
¿En algún momento este gráfico te voy a decir si una si una una distribución tiene una una distribución sesgada a la izquierda o a la derecha entre otros, entonces además de darte la dispersión de los datos, también te da una distribución y los otros lados son perdón, los otros lados son iguales, es lo mismo? Exacto, son iguales, o sea si los ves es igual, es un espego, es un espego o k esto no chicos, eso esto no, no, no digamos para efectos de esta set esta esto no tiene sentido.
1:56:06
1 hora 56 minutos 6 segundos
Y aquí chicos, esto es como una especie de una revelación lineal. Esto De hecho se los piden a la tarea. Vean que aquí yo yo coloco un scareplot del social Energy con respecto al a esta variable de de si es bueno hablando en público y vean chicos que, por ejemplo, si yo estoy tratando de pronosticar si el social Energy es una variable fuerte para poder pronosticar si una persona es buena hablando, vean que sí.
1:56:33
1 hora 56 minutos 33 segundos
Es una relación positiva, es una correlación positiva. Si yo trazo una línea acá como una revelación lineal, yo veo que entre más más grande sea el índice del social Energy. También va a ser más grande este índice que yo estoy tratando de de de evaluar. Luego tienen una relación positiva, es parte de una revelación o k luego chicos tiene el purplot que este purplot parece que como que vas a aparecer o algo haya visto de decirver.
1:57:03
1 hora 57 minutos 3 segundos
Pero personalmente eso es 1 de los mejores gráficos que yo he visto, que además de de hacer un scareplot por cada una de las variables, lo ven. O sea, yo hago un scareplot de de de empatía con respecto al learship de Learship con respecto a creatividad de social Energy con respecto al learship lo bueno, o sea yo tengo acá si fue como si fuese una matriz todas las relaciones en, en, en, en un en un gráfico de scareplot.
1:57:32
1 hora 57 minutos 32 segundos
Y adicional a esto, vean que yo tengo las famosas distribuciones que vimos en los 3 estadísticos. Lo ven entonces chicos, vean bastante, bastante fácil. Vean el earship, el earship. Si nosotros lo tratamos de graficar, vean que se ve claramente que los extrovertidos tienen una una distribución más si hacia la derecha con respecto a los introvertidos y estamos hablando del earship.
1:57:57
1 hora 57 minutos 57 segundos
Luego, o sea, son claramente distintos y nosotros sacamos un test estadístico, son distintos. Gráficamente eso lo que estamos viendo aquí, tal vez algunos gráficos cualitativos, un bueno para poder cuantificar cuántas cuántos valores tenemos de cada 1. Esto ya lo habíamos hecho, esto es exactamente lo mismo, el porcentaje, si estamos hablando de un PAI, un gráfico de de PAI o un pay share, un gráfico pastel, vean que aquí me dan los porcentajes de cuántas variables tenemos en cada una.
1:58:28
1 hora 58 minutos 28 segundos
Esta variable contingencia ahorita no la vamos a ver porque no tiene tanto sentido ahorita analizarlo, pero muchachos vean que aquí es donde ya empiezo a utilizar las pruebas estadísticas, utilizo Nova, utilizo manova y utilizo chi cuadrado, entonces muchachos, aquí utilizo por ejemplo la habilidad de hablar en público con respecto a la personalidad, entonces aquí si nosotros nos vamos a los gráficos vamos a ver acá.
1:58:57
1 hora 58 minutos 57 segundos
Vean que aquí chicos, la distancia entre esas 3 distribuciones es notablemente separadas. Son muy diferentes, notablemente lo ven. ¿Entonces tienen sentido que el Pi vale para la Nova? Lo en el Pi vale, me dé cero 0.0 es menor a 0.05. Eso me quiere decir que definitivamente hay una diferencia significativa.
1:59:23
1 hora 59 minutos 23 segundos
Entre esas variables con respecto a si una persona es introvertido o no, es o no es o es extrovertido, entonces, chicos, si estamos tratando de pronosticar si una persona es introvertido o extrovertido, esta variable tiene peso o no tiene peso. ¿Sí, sí, tiene peso, claro, tiene un peso bastante grande, verdad? Entonces chicos, si ustedes el día de mañana están tratando de entrenar un modelo predictivo para para pronosticar eso.
1:59:50
1 hora 59 minutos 50 segundos
Es una de las variables de peso que que que fijo el modelo le va a servir para pronosticar y ya no va a ser como todo El Mundo en el mercado que solamente le dan al modelo o o los stakeholders que le dicen, mira, es que quiero hacer esto, pero quiero pronosticar eso. ¿Pero cuando ustedes van a ver los datos, nada de esos datos es significativamente diferente, verdad? Entonces a veces la gente quiere que 1 haga un modelo predictivo y no tiene los insumos necesarios.

2:00:16
2 horas 16 segundos
¿Entonces, como les estaba mencionando antes, había gente que me decía a mí que querían hacer un gráfico de series de tiempo cuando no tenían ni siquiera datos, entonces me dicen, Mira, es que cuánto vamos a vender el el día de mañana de un nuevo producto que ni siquiera ha salido del mercado? ¿Entonces, cómo utilizar inteligencia artificial ahí, verdad? ¿No se puede okay, chicos manoa también ven el P vale o qué es lo que me interesa, muchachos, es que vayan al p, vale?
2:00:41
2 horas 41 segundos
Claramente todo el tema de la variación SQ la variación es es importante, pero me interesa más el pival que es lo que acabamos de ver. Si todos los pivalus son por menos del 0.05, vean que es significativamente estadístico. Vean que aquí chicos, para el manoa, para el manoa utilicé social Energy. Utilicé solamente dos variables vean con respecto a la al personal type entonces.
2:01:09
2 horas 1 minuto 9 segundos
¿Esos dos en esos dos hay una diferencia estadística entre esos dos variables, entonces esas dos variables quiere decir que hay ahí algún criterio de separación que me separa perfectamente? ¿Quién es introvertido y quién es extrovertido o k el chico? Ahora ahorita no, no lo vamos a ver muchachos, espero la relación lineal múltiple, sí, ya ya vamos a terminar muchachos solamente 5 minutos aquí sí, es bastante importante chicos.
2:01:38
2 horas 1 minuto 38 segundos
¿En el video recuerden que es el r cuadrado qué quiere decir el R cuadrado? La dimensionalidad es la dimensionalidad. Daniel, no, no tanto, no, no tanto, es la recta de mejor ajuste. ¿Cuánto los los puntos? No, sí, va por ahí. Roberto, exacto, sí, sí, sí, sí.
2:02:04
2 horas 2 minutos 4 segundos
Ahí Daniel, cuando decías dimensionaría hasta qué qué querías referirte para para ver si si llegaba por ahí, tal vez tal vez lo lo confundí temprano en la en la lección me habías mencionado de que un gráfico en R dos, si no me equivoco, era pues un gráfico de 2 dimensiones. Ah, no, no, Ah o k no, no, no, no, no, sí, sí, ya te entendí por dónde va el asunto, pero este es un contexto diferente, tal vez chicos y ya por donde decía Roberto el R cuadrado.
2:02:32
2 horas 2 minutos 32 segundos
Entre más se acerque a 1 es mejor. Ajá. ¿Cuesta mucho que se acerque a 1, digamos en realidad que de 1, verdad? Pero si da 0.99 en un modelo es muy, muy bueno. Ahora bien, que esos, o sea que en un modelo que fuera fuera del laboratorio de eso cuesta muchísimo también.
2:02:59
2 horas 2 minutos 59 segundos
Claro, totalmente totalmente correcto. Chicos, como dice Roberto, el R cuadrado entre más cercano, 1 es mejor. ¿Qué significa el R cuadrado? Chicos, el r cuadrado es la inercia explicada al modelo. En otras palabras, es cuánto poder predictivo o cuánta explicatividad tiene el modelo con respecto a la variable a la variable predecir que nosotros queremos pronosticar, por ejemplo.
2:03:25
2 horas 3 minutos 25 segundos
Vamos a vamos a pronosticar si un plato de comida en un hotel es aprobado por un cliente o no por 1, por 1, sí por un cliente. Entonces tenemos diferentes variables, tenemos la variable de servicios al cliente, tenemos la variabilidad de sabor de la comida, tenemos la variabilidad del precio del plato, por ejemplo, si todo eso es bueno, cierto, si todo eso es bueno.
2:03:51
2 horas 3 minutos 51 segundos
Posiblemente, digamos, el cliente, vaya, le vaya a dar, le vaya a dar. ¿Me gusta a la comida, verdad? Le vaya a gustar el plato de comida. Entonces esas 3 variables tienen una explicatividad bastante buena. Si una persona le vaya a gustar el plato de comida o no, entonces va a tener un r cuadrado bastante bien, pero chicos, por ejemplo, si yo estoy tratando de pronosticar eso, si si quedemos con el mismo ejemplo, si una persona le gusta un plato de comida o no le gusta un plato de comida.
2:04:17
2 horas 4 minutos 17 segundos
¿Pero por ejemplo, voy a salir con una tontera y estoy recolectando una variable para saber si ese día llovió o no llovió, verdad? Entonces tengo 3 variables, pero adicionalmente le agregué si llovió o no llovió. ¿Ustedes creen chicos, si esa variable llovió o no llovió? ¿Le le agrega a explicatividad al modelo para nada, verdad? O sea, no tiene como tanta relación a menos que sea algo ahí como muy muy raro entonces posiblemente chicos, si yo.
2:04:45
2 horas 4 minutos 45 segundos
No sé, tengo esas 3 variables que sí eran importantes, si el R cuadrado era 70, si le agrego esa que si llovió o no llovió, posiblemente solamente se mueva 71. ¿No le agrega absolutamente nada explicatividad al modelo, entonces el r cuadrado muchachos es 1 de los coeficientes más importantes en una revisión lineal, verdad? Si ustedes están tratando, por ejemplo, de emitir alguna conclusión y ese RR cuadrado es por ejemplo, 0,00001, literalmente todo lo que ustedes están diciendo sea mentira.
2:05:15
2 horas 5 minutos 15 segundos
O K profe, al final ese r al cuadrado es como el porcentaje de probabilidad. Por ejemplo, ese 0.66 sería como un 66% y el 1 sería el 100% de probabilidad. Sí es un es un porcentaje, es un porcentaje correcto. Chicos. Acá me está diciendo, me está diciendo buena pregunta Sharon sobre la asistencia, no he habilitado la asistencia, chicos, ahí me disculparán, pero le tomé aquí un pantallazo a las personas que asistieron.
2:05:45
2 horas 5 minutos 45 segundos
Igualmente tengo la grabación y entonces la grabación me da también las personas que asistieron. Entonces ahí yo les pongo la asistencia, chicos, para que sepan muchachos. Entonces bueno, el R cuadrado también teníamos el F Estadístico. El F Estadístico también nos nos dice algo muy similar al al al F al r cuadrado también nos dice qué tan, qué tan.
2:06:08
2 horas 6 minutos 8 segundos
¿Tan correcto es el modelo que tan usable es el modelo? O sea, si el modelo es concluyente, no es concluyente, también nos está diciendo acá y vean que eso también es como parecido a un criterio. el P value aquí es 0.000 y es menor a 0.05, entonces acá nos quiere decir que este modelo también es bastante bueno para para para poder emitir un juicio. ¿Y aquí luego chicos, viene la parte de los coeficientes, que esta parte de los coeficientes es, es, es, es, es oro, verdad? Eso es oro.
2:06:36
2 horas 6 minutos 36 segundos
Vean que aquí chicos, si yo estoy tratando de pronosticar si una persona es buena hablando o no, es bueno hablando con respecto al social Energy y si la persona es interactive o no es extrorectivo, vean qué es lo que está pasando. O sea, yo estoy tratando de pronosticar si una persona es buena hablando o nos van hablando. Vean qué lo que pasa, si la persona tiene 111 energía social buena, vean que ese coeficiente es positivo. Ese coeficiente es positivo.
2:07:05
2 horas 7 minutos 5 segundos
Entonces, al ser positivo, quiere decir que entre más grande sea el social Energy, la persona es mejor hablando, pero vean que lo que pasa aquí muchachos, aquí ven algo muy importante, vean el coeficiente negativo. El coeficiente negativo nos quiere decir claramente alguna relación opuesta. ¿Verdad que nos quiere decir esto? Chicos, que por ejemplo, si una persona es introvertida va a tener 1 − 2.48.
2:07:33
2 horas 7 minutos 33 segundos
Puntos en esa nota final de si una persona es buena hablando, si la persona es introvertida y al contrario sí es extrovertida. Loen, aquí chicos hay que fijarnos nosotros siempre. Los Pi valio vean que cada 1 de los coeficientes también tiene un Pi valio lo ven tienen un Pi valio. Si esos Pi valio son 0.05 menor a 0.05 esa conclusión es esa conclusión es correcta.

2:08:00
2 horas 8 minutos
Pero vean que el social Energy y la empatía parece ser que no no recurren a eso es mayor a 0.05. Entonces esto no es concluyente. No podemos emitir un juicio, a diferencia de estos dos, que es introvertido y extrovertido porque el P o Ares está por debajo del 0.05 o k chicos, vamos a dejarlo hasta aquí, muchachos, vamos a dejarlo hasta aquí chicos.
2:08:28
2 horas 8 minutos 28 segundos
¿Vamos a ver cómo nos va la tarea muchachos, cualquier cosa, sé chicos que que que le corté 1 hora la clase, pero era era necesario, muchachos tenía una reunión con una gente Argentina que les estaba mencionando, pero las próximas clases ya nos vamos a conectar a las 7 chicos, entonces la clase va a ser completamente las 3:00? H, Bueno, si si nos da nos da, nos da los temas claramente.
2:08:54
2 horas 8 minutos 54 segundos
Como le decía la vez pasada Sofía, van a haber momentos en donde vayamos a terminar 1 hora antes o media hora antes, entonces eso va, va va a depender de acuerdo a la A la materia que estamos viendo chicos. Entonces si tienen dudas muchachos, preguntas con algo, por favor escríame yo saco el tiempo 1 hora o 2 horas para que veamos algo si necesitan 111 sesión extra para poder profundizar esos temas.
2:09:24
2 horas 9 minutos 24 segundos
Díganme muchachos, para ver si coordinamos con el grupo y vemos si hacemos una clase, un sábado o un domingo, o lo que sea, vemos las agendas de de de todos nosotros y ver si nos podemos acomodar. Pero la idea chicos es que bueno, si lo ven muchachos, posiblemente el curso se vaya haciendo cada vez más difícil. Entonces porque pues bueno, vamos a ver mucha álgebra, vamos a ver mucha.
2:09:47
2 horas 9 minutos 47 segundos
Muchas, muchas ecuaciones de distancia, de oración, programación sobre todo. Entonces chicos, la idea es que nos vayamos cada vez más poniendo de acuerdo con todo esto, muchachos, dudas, preguntas, nada más ni repasar lo que no se explico porque sí es bastante. Sí, Braulio. Lo de lo demás es puro código. Entonces si hay alguna duda, avísenme chicos, lo vemos.
2:10:16
2 horas 10 minutos 16 segundos
Si no, no ahí, si no me molesta que utilicen HGPT para nada, o sea que les explique un poco sobre aquí está haciendo el Código, no hay ningún problema, la idea es que ustedes repliquen ese código, o sea utilicen ese mismo código para para la tarea. Pero chicos, créanme que que que cuando ya vayan a programar y vayan solución a la tarea, vayan van a ver que que los sistemas se les va a hacer más sencillo, cuando lo hagan en práctica van a ver que que se les va a facilitar cada vez más todo esto que estamos viendo o k.
2:10:47
2 horas 10 minutos 47 segundos
Listo profe, hasta luego listo chicos, hasta luego. Luego buenas noches, gracias. Es que me acabo de acordar que Sofía me dijo que le preguntara algo, sofi no pudo asistir porque está en la incapacitar en cualquier cosa. Verdad que las tareas usted ha dicho que no había problema, si la entregamos después es que ella no la pudo entregar.
2:11:16
2 horas 11 minutos 16 segundos
No, no hay ningún problema. Kaila, ustedes tienen todo el chance, o sea, tienen todo el chance del cuatrimestre hasta las 15 semanas y en el momento que ustedes pueden entregar o ustedes la entregan, no hay ningún problema. Listo, muchísimas gracias. Hasta luego, listo hasta luego.