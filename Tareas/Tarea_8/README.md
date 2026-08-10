# Tarea 8 — Aprendizaje No Supervisado: K-Means y Clustering Jerárquico

**Lead University · Minería de Datos · Métodos NO Supervisados**

## Objetivo

Aplicar K-Means y clustering jerárquico aglomerativo sobre dos datasets de dominios distintos,
determinando el número óptimo de clusters con el Método del Codo de Jambú, validando el corte con el
dendrograma y comparando ambas particiones sobre el plano del PCA.

---

## Estructura de archivos

```
Tarea_8/
├── datos/
│   ├── Customers.csv             # Customer Segmentation (2 000 × 8): clientes de centro comercial
│   └── wine-clustering.csv       # Wine Clustering (178 × 13): propiedades fisicoquímicas
├── scripts/
│   ├── __init__.py
│   ├── kmeans_analysis.py        # Clase KMeansAnalysis (codo, silueta, centroides, biplot PCA)
│   └── clustering_jerarquico.py  # Clase ClusteringJerarquico (enlaces, dendrogramas, cofenética)
├── 1_Customers.ipynb             # Dataset 1 (negocio)
├── 2_Wine.ipynb                  # Dataset 2 (científico)
├── tarea_clustering.pdf          # Enunciado
└── README.md
```

---

## Descripción de notebooks

| Notebook | Dataset | Ejercicios |
|---|---|---|
| `1_Customers.ipynb` | `Customers.csv` | a–e: K = 4, segmentación de clientes por perfil de gasto |
| `2_Wine.ipynb` | `wine-clustering.csv` | a–e: K = 3, agrupamiento de vinos por composición química |

Ambos notebooks siguen los cinco puntos del enunciado: a) selección y exploración de datos,
b) preprocesamiento con estandarización obligatoria, c) determinación de K con el Codo de Jambú y la
silueta, d) aplicación de K-Means y del clustering jerárquico con dendrograma, y e) análisis y
conclusiones visuales sobre el plano del PCA.

**Resultado de `1_Customers.ipynb`:** el codo de Jambú fija K = 4 (la reducción marginal de inercia cae
de 11,8 % a 7,0 %) y el dendrograma de Ward lo confirma, pero la silueta nunca supera 0,167 y el ARI
entre los dos algoritmos es de solo 0,243. El dataset no tiene estructura de clusters real: los cuatro
segmentos son una partición útil impuesta sobre un continuo, y así se reporta.

**Resultado de `2_Wine.ipynb`:** K = 3 queda confirmado por triple validación convergente — codo de
Jambú (28,2 % y 23,1 % de reducción en los dos primeros pasos, 5,2 % en el tercero), silueta (máximo
global en k=3, 0,2849) y dendrograma de Ward (el mayor salto de altura del árbol, de 12,57 a 27,65). El
ARI entre K-Means y Ward es de 0,853, con 169 de 178 muestras (94,9 %) en la misma asignación, y los
tres grupos tienen perfil enológico coherente: vinos ligeros, tánicos/ácidos y estructurados.

---

## Uso de las clases

```python
from scripts import ClusteringJerarquico, KMeansAnalysis

# K-Means
km = KMeansAnalysis(X_scaled)
km.codo_jambu(max_k=10)                     # inercia (WCSS) de k=1 a k=10
km.silhouette_kmeans(max_k=10)              # silueta media por k
etiquetas, centros = km.ajustar(n_clusters=4, random_state=42)
km.plot_barras_centros()
km.plot_radar_centros()
km.aplicar_pca(n_components=2)
km.plot_biplot(anotar=False)

# Clustering jerárquico
cj = ClusteringJerarquico(X_scaled)
cj.calcular_agregaciones()                  # ward, average, single, complete
cj.correlacion_cophenetica()
cj.plot_dendrogramas_comparacion(truncate=30)
cj.plot_dendrograma_corte(alturas=[35.1], etiquetas=['4 clústeres'])
grupos = cj.obtener_grupos(k=4, method='ward')
```

Las clases trasladan el código de los notebooks de clase conservando los nombres de las funciones, las
variables y los parámetros originales; cada método lleva un comentario indicando de qué celda proviene.

| Script | Clase | Notebook base |
|---|---|---|
| `kmeans_analysis.py` | `KMeansAnalysis` | `Clases/Notebooks/Clase 8 - Kmeans.ipynb` |
| `clustering_jerarquico.py` | `ClusteringJerarquico` | `Clases/Notebooks/Clase 7 - Clustering jerarquico.ipynb` |

Se marcan con `# Extensión propia — no existe en el notebook de clase` los añadidos que el enunciado
requiere y los notebooks de clase no cubren: el truncado de dendrogramas (los notebooks de clase trabajan
con ~20 observaciones, aquí hay 178 y 2 000), `correlacion_cophenetica()`, `tabla_perfiles()`,
`plot_dendrogramas_comparacion()` y las anotaciones opcionales del biplot.

---

## Ejecución

### Dependencias

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
```

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

Desde la carpeta `Tarea_8/`, para que las rutas relativas a `datos/` y `scripts/` funcionen:

```bash
jupyter notebook 1_Customers.ipynb
jupyter notebook 2_Wine.ipynb
```
