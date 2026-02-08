📌 **Meta-features por grupo (nombre y significado)**

---

## 🧠 General

Caracterizan propiedades básicas del dataset: tamaño, proporciones, etc.

| Meta-feature | Qué mide |
|-------------|----------|
| attr_to_inst | Razón entre número de atributos y número de instancias. |
| cat_to_num | Razón entre atributos categóricos y numéricos. |
| freq_class | Frecuencia relativa de cada clase. |
| inst_to_attr | Razón entre instancias y atributos. |
| nr_attr | Número total de atributos. |
| nr_bin | Número de atributos binarios. |
| nr_cat | Número de atributos categóricos. |
| nr_class | Número de clases distintas. |
| nr_inst | Número de instancias (filas). |
| nr_num | Número de atributos numéricos. |
| num_to_cat | Razón entre número de numéricos y categóricos. |

*(Estos metafeatures son escalares o proporciones según conteos de filas/características.)*

---

## 📊 Statistical

Proporcionan estadísticas descriptivas sobre atributos numéricos del dataset.

| Meta-feature | Qué mide |
|-------------|----------|
| can_cor | Correlaciones canónicas entre columnas. |
| cor | Correlación absoluta entre pares de atributos. |
| cov | Covarianza entre pares de atributos. |
| eigenvalues | Valores propios de la matriz de covarianza. |
| g_mean | Media geométrica de atributos. |
| gravity | Distancia entre centro de masa de minorías y mayorías. |
| h_mean | Media armónica de atributos. |
| iq_range | Rango intercuartílico. |
| kurtosis | Curtosis de los atributos. |
| lh_trace | Traza de Lawley-Hotelling (dispersión multivariante). |
| mad | Desviación absoluta mediana. |
| max, min | Máximo y mínimo por atributo. |
| mean | Media por atributo. |
| median | Mediana. |
| nr_cor_attr | Número de pares de atributos altamente correlacionados. |
| nr_disc | Número de relaciones correlacionales canónicas con clase. |
| nr_norm | Número de atributos normalmente distribuidos. |
| nr_outliers | Cantidad de atributos con al menos un outlier. |
| p_trace | Rastreo de Pillai (otra medida de dispersión multivariante). |
| range | Rango (max-min). |
| roy_root | Raíz de Roy (estadística multivariante). |
| sd | Desviación estándar. |
| sd_ratio | Test de homogeneidad de covarianzas. |
| skewness | Sesgo de distribución. |
| sparsity | Medida de sparse/escasez de valores. |
| t_mean | Media recortada (trimmed mean). |
| var | Varianza. |
| w_lambda | Wilks’ Lambda (otra medida multivariante). |

*(Los nombres con `.mean` y `.sd` se derivan de aplicar funciones de resumen a estas medidas.)*

---

## 🧮 Information Theory (Teoría de la Información)

Miden cantidad de información y relaciones entre atributos y clase.

| Meta-feature | Qué mide |
|-------------|----------|
| attr_conc | Coeficiente de concentración entre atributos. |
| attr_ent | Entropía de cada atributo predictivo (Shannon). |
| class_conc | Coeficiente de concentración entre atributo y clase. |
| class_ent | Entropía de la clase objetivo (Shannon). |
| eq_num_attr | Número de atributos equivalentes para una tarea. |
| joint_ent | Entropía conjunta entre atributos y clase. |
| mut_inf | Información mutua entre atributo y clase. |
| ns_ratio | Proporción de ruido entre atributos. |

*(Estas medidas son relevantes para datasets con distintos tipos de atributos y clases.)*

---

## 🌲 Model-based (Basado en modelos)

Extraen características del modelo entrenado (usualmente un árbol de decisión).

| Meta-feature | Qué mide |
|-------------|----------|
| leaves | Número de hojas en árbol de decisión. |
| leaves_branch | Tamaño de ramas del árbol. |
| leaves_corrob | Corroboración entre hojas. |
| leaves_homo | Homogeneidad de hojas. |
| leaves_per_class | Proporción de hojas por clase. |
| nodes | Número de nodos no hoja. |
| nodes_per_attr | Nodos por atributo. |
| nodes_per_inst | Nodos por instancia. |
| nodes_per_level | Nodos por nivel. |
| nodes_repeated | Número de nodos repetidos. |
| tree_depth | Profundidad del árbol. |
| tree_imbalance | Desequilibrio del árbol. |
| tree_shape | Forma del árbol. |
| var_importance | Importancia de variables según el árbol. |

*(Estos se obtienen tras entrenar internamente un modelo simple, típicamente un árbol.)*

---

## 🧪 Landmarking

Basados en desempeño de métodos “rápidos” (clasificadores simples).

| Meta-feature | Qué mide |
|-------------|----------|
| best_node | Rendimiento del mejor nodo de árbol simple. |
| elite_nn | Rendimiento de 1-NN de élite. |
| linear_discr | Rendimiento del clasificador discriminante lineal. |
| naive_bayes | Rendimiento de Naive Bayes. |
| one_nn | Rendimiento de k-NN (k=1). |
| random_node | Rendimiento de nodo aleatorio. |
| worst_node | Rendimiento de peor nodo informativo. |

---

## 📏 Relative Landmarking

Versión ordenada por ranking de los valores de landmarking (mejor a peor), útil para comparar performances relativas de métodos simples.

*(Ejemplo: `best_node.mean.relative`, etc.)*

---

## 🔍 Clustering

Índices de clusterización sobre todo el dataset.

| Meta-feature | Qué mide |
|-------------|----------|
| ch | Índice de Calinski-Harabasz (cluster separation). |
| int | Índice INT (diversidad de clusters). |
| nre | Entropía relativa normalizada. |
| pb | Correlación punto-biserial entre distancia e igualdad de clase. |
| sc | Número de clusters pequeños. |
| sil | Silhouette medio. |
| vdb | Índice Davies-Bouldin. |
| vdu | Índice Dunn. |

---

## 🧩 Itemset

Asocia atributos binarios y sus relaciones.

| Meta-feature | Qué mide |
|-------------|----------|
| one_itemset | Medida de itemsets simples. |
| two_itemset | Itemsets de longitud 2. |

---

## 🌐 Concept

Mide variabilidad y densidad de clases en el espacio de instancias.

| Meta-feature | Qué mide |
|-------------|----------|
| cohesiveness | Densidad de distribución de ejemplos. |
| conceptvar | Variación de concepto entre clases. |
| impconceptvar | Versión mejorada de la variación de concepto. |
| wg_dist | Distancia ponderada que captura densidad o sparse. |