# Apéndice
En esta carpeta se encuentran todas las gráficas y resultados calculados para la realización del trabajo. Dada la masiva cantidad de gráficas realizadas no nos fue posible incluirlas a todas en el mismo documento de la tesina. Por lo cual, incluimos dichos resultados en este repositorio.

La estructura del apéndice es la siguiente:
- `{clustering-method}`: Método de clustering utilizado en los experimentos. Estos son "Hierarchical Clustering", "Fuzzy Clustering" y "Evidence Accumulation Clustering".
	- `{ground-truth}`: Método utilizado para comparar los resultados (también determina el número de clusters a buscar). Estos son "Abadi" y "Auto Gaussian Mixture".
    	- `Base`: Corrida sin remover outliers
    	- `Isolation Forest`: Removiendo partículas con Isolation Forest
    	- `RCut`: Removiendo partículas con RCut

Además, contamos con algunos experimentos extra:
- `fuzzy clustering/abadi extra`: Experimento aumentando el valor de fuzzines utilizado por el algoritmo para entender cómo este afecta al buscar dos clusters.
- `hierarchical clustering/abadi-all_p`: Experimento realizado en el cual corremos "hierarchical clustering" comparándolo contra los resultados obtenidos por Abadi, pero utilizando todos los parámetros de circularidad en lugar de solo *eps* como hace este último. Dicho experimento fue descartados del trabajo una vez establecido el marco en el cual realizaríamos el análisis.
