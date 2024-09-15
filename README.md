# Galaxy Machine Learning Decomposition
Este repositorio contiene todos los scripts desarrollados para mi tesina de grado presentada en cumplimiento para el título de Licenciatura en Ciencias de la Computación, de la Universidad Nacional de Rosario. Titulada "Métodos alternativos de clustering para la descomposición dinámica de galaxias en simulaciones astrofísicas."

Puede encontrar el apéndice con todos los resultados obtenidos y graficados en la carpeta [apendice](https://github.com/originalnicodr/GalaxyMLDecompositionThesis/tree/main/apendice).

## Instalación
Para la correcta ejecución de los scripts primer necesitará instalar un env con lo siguiente:
- Python 3.10
- La branch `dev` de la libreria [galaxy-chop](https://github.com/vcristiani/galaxy-chop/tree/dev) (junto a sus dependencias)
- Este fork de la libreria [combo](https://github.com/originalnicodr/combo)
- Las dependencias encontradas en el archivo de `requirements.txt`

## Ejecución
Los resultados son serializados utilizando la librería *pickle*. Asegúrese de utilizar la misma versión de la librería para poder levantar los resultados serializados. Además, si se ejecuta un script con una galaxia pasada como parámetro esta debe existir en la carpeta `tests/datasets` encontrada en la misma locación que el script en sí.

A continuación una pequeña descripción de la función de cada script. Para más información sobre cada parámetro y sus valores por defecto ejecutar los scripts con la bandera `-h`.

Si se quiere correr alguno de los algoritmos de clustering descritos a continuación con eliminación de outliers primero deberá correr `ground-truth.py` con el método preferido. Los outliers serán serializados e identificados por los demás métodos antes de correr.

- **ground-truth.py**: Correr los métodos interpretados como ground truth en el trabajo (Abadi y AGMM) sobre una galaxia en particular:
	- `-galn`, `--galaxyname` (string): Nombre de la galaxia a utilizar (extensión incluida).
	- `-cm`, `--clusteringmethod` (string): El algoritmo de clustering a correr, puede ser "Abadi" o "AutoGMM". Referirse al código (y a los ejemplos de scripts de bash) para las abreviaciones de ambos.
	- `-orm`, `--outliersremovalmethod` (string): El algoritmo eliminación de outliers utilizado. Si no se le pasa ninguno no se eliminarán outliers. Acepta "RCut" e "Isolation Forest". Nuevamente, referirse al código (y a los ejemplos de scripts de bash) para las abreviaciones de ambos.
- **hierarchical-clustering.py**: Correr Hierarchical Clustering sobre una galaxia en particular.
	- `-galn`, `--galaxyname` (string): Nombre de la galaxia a utilizar (extensión incluida).
	- `-c`, `--complete`: Bandera para correr múltiples veces el algoritmo con todos los modos de linkage disponibles. Caso contrario utilizará solo "ward".
- **fuzzy-clustering-hyperparams.py**: Correr Fuzzy C-Means sobre una galaxia en particular explorando hiperparametros. Lamentablemente no parametricé la lista de hiperparametros, así que para modificarla va a tener que reescribirla dentro del script.
	- `-galn`, `--galaxyname` (string): Nombre de la galaxia a utilizar (extensión incluida).
- **fuzzy-clustering.py**: Correr Fuzzy C-Means sobre una galaxia en particular.
	- `-galn`, `--galaxyname` (string): Nombre de la galaxia a utilizar (extensión incluida).
	- `-e`, `--error` (float): Criterio de parada para el algoritmo.
	- `-m`, `--maxiter` (int): Máximo número de iteraciones permitidas para el algoritmo.
	- `-f`, `--fussiness` (float): Valor de fuzzines utilizado por el algoritmo.
- **eac-clustering.py**: Correr Evidence Accumulation Clustering sobre una galaxia en particular.
	- `-galn`, `--galaxyname` (string): Nombre de la galaxia a utilizar (extensión incluida).
	- `-l`, `--linkage` (string): El linkage utilizado por Hierarchical Clustering en el tercer paso de EAC. Puede ser "single", "complete", "average", "weighted", "median centroid", o "ward".
- **optimize-lmaps.py**: Optimiza el mapeo de labels sobre los clusters encontrados por los métodos propuestos en el trabajo al maximizar la métrica de recall.
	- `-galn`, `--galaxyname` (string): Nombre de la galaxia a utilizar (extensión incluida).
	- `-rd`, `--results_directory` (string): Carpeta en donde se encuentra la carpeta de la galaxia con sus resultados serializados.
- **plot-main-data.py**: Graficar scatterplots e histogramas del espacio real y circular
	- `-galn`, `--galaxyname` (string): Nombre de la galaxia a utilizar (extensión incluida).
- **plot-velocidad-circular.py**: Plotea la velocidad angular de la galaxia dada componente a componente, comparándolo así con las componentes obtenidas por el ground truth y los resultados de ambos con diferentes métodos de eliminación de outliers. Para más información sobre cómo debe estar estructurada la carpeta con los resultados revisar la estructura encontrada en la carpeta [apendice](https://github.com/originalnicodr/GalaxyMLDecompositionThesis/tree/main/apendice).
	- `-galn`, `--galaxyname` (string): Nombre de la galaxia a utilizar (extensión incluida).
	- `-rd`, `--results_directory` (string): Carpeta en donde se encuentra la carpeta de la galaxia con sus resultados serializados.
- **heatmap.py**: Grafica en una tabla en formato de heatmap las métricas internas (Davies–Bouldin y Silhouette) y las métricas externas (Recall y Precision) de los resultados obtenidos con uno de los métodos de clustering propuestos. Para más información sobre cómo debe estar estructurada la carpeta con los resultados revisar la estructura encontrada en la carpeta [apendice](https://github.com/originalnicodr/GalaxyMLDecompositionThesis/tree/main/apendice).
	- `-rd`, `--results_directory` (string): Carpeta en donde se encuentra la carpeta de la galaxia con sus resultados serializados.

---
Puede encontrar ejemplos de uso de estos scripts en la carpeta `bash scripts`.

Si tenes algun problema al intentar correr crea una pull request.
