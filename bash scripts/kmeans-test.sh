#!/bin/bash
#SBATCH -p fatnode

python ground-truth.py -galn galaxy_TNG_611399 -cm abadi
python kmeans-clustering.py -galn galaxy_TNG_611399 -l single
#python eac-clustering.py -galn galaxy_TNG_611399 -l average
#python eac-clustering.py -galn galaxy_TNG_611399 -l complete
#python eac-clustering.py -galn galaxy_TNG_611399 -l weigthed
#python eac-clustering.py -galn galaxy_TNG_611399 -l ward

python plot-kmeans.py -galn galaxy_TNG_611399 #-rso
