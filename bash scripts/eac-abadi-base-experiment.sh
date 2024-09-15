#!/bin/bash
#SBATCH -p fatnode

python ground-truth.py -galn galaxy_TNG_375401 -cm autogmm
python eac-clustering.py -galn galaxy_TNG_375401 -l single
#python eac-clustering.py -galn galaxy_TNG_611399 -l average
#python eac-clustering.py -galn galaxy_TNG_611399 -l complete
#python eac-clustering.py -galn galaxy_TNG_611399 -l weigthed
#python eac-clustering.py -galn galaxy_TNG_611399 -l ward

python optimize-lmaps.py -galn galaxy_TNG_375401
python plot-main-data.py -galn galaxy_TNG_375401 #-rso
