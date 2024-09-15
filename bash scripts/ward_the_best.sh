#!/bin/bash

#debug
#python ground-truth.py -galn galaxy_TNG_611399 -cm auto -orm rc
#python hierarchical-clustering.py -galn galaxy_TNG_611399 -c
#python optimize-lmaps.py -galn galaxy_TNG_611399
#python plot-main-data.py -galn galaxy_TNG_611399

python ground-truth.py -galn galaxy_TNG_611399 -cm auto
python fuzzy-clustering.py -galn galaxy_TNG_611399
python optimize-lmaps.py -galn galaxy_TNG_611399
python plot-main-data.py -galn galaxy_TNG_611399 -rso

python ground-truth.py -galn galaxy_TNG_611399 -cm abadi -orm rc
python fuzzy-clustering.py -galn galaxy_TNG_611399
python optimize-lmaps.py -galn galaxy_TNG_611399
python plot-main-data.py -galn galaxy_TNG_611399




python ground-truth.py -galn galaxy_TNG_375401 -cm abadi
python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_375401 -p m

python ground-truth.py -galn galaxy_TNG_375401 -cm abadi
python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_375401 -p a



python optimize-lmaps.py -galn galaxy_TNG_375401
python plot-main-data.py -galn galaxy_TNG_375401 -rso






OMP_NUM_THREADS=4 python ground-truth.py -galn galaxy_TNG_490577 -cm auto 
OMP_NUM_THREADS=4 python ground-truth.py -galn galaxy_TNG_469438 -cm auto 
OMP_NUM_THREADS=4 python ground-truth.py -galn galaxy_TNG_468064 -cm auto 
OMP_NUM_THREADS=4 python ground-truth.py -galn galaxy_TNG_420815 -cm auto 
OMP_NUM_THREADS=4 python ground-truth.py -galn galaxy_TNG_386429 -cm auto 
OMP_NUM_THREADS=4 python ground-truth.py -galn galaxy_TNG_375401 -cm auto 
OMP_NUM_THREADS=4 python ground-truth.py -galn galaxy_TNG_389511 -cm auto 
OMP_NUM_THREADS=4 python ground-truth.py -galn galaxy_TNG_393336 -cm auto 
OMP_NUM_THREADS=4 python ground-truth.py -galn galaxy_TNG_405000 -cm auto 

python hierarchical-clustering.py -galn galaxy_TNG_490577 -c -p a
python hierarchical-clustering.py -galn galaxy_TNG_469438 -c -p a
python hierarchical-clustering.py -galn galaxy_TNG_468064 -c -p a
python hierarchical-clustering.py -galn galaxy_TNG_420815 -c -p a
python hierarchical-clustering.py -galn galaxy_TNG_386429 -c -p a
python hierarchical-clustering.py -galn galaxy_TNG_375401 -c -p a
python hierarchical-clustering.py -galn galaxy_TNG_389511 -c -p a
python hierarchical-clustering.py -galn galaxy_TNG_393336 -c -p a
python hierarchical-clustering.py -galn galaxy_TNG_405000 -c -p a

python plot-main-data.py -galn galaxy_TNG_490577 -rso
python plot-main-data.py -galn galaxy_TNG_469438 -rso
python plot-main-data.py -galn galaxy_TNG_468064 -rso
python plot-main-data.py -galn galaxy_TNG_420815 -rso
python plot-main-data.py -galn galaxy_TNG_386429 -rso
python plot-main-data.py -galn galaxy_TNG_375401 -rso
python plot-main-data.py -galn galaxy_TNG_389511 -rso
python plot-main-data.py -galn galaxy_TNG_393336 -rso
python plot-main-data.py -galn galaxy_TNG_405000 -rso