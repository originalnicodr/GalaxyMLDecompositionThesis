#!/bin/bash

#debug
#python ground-truth.py -galn galaxy_TNG_611399.h5 -cm auto -orm rc
#python hierarchical-clustering.py -galn galaxy_TNG_611399.h5 -n 3 -c
#python plot-main-data.py -galn galaxy_TNG_611399.h5

OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_490577 -cm auto -orm rc
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_469438 -cm auto -orm rc
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_468064 -cm auto -orm rc
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_420815 -cm auto -orm rc
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_386429 -cm auto -orm rc
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_375401 -cm auto -orm rc
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_389511 -cm auto -orm rc
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_393336 -cm auto -orm rc
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_405000 -cm auto -orm rc

python hierarchical-clustering.py -galn galaxy_TNG_490577
python hierarchical-clustering.py -galn galaxy_TNG_469438
python hierarchical-clustering.py -galn galaxy_TNG_468064
python hierarchical-clustering.py -galn galaxy_TNG_420815
python hierarchical-clustering.py -galn galaxy_TNG_386429
python hierarchical-clustering.py -galn galaxy_TNG_375401
python hierarchical-clustering.py -galn galaxy_TNG_389511
python hierarchical-clustering.py -galn galaxy_TNG_393336
python hierarchical-clustering.py -galn galaxy_TNG_405000

python plot-main-data.py -galn galaxy_TNG_490577 -rso
python plot-main-data.py -galn galaxy_TNG_469438 -rso
python plot-main-data.py -galn galaxy_TNG_468064 -rso
python plot-main-data.py -galn galaxy_TNG_420815 -rso
python plot-main-data.py -galn galaxy_TNG_386429 -rso
python plot-main-data.py -galn galaxy_TNG_375401 -rso
python plot-main-data.py -galn galaxy_TNG_389511 -rso
python plot-main-data.py -galn galaxy_TNG_393336 -rso
python plot-main-data.py -galn galaxy_TNG_405000 -rso