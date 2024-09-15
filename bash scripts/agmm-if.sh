#!/bin/bash

#debug
#python ground-truth.py -galn galaxy_TNG_611399.h5 -cm auto -orm rc
#python hierarchical-clustering.py -galn galaxy_TNG_611399.h5 -n 3 -c
#python plot-main-data.py -galn galaxy_TNG_611399.h5

OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_490577.h5 -cm auto -orm if
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_469438.h5 -cm auto -orm if
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_468064.h5 -cm auto -orm if
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_420815.h5 -cm auto -orm if
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_386429.h5 -cm auto -orm if
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_375401.h5 -cm auto -orm if
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_389511.h5 -cm auto -orm if
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_393336.h5 -cm auto -orm if
OMP_NUM_THREADS=32 python ground-truth.py -galn galaxy_TNG_405000.h5 -cm auto -orm if

python hierarchical-clustering.py -galn galaxy_TNG_490577.h5
python hierarchical-clustering.py -galn galaxy_TNG_469438.h5
python hierarchical-clustering.py -galn galaxy_TNG_468064.h5
python hierarchical-clustering.py -galn galaxy_TNG_420815.h5
python hierarchical-clustering.py -galn galaxy_TNG_386429.h5
python hierarchical-clustering.py -galn galaxy_TNG_375401.h5
python hierarchical-clustering.py -galn galaxy_TNG_389511.h5
python hierarchical-clustering.py -galn galaxy_TNG_393336.h5
python hierarchical-clustering.py -galn galaxy_TNG_405000.h5

python plot-main-data.py -galn galaxy_TNG_490577.h5
python plot-main-data.py -galn galaxy_TNG_469438.h5
python plot-main-data.py -galn galaxy_TNG_468064.h5
python plot-main-data.py -galn galaxy_TNG_420815.h5
python plot-main-data.py -galn galaxy_TNG_386429.h5
python plot-main-data.py -galn galaxy_TNG_375401.h5
python plot-main-data.py -galn galaxy_TNG_389511.h5
python plot-main-data.py -galn galaxy_TNG_393336.h5
python plot-main-data.py -galn galaxy_TNG_405000.h5