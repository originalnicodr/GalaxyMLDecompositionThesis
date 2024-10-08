#!/bin/bash

#debug
python ground-truth.py -galn galaxy_TNG_611399 -cm ab -orm if
python hierarchical-clustering.py -galn galaxy_TNG_611399
python plot-main-data.py -galn galaxy_TNG_611399 -rso

python ground-truth.py -galn galaxy_TNG_490577 -cm ab -orm if
python ground-truth.py -galn galaxy_TNG_469438 -cm ab -orm if
python ground-truth.py -galn galaxy_TNG_468064 -cm ab -orm if
python ground-truth.py -galn galaxy_TNG_420815 -cm ab -orm if
python ground-truth.py -galn galaxy_TNG_386429 -cm ab -orm if
python ground-truth.py -galn galaxy_TNG_375401 -cm ab -orm if
python ground-truth.py -galn galaxy_TNG_389511 -cm ab -orm if
python ground-truth.py -galn galaxy_TNG_393336 -cm ab -orm if
python ground-truth.py -galn galaxy_TNG_405000 -cm ab -orm if

python hierarchical-clustering.py -galn galaxy_TNG_490577 -p m
python hierarchical-clustering.py -galn galaxy_TNG_469438 -p m
python hierarchical-clustering.py -galn galaxy_TNG_468064 -p m
python hierarchical-clustering.py -galn galaxy_TNG_420815 -p m
python hierarchical-clustering.py -galn galaxy_TNG_386429 -p m
python hierarchical-clustering.py -galn galaxy_TNG_375401 -p m
python hierarchical-clustering.py -galn galaxy_TNG_389511 -p m
python hierarchical-clustering.py -galn galaxy_TNG_393336 -p m
python hierarchical-clustering.py -galn galaxy_TNG_405000 -p m

python plot-main-data.py -galn galaxy_TNG_490577 -rso
python plot-main-data.py -galn galaxy_TNG_469438 -rso
python plot-main-data.py -galn galaxy_TNG_468064 -rso
python plot-main-data.py -galn galaxy_TNG_420815 -rso
python plot-main-data.py -galn galaxy_TNG_386429 -rso
python plot-main-data.py -galn galaxy_TNG_375401 -rso
python plot-main-data.py -galn galaxy_TNG_389511 -rso
python plot-main-data.py -galn galaxy_TNG_393336 -rso
python plot-main-data.py -galn galaxy_TNG_405000 -rso