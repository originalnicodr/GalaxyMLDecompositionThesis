#!/bin/bash

#debug
#python ground-truth.py -galn galaxy_TNG_611399 -cm ab
#python hierarchical-clustering.py -galn galaxy_TNG_611399 -c
#python plot-main-data.py -galn galaxy_TNG_611399 -rso

python ground-truth.py -galn galaxy_TNG_490577 -cm ab 
python ground-truth.py -galn galaxy_TNG_469438 -cm ab 
python ground-truth.py -galn galaxy_TNG_468064 -cm ab 
python ground-truth.py -galn galaxy_TNG_420815 -cm ab 
python ground-truth.py -galn galaxy_TNG_386429 -cm ab 
python ground-truth.py -galn galaxy_TNG_375401 -cm ab 
python ground-truth.py -galn galaxy_TNG_389511 -cm ab 
python ground-truth.py -galn galaxy_TNG_393336 -cm ab 
python ground-truth.py -galn galaxy_TNG_405000 -cm ab 

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