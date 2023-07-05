#!/bin/bash
python abadi.py -galn galaxy_TNG_490577.h5
python abadi.py -galn galaxy_TNG_469438.h5
python abadi.py -galn galaxy_TNG_468064.h5
python abadi.py -galn galaxy_TNG_420815.h5
python abadi.py -galn galaxy_TNG_386429.h5
python abadi.py -galn galaxy_TNG_375401.h5
python abadi.py -galn galaxy_TNG_389511.h5
python abadi.py -galn galaxy_TNG_393336.h5
python abadi.py -galn galaxy_TNG_405000.h5

python jerarquical-clustering.py -galn galaxy_TNG_490577.h5
python jerarquical-clustering.py -galn galaxy_TNG_469438.h5
python jerarquical-clustering.py -galn galaxy_TNG_468064.h5
python jerarquical-clustering.py -galn galaxy_TNG_420815.h5
python jerarquical-clustering.py -galn galaxy_TNG_386429.h5
python jerarquical-clustering.py -galn galaxy_TNG_375401.h5
python jerarquical-clustering.py -galn galaxy_TNG_389511.h5
python jerarquical-clustering.py -galn galaxy_TNG_393336.h5
python jerarquical-clustering.py -galn galaxy_TNG_405000.h5

#python jerarquical-clustering.py -galn galaxy_TNG_490577.h5 -c
#python jerarquical-clustering.py -galn galaxy_TNG_469438.h5 -c
#python jerarquical-clustering.py -galn galaxy_TNG_468064.h5 -c
#python jerarquical-clustering.py -galn galaxy_TNG_420815.h5 -c
#python jerarquical-clustering.py -galn galaxy_TNG_386429.h5 -c
#python jerarquical-clustering.py -galn galaxy_TNG_375401.h5 -c
#python jerarquical-clustering.py -galn galaxy_TNG_389511.h5 -c
#python jerarquical-clustering.py -galn galaxy_TNG_393336.h5 -c
#python jerarquical-clustering.py -galn galaxy_TNG_405000.h5 -c

#python heatmap.py "results_rcut" -lmap 0 1 0 1 1 0 1 0 0

#python plot-data.py -galn galaxy_TNG_490577.h5 -lmap 0 0
#python plot-data.py -galn galaxy_TNG_469438.h5 -lmap 0 1
#python plot-data.py -galn galaxy_TNG_468064.h5 -lmap 0 0
#python plot-data.py -galn galaxy_TNG_420815.h5 -lmap 0 1
#python plot-data.py -galn galaxy_TNG_386429.h5 -lmap 0 1
#python plot-data.py -galn galaxy_TNG_375401.h5 -lmap 0 0
#python plot-data.py -galn galaxy_TNG_389511.h5 -lmap 0 1
#python plot-data.py -galn galaxy_TNG_393336.h5 -lmap 0 0
#python plot-data.py -galn galaxy_TNG_405000.h5 -lmap 0 0