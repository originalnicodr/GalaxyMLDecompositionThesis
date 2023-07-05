#!/bin/bash

#python abadi-isolation-forest.py -galn galaxy_TNG_490577.h5
#python abadi-isolation-forest.py -galn galaxy_TNG_469438.h5
#python abadi-isolation-forest.py -galn galaxy_TNG_468064.h5
#python abadi-isolation-forest.py -galn galaxy_TNG_420815.h5
#python abadi-isolation-forest.py -galn galaxy_TNG_386429.h5
#python abadi-isolation-forest.py -galn galaxy_TNG_375401.h5
#python abadi-isolation-forest.py -galn galaxy_TNG_389511.h5
#python abadi-isolation-forest.py -galn galaxy_TNG_393336.h5
#python abadi-isolation-forest.py -galn galaxy_TNG_405000.h5

#python jerarquical-clustering.py -galn galaxy_TNG_490577.h5
#python jerarquical-clustering.py -galn galaxy_TNG_469438.h5
#python jerarquical-clustering.py -galn galaxy_TNG_468064.h5
#python jerarquical-clustering.py -galn galaxy_TNG_420815.h5
#python jerarquical-clustering.py -galn galaxy_TNG_386429.h5
#python jerarquical-clustering.py -galn galaxy_TNG_375401.h5
#python jerarquical-clustering.py -galn galaxy_TNG_389511.h5
#python jerarquical-clustering.py -galn galaxy_TNG_393336.h5
#python jerarquical-clustering.py -galn galaxy_TNG_405000.h5

python plot-data.py -galn galaxy_TNG_490577.h5 -lmap 0 0
python plot-data.py -galn galaxy_TNG_469438.h5 -lmap 0 1
python plot-data.py -galn galaxy_TNG_468064.h5 -lmap 0 1
python plot-data.py -galn galaxy_TNG_420815.h5 -lmap 0 0
python plot-data.py -galn galaxy_TNG_386429.h5 -lmap 0 1
python plot-data.py -galn galaxy_TNG_375401.h5 -lmap 0 1
python plot-data.py -galn galaxy_TNG_389511.h5 -lmap 0 0
python plot-data.py -galn galaxy_TNG_393336.h5 -lmap 0 1
python plot-data.py -galn galaxy_TNG_405000.h5 -lmap 0 1

#python heatmap.py -rd "results_isolation_forest" -lmap 0 1 1 0 1 1 0 1 1
