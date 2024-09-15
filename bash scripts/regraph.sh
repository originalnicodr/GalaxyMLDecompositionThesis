#!/bin/bash
#SBATCH -p fatnode

python optimize-lmaps.py -galn galaxy_TNG_375401 
python plot-main-data.py -galn galaxy_TNG_375401

python optimize-lmaps.py -galn galaxy_TNG_469438 
python plot-main-data.py -galn galaxy_TNG_469438
