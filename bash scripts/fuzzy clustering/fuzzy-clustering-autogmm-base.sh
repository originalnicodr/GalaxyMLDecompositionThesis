#!/bin/bash
# ---
#autoGMM base

#python ground-truth.py -galn galaxy_TNG_490577 -cm auto
#python ground-truth.py -galn galaxy_TNG_469438 -cm auto
python ground-truth.py -galn galaxy_TNG_468064 -cm auto
python ground-truth.py -galn galaxy_TNG_420815 -cm auto
#python ground-truth.py -galn galaxy_TNG_405000 -cm auto
#python ground-truth.py -galn galaxy_TNG_386429 -cm auto
#python ground-truth.py -galn galaxy_TNG_375401 -cm auto
#python ground-truth.py -galn galaxy_TNG_389511 -cm auto
#python ground-truth.py -galn galaxy_TNG_393336 -cm auto

#python fuzzy-clustering.py -galn galaxy_TNG_490577 -f 3
#python fuzzy-clustering.py -galn galaxy_TNG_469438 -f 3
python fuzzy-clustering.py -galn galaxy_TNG_468064 -f 3
python fuzzy-clustering.py -galn galaxy_TNG_420815 -f 3
#python fuzzy-clustering.py -galn galaxy_TNG_405000 -f 3
#python fuzzy-clustering.py -galn galaxy_TNG_386429 -f 3
#python fuzzy-clustering.py -galn galaxy_TNG_375401 -f 3
#python fuzzy-clustering.py -galn galaxy_TNG_389511 -f 3
#python fuzzy-clustering.py -galn galaxy_TNG_393336 -f 3

#python optimize-lmaps.py -galn galaxy_TNG_490577
#python optimize-lmaps.py -galn galaxy_TNG_469438
#python optimize-lmaps.py -galn galaxy_TNG_468064
#python optimize-lmaps.py -galn galaxy_TNG_420815
#python optimize-lmaps.py -galn galaxy_TNG_405000
#python optimize-lmaps.py -galn galaxy_TNG_386429
#python optimize-lmaps.py -galn galaxy_TNG_375401
#python optimize-lmaps.py -galn galaxy_TNG_389511
#python optimize-lmaps.py -galn galaxy_TNG_393336


# RUN OUTSIDE

#python plot-main-data.py -galn galaxy_TNG_490577
#python plot-main-data.py -galn galaxy_TNG_469438
#python plot-main-data.py -galn galaxy_TNG_468064
#python plot-main-data.py -galn galaxy_TNG_420815
#python plot-main-data.py -galn galaxy_TNG_405000
#python plot-main-data.py -galn galaxy_TNG_386429
#python plot-main-data.py -galn galaxy_TNG_375401
#python plot-main-data.py -galn galaxy_TNG_389511
#python plot-main-data.py -galn galaxy_TNG_393336



#galaxy_TNG_490577: 5
#galaxy_TNG_469438: 3
#galaxy_TNG_468064: 5
#galaxy_TNG_420815: 3
#galaxy_TNG_405000: 3
#galaxy_TNG_386429: 2
#galaxy_TNG_375401: 3
#galaxy_TNG_389511: 1.5
#galaxy_TNG_393336: 3


#galaxy_TNG_490577: 5
#galaxy_TNG_469438: 2.75
#galaxy_TNG_468064: 4.5
#galaxy_TNG_420815: 3
#galaxy_TNG_405000: 2.5
#galaxy_TNG_386429: 2
#galaxy_TNG_375401: 2.75
#galaxy_TNG_389511: 1.5
#galaxy_TNG_393336: 3.5

#promedio = 3

python fuzzy-clustering.py -galn galaxy_TNG_375401 -f 3
