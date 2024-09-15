#!/bin/bash
# ---

#python ground-truth.py -galn galaxy_TNG_611399 -cm ab
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_611399

#python optimize-lmaps.py -galn galaxy_TNG_611399
#python plot-main-data.py -galn galaxy_TNG_611399


#python optimize-lmaps.py -galn galaxy_TNG_405000
#python plot-main-data.py -galn galaxy_TNG_405000

#python ground-truth.py -galn galaxy_TNG_490577 -cm abadi
#python ground-truth.py -galn galaxy_TNG_469438 -cm abadi
#python ground-truth.py -galn galaxy_TNG_468064 -cm abadi
#python ground-truth.py -galn galaxy_TNG_420815 -cm abadi
#python ground-truth.py -galn galaxy_TNG_386429 -cm abadi
#python ground-truth.py -galn galaxy_TNG_375401 -cm abadi
#python ground-truth.py -galn galaxy_TNG_389511 -cm abadi
#python ground-truth.py -galn galaxy_TNG_393336 -cm abadi
#python ground-truth.py -galn galaxy_TNG_405000 -cm abadi

#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_490577
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_469438
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_468064
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_420815
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_405000
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_386429
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_375401
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_389511
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_393336

python fuzzy-clustering.py -galn galaxy_TNG_490577 -f 1.4
python fuzzy-clustering.py -galn galaxy_TNG_469438 -f 1.4
python fuzzy-clustering.py -galn galaxy_TNG_468064 -f 1.4
python fuzzy-clustering.py -galn galaxy_TNG_420815 -f 1.4
python fuzzy-clustering.py -galn galaxy_TNG_405000 -f 1.4
python fuzzy-clustering.py -galn galaxy_TNG_386429 -f 1.4
python fuzzy-clustering.py -galn galaxy_TNG_375401 -f 1.4
python fuzzy-clustering.py -galn galaxy_TNG_389511 -f 1.4
python fuzzy-clustering.py -galn galaxy_TNG_393336 -f 1.4

# base

# rcut

# isolation forest

#galaxy_TNG_490577: 1.5
#galaxy_TNG_469438: 1.5
#galaxy_TNG_468064: 1.5
#galaxy_TNG_420815: 1.5
#galaxy_TNG_405000: 1.5
#galaxy_TNG_393336: 1.5
#galaxy_TNG_389511: 1.5
#galaxy_TNG_386429: 1.5
#galaxy_TNG_375401: 1.5

# promedio: 1.5

#galaxy_TNG_490577: 1.3
#galaxy_TNG_469438: 1.4
#galaxy_TNG_468064: 1.4
#galaxy_TNG_420815: 1.3
#galaxy_TNG_405000: 1.4
#galaxy_TNG_393336: 1.3
#galaxy_TNG_389511: 1.3
#galaxy_TNG_386429: 1.4
#galaxy_TNG_375401: 1.01

#promedio: 1.3