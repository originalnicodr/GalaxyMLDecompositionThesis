#!/bin/bash
# ---

#python ground-truth.py -galn galaxy_TNG_611399 -cm ab
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_611399

#python optimize-lmaps.py -galn galaxy_TNG_611399
#python plot-main-data.py -galn galaxy_TNG_611399


#python optimize-lmaps.py -galn galaxy_TNG_405000
#python plot-main-data.py -galn galaxy_TNG_405000

python ground-truth.py -galn galaxy_TNG_490577 -cm abadi -orm rcut
python ground-truth.py -galn galaxy_TNG_469438 -cm abadi -orm rcut
python ground-truth.py -galn galaxy_TNG_468064 -cm abadi -orm rcut
python ground-truth.py -galn galaxy_TNG_420815 -cm abadi -orm rcut
python ground-truth.py -galn galaxy_TNG_386429 -cm abadi -orm rcut
python ground-truth.py -galn galaxy_TNG_375401 -cm abadi -orm rcut
python ground-truth.py -galn galaxy_TNG_389511 -cm abadi -orm rcut
python ground-truth.py -galn galaxy_TNG_393336 -cm abadi -orm rcut
python ground-truth.py -galn galaxy_TNG_405000 -cm abadi -orm rcut

python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_490577
python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_469438
python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_468064
python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_420815
python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_405000
python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_386429
python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_375401
python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_389511
python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_393336

# base

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


#galaxy_TNG_490577: 1.3 / 1.4
#galaxy_TNG_469438: 1.3
#galaxy_TNG_468064: 1.4
#galaxy_TNG_420815: 1.3
#galaxy_TNG_405000: 1.4
#galaxy_TNG_393336: 1.3
#galaxy_TNG_389511: 1.4
#galaxy_TNG_386429: 1.4
#galaxy_TNG_375401: 1.4

# promedio: 1.4

# rcut

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
#galaxy_TNG_468064: 1.2
#galaxy_TNG_420815: 1.3
#galaxy_TNG_405000: 1.4
#galaxy_TNG_393336: 1.3
#galaxy_TNG_389511: 1.01
#galaxy_TNG_386429: 1.3
#galaxy_TNG_375401: 1.4

# promedio: 1.3

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