#!/bin/bash
# ---
#autoGMM base

python ground-truth.py -galn galaxy_TNG_490577 -cm auto -orm rcut
python ground-truth.py -galn galaxy_TNG_469438 -cm auto -orm rcut
python ground-truth.py -galn galaxy_TNG_468064 -cm auto -orm rcut
python ground-truth.py -galn galaxy_TNG_420815 -cm auto -orm rcut
python ground-truth.py -galn galaxy_TNG_386429 -cm auto -orm rcut
python ground-truth.py -galn galaxy_TNG_375401 -cm auto -orm rcut
python ground-truth.py -galn galaxy_TNG_389511 -cm auto -orm rcut
python ground-truth.py -galn galaxy_TNG_393336 -cm auto -orm rcut
python ground-truth.py -galn galaxy_TNG_405000 -cm auto -orm rcut

#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_490577
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_469438
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_468064
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_420815
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_405000
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_386429
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_375401
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_389511
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_393336

python fuzzy-clustering.py -galn galaxy_TNG_490577 -f 3 #6.5
python fuzzy-clustering.py -galn galaxy_TNG_469438 -f 3 #6.5
python fuzzy-clustering.py -galn galaxy_TNG_468064 -f 3 #6.5
python fuzzy-clustering.py -galn galaxy_TNG_420815 -f 3 #6.5
python fuzzy-clustering.py -galn galaxy_TNG_405000 -f 3 #6.5
python fuzzy-clustering.py -galn galaxy_TNG_386429 -f 3 #6.5
python fuzzy-clustering.py -galn galaxy_TNG_375401 -f 3 #6.5
python fuzzy-clustering.py -galn galaxy_TNG_389511 -f 3 #6.5
python fuzzy-clustering.py -galn galaxy_TNG_393336 -f 3 #6.5

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



# rcut
#galaxy_TNG_490577: 5
#galaxy_TNG_469438: 3
#galaxy_TNG_468064: 7.5
#galaxy_TNG_420815: 250
#galaxy_TNG_405000: 2 (feo cerca den 140, dif de .1)
#galaxy_TNG_386429: 10
#galaxy_TNG_375401: 500
#galaxy_TNG_389511: 500
#galaxy_TNG_393336: 2 (feo cerca den 140, dif de 0.08)


#promedio: 140
#mediana: 7.5

#galaxy_TNG_490577: 5.5
#galaxy_TNG_469438: 5.5
#galaxy_TNG_468064: 7
#galaxy_TNG_420815: 5.5
#galaxy_TNG_405000: 5.5
#galaxy_TNG_386429: 9.5
#galaxy_TNG_375401: 9.5
#galaxy_TNG_389511: 5.5
#galaxy_TNG_393336: 5.5

#promedio 6.5
