#!/bin/bash
# ---

#python ground-truth.py -galn galaxy_TNG_611399 -cm ab
#python fuzzy-clustering-hyperparams.py -galn galaxy_TNG_611399

#python optimize-lmaps.py -galn galaxy_TNG_611399
#python plot-main-data.py -galn galaxy_TNG_611399


#python optimize-lmaps.py -galn galaxy_TNG_405000
#python plot-main-data.py -galn galaxy_TNG_405000

python ground-truth.py -galn galaxy_TNG_490577 -cm auto -orm if
python ground-truth.py -galn galaxy_TNG_469438 -cm auto -orm if
python ground-truth.py -galn galaxy_TNG_468064 -cm auto -orm if
python ground-truth.py -galn galaxy_TNG_420815 -cm auto -orm if
python ground-truth.py -galn galaxy_TNG_386429 -cm auto -orm if
python ground-truth.py -galn galaxy_TNG_375401 -cm auto -orm if
python ground-truth.py -galn galaxy_TNG_389511 -cm auto -orm if
python ground-truth.py -galn galaxy_TNG_393336 -cm auto -orm if
python ground-truth.py -galn galaxy_TNG_405000 -cm auto -orm if

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

#galaxy_TNG_490577: 
#galaxy_TNG_469438: 
#galaxy_TNG_468064: 
#galaxy_TNG_420815: 
#galaxy_TNG_405000: 
#galaxy_TNG_386429: 
#galaxy_TNG_375401: 
#galaxy_TNG_389511: 
#galaxy_TNG_393336: 

#promedio 129

#isolation forest

#galaxy_TNG_375401: 500
#galaxy_TNG_386429: 7.5
#galaxy_TNG_389511: 1.5 (feo en 500)
#galaxy_TNG_393336: 500
#galaxy_TNG_405000: 3 / 500
#galaxy_TNG_420815: 5
#galaxy_TNG_468064: 5
#galaxy_TNG_469438: 5
#galaxy_TNG_490577: 3

#mediana 5

#galaxy_TNG_375401: 6.5
#galaxy_TNG_386429: 6.5
#galaxy_TNG_389511: 3.5
#galaxy_TNG_393336: 3.5
#galaxy_TNG_405000: 3.5
#galaxy_TNG_420815: 6.5
#galaxy_TNG_468064: 6.5
#galaxy_TNG_469438: 3.5
#galaxy_TNG_490577: 3.5


#promedio: 4.5
