#!/bin/bash

python assert-ward-linkage.py -galn galaxy_TNG_611399.h5 -l single
python assert-ward-linkage.py -galn galaxy_TNG_611399.h5 -l complete
python assert-ward-linkage.py -galn galaxy_TNG_611399.h5 -l average
python assert-ward-linkage.py -galn galaxy_TNG_611399.h5 -l ward
python abadi-results.py -galn -galn galaxy_TNG_611399.h5


