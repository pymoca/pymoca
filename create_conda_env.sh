#!/bin/bash
conda update -q conda
# Useful for debugging any issues with conda
conda info -a
conda create -q -n pymola python=3.5 numpy scipy sympy coverage coveralls matplotlib gcc cython
source activate pymola
# for some reason conda install of casadi doesn't work correctly, works with pip
# note this installs within the conda env as well since we source pymola above
pip install casadi
