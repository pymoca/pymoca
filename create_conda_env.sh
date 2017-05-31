#!/bin/bash
set -e
conda update -q conda
conda env remove -n pymola || echo "no old pymola env found"
# Useful for debugging any issues with conda
conda info -a
conda config --add channels conda-forge
conda create -n pymola python=3.5 numpy scipy sympy coverage \
	matplotlib gcc cython jupyter lapack pydotplus casadi coveralls
source activate pymola
# for some reason conda install of casadi doesn't work correctly, works with pip
# note this installs within the conda env as well since we source pymola above
pip install antlr4-python3-runtime control slycot
