#!/bin/bash
conda update -q conda
# Useful for debugging any issues with conda
conda info -a
conda create -n pymola python=3.5
conda config --add channels conda-forge
conda install numpy scipy sympy coverage matplotlib gcc cython \
	jupyter lapack pydotplus casadi coveralls
source activate pymola
# for some reason conda install of casadi doesn't work correctly, works with pip
# note this installs within the conda env as well since we source pymola above
pip install antlr4-python3-runtime control slycot
