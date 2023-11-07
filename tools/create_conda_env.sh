#!/bin/bash
set -e
if [ "$#" -eq 1 ]; then
    config=$1
else
    echo "usage: $0 CONFIG"
    echo "         possible configs (testing, enduser)"
    exit 1
fi

conda update -q conda
# Useful for debugging any issues with conda
conda info -a
conda config --add channels conda-forge
conda create -n pymoca || echo "environment already created"
conda install -n pymoca jinja2 matplotlib numpy scipy sympy
source activate pymoca
# Note: casadi installed using pip since conda version currently
# not found by setup.py
python -m pip install antlr4-python3-runtime==4.13.1 casadi

if [ "$config" == "testing" ]; then
    conda install -n pymoca coverage coveralls
elif [ "$config" == "enduser" ]; then
    conda install -n pymoca jupyter lapack pydotplus control slycot
fi
