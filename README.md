# <img alt="Pymoca" src="branding/icons/pymocalogo.svg" height="60">

A Modelica to computer algebra system (CAS) compiler written in python.


[![CI](https://github.com/pymoca/pymoca/workflows/CI/badge.svg)](https://github.com/pymoca/pymoca/actions?query=workflow%3ACI)
[![Coverage](https://codecov.io/gh/pymoca/pymoca/branch/master/graph/badge.svg)](https://codecov.io/gh/pymoca/pymoca)
[![DOI](https://zenodo.org/badge/20664755.svg)](https://zenodo.org/badge/latestdoi/20664755)


## Install

Python / PyPI:

```bash
pip install pymoca
```

Anaconda / Conda:
1. Install anaconda 3.

2. Setup environment.

```bash
./create_conda_env.sh enduser
. activate pymoca
jupyter notebook
```

## Examples
* [Sympy Example](test/notebooks/Spring.ipynb)
* [Casadi Example](test/notebooks/Casadi.ipynb)
* [ModelicaXML Example](test/notebooks/XML.ipynb)

## Roadmap

### Completed Tasks

* Parsing Modelica
* Sympy Simulation/CAS creation for simple models
* Casadi CAS creation for simple models

### TODO

* Gather requirements and unify Casadi model
* Lazy parsing for reading large libraries
* Support more of Modelica language elements

<!--- vim:ts=4:sw=4:expandtab:
!-->
