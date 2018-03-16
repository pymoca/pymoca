# <img alt="Pymoca" src="branding/icons/pymocalogo.svg" height="60">

A Modelica to computer algebra system (CAS) compiler written in python.


[![Travis](https://img.shields.io/travis/pymoca/pymoca/master.svg?label=Travis%20CI)](https://travis-ci.org/pymoca/pymoca)
[![AppVeyor](https://img.shields.io/appveyor/ci/pymoca/pymoca/master.svg?label=AppVeyor)](https://ci.appveyor.com/project/pymoca/pymoca)
[![Coverage Status](https://img.shields.io/coveralls/pymoca/pymoca/master.svg)](https://coveralls.io/r/pymoca/pymoca)


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
