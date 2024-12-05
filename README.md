# <img alt="Pymoca" src="branding/icons/pymocalogo.svg" height="60">

A Modelica to computer algebra system (CAS) compiler written in python.


[![CI](https://github.com/pymoca/pymoca/workflows/CI/badge.svg)](https://github.com/pymoca/pymoca/actions?query=workflow%3ACI)
[![Coverage](https://codecov.io/gh/pymoca/pymoca/branch/master/graph/badge.svg)](https://codecov.io/gh/pymoca/pymoca)
[![DOI](https://zenodo.org/badge/20664755.svg)](https://zenodo.org/badge/latestdoi/20664755)

## Overview

Pymoca supports a subset of [Modelica](https://modelica.org) as defined by the Pymoca [test suite](test). While it contains backends for [CasADi](https://casadi.org), [SymPy](https://www.sympy.org/), and [ModelicaXML](https://github.com/modelica-association/ModelicaXML), most development and usage has been in support of the CasADi backend. The others are out of date and not actively maintained. Documentation is essentially the code itself and GitHub discussions.

## Install

Python / PyPI:

```bash
pip install pymoca
```

Anaconda / Conda:
1. Install anaconda 3.

2. Setup environment using [tools/create_conda_env.sh](tools/create_conda_env.sh):

```bash
tools/create_conda_env.sh enduser
. activate pymoca
jupyter notebook
```

## Examples

* [Casadi Example](test/notebooks/Casadi.ipynb)
* [Sympy Example](test/notebooks/Spring.ipynb)
* [ModelicaXML Example](test/notebooks/XML.ipynb)

## Roadmap

### Completed Tasks

* Parsing Modelica
* Sympy Simulation/CAS creation for simple models
* Casadi CAS creation for simple models

### TODO

* Implement support for MODELICAPATH as defined in the [Modelica Language Specification](https://modelica.org/documents/MLS.pdf) chapter 13
* Parse the [Modelica Standard Library 4.0](https://github.com/modelica/ModelicaStandardLibrary/tree/maint/4.0.x) without errors

See [GitHub Projects](https://github.com/pymoca/pymoca/projects?type=classic) and [Issues](https://github.com/pymoca/pymoca/issues) lists for more details and additional items.

<!--- vim:ts=4:sw=4:expandtab:
!-->
