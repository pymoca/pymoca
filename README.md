# <img alt="Pymoca" src="https://raw.githubusercontent.com/pymoca/pymoca/refs/heads/master/branding/icons/pymocalogo.svg" height="60">
A Modelica to computer algebra system (CAS) translator written in Python.

[![CI](https://github.com/pymoca/pymoca/workflows/CI/badge.svg)](https://github.com/pymoca/pymoca/actions?query=workflow%3ACI)
[![Coverage](https://codecov.io/gh/pymoca/pymoca/branch/master/graph/badge.svg)](https://codecov.io/gh/pymoca/pymoca)
[![DOI](https://zenodo.org/badge/20664755.svg)](https://zenodo.org/badge/latestdoi/20664755)

## Overview
Pymoca can be used in applications that need to translate [Modelica](https://modelica.org) mathematical models into other forms. Pymoca can "flatten" a model containing a connected set of components defined by object-oriented Modelica classes into a set of variables and simultaneous equations that are easier to further process for analysis or simulation. It is particularly suited to provide Modelica models in symbolic form to [computer algebra systems](https://en.wikipedia.org/wiki/Computer_algebra_system). A common use in this context is to provide differential and algebraic equations for use in [optimal control problems](https://en.wikipedia.org/wiki/Optimal_control). Pymoca can translate Modelica to [CasADi](https://web.casadi.org), [SymPy](https://www.sympy.org), and [ModelicaXML](https://github.com/modelica-association/ModelicaXML), but most development and usage has been with CasADi.

## Install

For parser support without backend dependencies:
```bash
pip install pymoca
```

Other options are:
```bash
pip install "pymoca[casadi]"    # CasADi backend dependencies
pip install "pymoca[sympy]"     # SymPy backend dependencies
pip install "pymoca[lxml]"      # ModelicaXML backend dependencies

pip install "pymoca[examples]"  # To run Jupyter notebook examples in the repo

pip install "pymoca[all]"       # All of the above
```

## Usage

Pymoca reads and understands Modelica code (`pymoca.parser`) and provides access to an internal representation of the code called an Abstract Syntax Tree or AST (`pymoca.ast`). The AST is further processed to generate output in various formats (`pymoca.backends`). The `pymoca.tree` module provides functionality to transform the AST into a form that can be more easily used by the backends to generate the target output. In particular, `pymoca.tree` provides classes and functions to convert a hierarchical, object-oriented Modelica model of connected components into a "flat" system of equations and associated variables, parameters, and constants. Pymoca error checking is not always complete or easy to understand, so it is better to develop the Modelica code with other tools and then use Pymoca for translation.

The [test suite](https://github.com/pymoca/pymoca/tree/master/test) contains examples showing how to use Pymoca and the subset of Modelica that it currently supports.

Here is an example using a simple spring and damper model from the test suite:

```Python
from pprint import pprint

import pymoca.parser
import pymoca.backends.casadi.generator as casadi_backend


MODELICA_MODEL = """
model Spring
    Real x, v_x;
    parameter Real c = 0.1;
    parameter Real k = 2;
equation
    der(x) = v_x;
    der(v_x) = -k*x - c*v_x;
end Spring;
"""

print("Modelica Model:\n", MODELICA_MODEL)

print("\nEquations from the parsed AST in a JSON representation:")
ast = pymoca.parser.parse(MODELICA_MODEL)
pprint(ast.to_json(ast.classes["Spring"].equations))

print("\nGenerated CasADi model:")
casadi_model = casadi_backend.generate(ast, "Spring")
print(casadi_model)
```

Some more interesting examples are in Jupyter notebooks:

* [Casadi Example](https://github.com/pymoca/pymoca/blob/master/test/notebooks/Casadi.ipynb)
* [Sympy Example](https://github.com/pymoca/pymoca/blob/master/test/notebooks/Spring.ipynb)

## Roadmap

See the [GitHub Projects](https://github.com/orgs/pymoca/projects) for plans. In particular, see the [Info Panel in the Modelica Flattening project](https://github.com/orgs/pymoca/projects/1/views/1?pane=info) for an overview of a project getting some current focus. Breaking API changes are expected.

<!--- vim:ts=4:sw=4:expandtab:
!-->
