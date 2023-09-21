# <img alt="Pymoca" src="branding/icons/pymocalogo.svg" height="60">

A Modelica to computer algebra system (CAS) translator written in Python.

[![CI](https://github.com/pymoca/pymoca/workflows/CI/badge.svg)](https://github.com/pymoca/pymoca/actions?query=workflow%3ACI)
[![Coverage](https://codecov.io/gh/pymoca/pymoca/branch/master/graph/badge.svg)](https://codecov.io/gh/pymoca/pymoca)
[![DOI](https://zenodo.org/badge/20664755.svg)](https://zenodo.org/badge/latestdoi/20664755)

## Overview

Modelica is a language for modeling dynamical systems as reusable components that can be connected together to form a system model. The behavior of a component is described by equations and algorithms. Equations in Modelica can be closer to the mathematical form you might see in a text book, for instance `p*V = n*R*T`, rather than the assignment statements you see in Python. See https://modelica.org to learn about the Modelica language and ecosystem.

Pymoca is designed to be used as a library that reads and understands Modelica code (`pymoca.parser`) and provides programmatic access to an internal representation of the code called an Abstract Syntax Tree or AST (`pymoca.ast`). The AST is further processed to generate output in various formats (`pymoca.backends`). The `pymoca.tree` module provides functionality to transform the AST into a form that can be more easily used by the backends to generate the target output. In particular, `pymoca.tree` provides classes and functions to convert a hierarchical, object-oriented Modelica model of connected components into a "flat" system of equations and variables.

Pymoca supports a subset of [Modelica](https://modelica.org) as defined by the Pymoca [test suite](test). It contains backends to translate Modelica to [CasADi](https://casadi.org), [SymPy](https://www.sympy.org), and [ModelicaXML](https://github.com/modelica-association/ModelicaXML), but most development and usage has been with the CasADi backend.

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

## Examples

The `*_test.py` scripts in the [test suite](test) are the only actively maintained examples that show how to use Pymoca and the Modelica code that it can handle.

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

Some more interesting examples are in Jupyter notebooks in the repo:

* [Casadi Example](test/notebooks/Casadi.ipynb)
* [Sympy Example](test/notebooks/Spring.ipynb)
* [ModelicaXML Example](test/notebooks/XML.ipynb)

## Roadmap

See the [GitHub Projects](https://github.com/orgs/pymoca/projects) for plans. In particular, see the [Info Panel in the Modelica Flattening project](https://github.com/orgs/pymoca/projects/1/views/1?pane=info) for an overview of a project getting some current focus. Breaking API changes are expected.

<!--- vim:ts=4:sw=4:expandtab:
!-->
