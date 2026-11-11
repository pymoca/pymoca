#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""

import os
import sys
import time

import pymoca.backends.sympy.generator as gen_sympy
from pymoca import ast
from pymoca import parser
from pymoca import tree

import pytest

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(TEST_DIR, "models")
GENERATED_DIR = os.path.join(TEST_DIR, "generated")


def _flush():
    sys.stdout.flush()
    sys.stdout.flush()
    time.sleep(0.01)


def test_estimator():
    with open(os.path.join(MODEL_DIR, "Estimator.mo"), "r") as f:
        txt = f.read()
    ast_tree = parser.parse(txt)
    text = gen_sympy.generate(ast_tree, "Estimator")
    with open(os.path.join(GENERATED_DIR, "Estimator.py"), "w") as f:
        f.write(text)
    from test.generated.Estimator import Estimator as Estimator

    e = Estimator()
    e.linearize_symbolic()
    e.linearize()
    res = e.simulate(x0=[1.0])  # noqa: F841
    _flush()


def test_spring():
    with open(os.path.join(MODEL_DIR, "SpringSystem.mo"), "r") as f:
        txt = f.read()
    ast_tree = parser.parse(txt)
    flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="SpringSystem"))
    print(flat_tree)
    text = gen_sympy.generate(ast_tree, "SpringSystem")
    with open(os.path.join(GENERATED_DIR, "Spring.py"), "w") as f:
        f.write(text)
    from test.generated.Spring import SpringSystem as SpringSystem

    e = SpringSystem()
    e.linearize_symbolic()
    e.linearize()
    res = e.simulate(x0=[1.0, 1.0])  # noqa: F841
    _flush()


def test_aircraft():
    with open(os.path.join(MODEL_DIR, "Aircraft.mo"), "r") as f:
        txt = f.read()
    ast_tree = parser.parse(txt)
    text = gen_sympy.generate(ast_tree, "Aircraft")
    with open(os.path.join(GENERATED_DIR, "Aircraft.py"), "w") as f:
        f.write(text)
    from test.generated.Aircraft import Aircraft as Aircraft

    e = Aircraft()
    e.linearize_symbolic()
    e.linearize()
    res = e.simulate()  # noqa: F841
    _flush()


def test_quad():
    with open(os.path.join(MODEL_DIR, "Quad.mo"), "r") as f:
        txt = f.read()
    ast_tree = parser.parse(txt)
    text = gen_sympy.generate(ast_tree, "Quad")
    with open(os.path.join(GENERATED_DIR, "Quad.py"), "w") as f:
        f.write(text)
    from test.generated.Quad import Quad as Quad

    e = Quad()
    e.linearize_symbolic()
    e.linearize()
    res = e.simulate()  # noqa: F841
    _flush()


@pytest.mark.skip
def test_connector():
    with open(os.path.join(MODEL_DIR, "Connector.mo"), "r") as f:
        txt = f.read()
    ast_tree = parser.parse(txt)
    # print(ast_tree)

    flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="Aircraft"))  # noqa: F841
    # print(flat_tree)

    walker = tree.TreeWalker()  # noqa: F841
    classes = ast_tree.classes  # noqa: F841
    root = ast_tree.classes["Aircraft"]  # noqa: F841

    _flush()


def test_time_builtin():
    """Tests Modelica `time` used in a model"""
    with open(os.path.join(MODEL_DIR, "SpringSystem.mo"), "r") as f:
        txt = f.read()
    ast_tree = parser.parse(txt)
    forced_spring_model = """
    model ForcedSpringSystem "SpringSystem with time-varying input force"
        SpringSystem sys;
    equation
        sys.u = 100.0*sin(2*time);
    end ForcedSpringSystem;
    """
    system_ast = parser.parse(forced_spring_model)
    ast_tree.extend(system_ast)
    flat_tree = tree.flatten(ast_tree, ast.ComponentRef(name="ForcedSpringSystem"))
    print(flat_tree)
    text = gen_sympy.generate(ast_tree, "ForcedSpringSystem")
    with open(os.path.join(GENERATED_DIR, "ForcedSpringSystem.py"), "w") as f:
        f.write(text)
    from test.generated.ForcedSpringSystem import ForcedSpringSystem as ForcedSpringSystem

    e = ForcedSpringSystem()
    e.linearize_symbolic()
    e.linearize()
    res = e.simulate(x0=[1.0, 1.0])  # noqa: F841
    _flush()


def test_subscripted_expression_rejected():
    """The SymPy backend does not support subscripts of a parenthesized expression."""
    ast_tree = parser.parse("model M Real v[2]; Real w; equation w = (2 * v)[1]; end M;")
    with pytest.raises(NotImplementedError, match="Subscripts of a parenthesized expression"):
        gen_sympy.generate(ast_tree, "M")


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__])
