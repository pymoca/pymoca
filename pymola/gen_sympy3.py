#!/usr/bin/env python
"""
Modelica compiler.
"""
from __future__ import print_function
import sys
import antlr4
import antlr4.Parser
import argparse

# compiler
from .generated.ModelicaLexer import ModelicaLexer
from .generated.ModelicaParser import ModelicaParser
from .generated.ModelicaListener import ModelicaListener

# sympy runtime
import sympy
import sympy.physics.mechanics as mech
from ordered_set import OrderedSet

#pylint: disable=invalid-name, no-self-use, missing-docstring, unused-variable, protected-access
#pylint: disable=too-many-public-methods


#=========================================================
# Commande Line Interface
#=========================================================

def main(argv):
    #pylint: disable=unused-argument
    "The main function"
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('out')
    parser.add_argument('-t', '--trace', action='store_true')
    parser.set_defaults(trace=False)
    args = parser.parse_args(argv)
    with open(args.src, 'r') as f:
        modelica_model = f.read()
    sympy_model = process_ast(modelica_model, trace=args.trace)

    with open(args.out, 'w') as f:
        f.write(sympy_model)

#=========================================================
# Process AST
#=========================================================

def process_ast(ast):
    # find declared symbols and parameters
    declared_symbols = OrderedSet()
    parameters = OrderedSet()
    inputs = OrderedSet()
    outputs = OrderedSet()
    for elem in ast['elist']:
        classdef = elem['classdef']
        rclassdef = elem['rclassdef']
        comp = elem['comp']
        rcomp = elem['rcomp']
        if classdef is not None:
            for comp in classdef['component_list']:
                if 'parameter' in classdef['type_prefix']:
                    parameters.add(str(comp))
                elif 'input' in classdef['type_prefix']:
                    inputs.add(str(comp))
                elif 'output' in classdef['type_prefix']:
                    outputs.add(str(comp))
                else:
                    declared_symbols.add(str(comp))

    # find states and symbols
    states = OrderedSet()
    symbols = OrderedSet()
    for eq_section in ast['eq_section']:
        for eq in eq_section['eqs']:
            eq_sympy = sympy.sympify(eq)
            for deriv in eq_sympy.atoms(sympy.Derivative):
                states.add(str(deriv.args[0].func))
            for sym in eq_sympy.atoms(sympy.Symbol):
                symbols.add(str(sym))
    # remove known variables to find algebraic variables
    for var_list in [states, parameters, inputs]:
        for var in var_list:
            symbols.remove(var)

    symbols.remove('t')

    #print('states', states)
    #print('symbols', symbols)
    #print('params', parameters)
        
    # build equations list
    alg_eq_list = []
    diff_eq_list = []
    for eq_section in ast['eq_section']:
        for eq in eq_section['eqs']:
            for state in states:
                eq = eq.replace('{:s}'.format(state),
                        '{:s}(t)'.format(state))
            for i in range(10):
                eq = eq.replace('(t)(t)', '(t)')
            sympy_eq = sympy.sympify(eq)
            if len(sympy_eq.atoms(sympy.Derivative)) > 0:
                diff_eq_list += [sympy_eq]
            else:
                alg_eq_list += [sympy_eq]
            
    f = sympy.Matrix(diff_eq_list)
    a = sympy.Matrix(alg_eq_list)
    u = sympy.Matrix([inputs]).T
    x = sympy.Matrix([sympy.sympify('{:s}(t)'.format(state))
        for state in states])
    y = sympy.Matrix([outputs]).T
    p = sympy.Matrix([parameters]).T
    return locals()

if __name__ == '__main__':
    main(sys.argv[1:])

# vi:ts=4 sw=4 et nowrap:
