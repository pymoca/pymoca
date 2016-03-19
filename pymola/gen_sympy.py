from __future__ import print_function, absolute_import, division, print_function, unicode_literals

import sys

from . import tree


class SympyGenerator(tree.TreeListener):
    indent='    '

    def __init__(self):
        self.src = {}

    def exitFile(self, tree):
        src = ""
        for key in sorted(tree.classes.keys()):
            src += self.src[tree.classes[key]]
        self.src[tree] = src

    def exitClass(self, tree):
        src = '''# DO NOT EDIT
# This file was generated with pymola.

import sympy
import sympy.physics.mechanics as mech

class {tree.name:s}(object):

{self.indent:s}t = sympy.symbols('t')
'''.format(**locals())

        src += "\n{self.indent:s}# states\n".format(**locals())
        for state in tree.states:
            src += "{self.indent:s}{state.name:s} = " \
                    "mech.dynamicsymbols('{state.name:s}')\n".format(**locals())

        src += "\n{self.indent:s}# parameters\n".format(**locals())
        for param in tree.parameters:
            src += "{self.indent:s}{param.name:s} = 0\n".format(**locals())

        src += "\n{self.indent:s}# constants\n".format(**locals())
        for const in tree.constants:
            src += "{self.indent:s}{const.name:s} = 0')\n".format(**locals())

        src += "\n{self.indent:s}# inputs\n".format(**locals())
        for input in tree.inputs:
            src += "{self.indent:s}{input.name:s} = sympy.symbols('{input.name:s}')\n".format(**locals())

        src += "\n{self.indent:s}# outputs\n".format(**locals())
        for output in tree.outputs:
            src += "{self.indent:s}{output.name:s} = sympy.symbols('{output.name:s}')\n".format(**locals())

        src += "\n{self.indent:s}# variables\n".format(**locals())
        for var in tree.variables:
            src += "{self.indent:s}{var.name:s} = sympy.symbols('{var.name:s}')\n".format(**locals())

        src += "\n{self.indent:s}# equations\n".format(**locals())
        src += "{self.indent:s}eqs = [\n{self.indent:s}{self.indent:s}".format(
                **locals()) + ',\n{self.indent:s}{self.indent:s}'.format(**locals()).join(
            [self.src[eq] for eq in tree.equations]) + '\n{self.indent:s}]\n'.format(**locals())
        self.src[tree] = src

    def exitExpression(self, tree):
        op = str(tree.operator)
        if op == 'der':
            src = '({tree.operands[0].name:s}).diff(t)'.format(**locals())
        else:
            src = "({operator:s} ".format(**tree.__dict__)
            for operand in tree.operands:
                src +=  self.src[operand]
            src += ")"
        self.src[tree] = src

    def exitPrimary(self, tree):
        self.src[tree] = "{value:s".format(**tree.__dict__)

    def exitComponentRef(self, tree):
        self.src[tree] = "{name:s}".format(**tree.__dict__)

    def exitSymbol(self, tree):
        self.src[tree] = "{name:s} = sympy.symbols('{name:s}')".format(**tree.__dict__)

    def exitEquation(self, tree):
        self.src[tree] = "{left:s} - ({right:s})".format(
            left=self.src[tree.left],
            right=self.src[tree.right])