from __future__ import print_function, absolute_import, division, print_function, unicode_literals
from . import tree

import jinja2
import os
import sys
import copy

import casadi as ca
import numpy as np
FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def hashcompare(self,other):
  return cmp(hash(self),hash(other))


def equality(self,other):
  return hash(self)==hash(other)

ca.MX.__cmp__ = hashcompare
ca.MX.__eq__  = equality


op_map = {'*':"__mul__", '+':"__add__","-":"__sub__","/":"__div__"}

def name_flat(tree):
    return tree.name.replace('.','__')

class CasadiSysModel:
    def __init__(self):
        self.states = []
        self.der_states = []
        self.inputs = []
        self.outputs = []
        self.constants = []
        self.parameters = []
        self.equations = []
    def __str__(self):
        r = ""
        r+="Model\n"
        r+="states: " + str(self.states) + "\n"
        r+="der_states: " + str(self.der_states) + "\n"
        r+="inputs: " + str(self.inputs) + "\n"
        r+="outputs: " + str(self.outputs) + "\n"
        r+="constants: " + str(self.constants) + "\n"
        r+="parameters: " + str(self.parameters) + "\n"
        r+="equations: " + str(self.equations) + "\n"
        return r
    def get_function(self):
        return ca.Function('check',self.states+self.der_states+self.inputs+self.outputs+self.constants+self.parameters,self.equations)

class CasadiGenerator(tree.TreeListener):

    def __init__(self):
        super(CasadiGenerator, self).__init__()
        self.src = {}
        self.nodes = {}
        self.derivative = {}

    def exitFile(self, tree):
        pass

    def exitClass(self, tree):
        states = []
        inputs = []
        outputs = []
        constants = []
        parameters = []
        variables = []
        symbols = sorted(tree.symbols.values(), key=lambda s: s.order)
        for s in symbols:
            if len(s.prefixes) == 0:
                states += [s]
            else:
                for prefix in s.prefixes:
                    if prefix == 'constant':
                        constants += [s]
                    elif prefix == 'parameter':
                        parameters += [s]
                    elif prefix == 'input':
                        inputs += [s]
                    elif prefix == 'output':
                        outputs += [s]


        for s in outputs:
            if s not in states:
                variables += [s]

        results = CasadiSysModel()
        results.states = [self.src[e] for e in states]
        results.der_states = [self.derivative[self.src[e]] for e in states]
        results.constants = [self.src[e] for e in constants]
        results.parameters = [self.src[e] for e in parameters]
        results.inputs = [self.src[e] for e in inputs]
        results.outputs = [self.src[e] for e in outputs]
        results.equations = [self.src[e] for e in tree.equations]
        self.results = results


    def exitExpression(self, tree):
        op = str(tree.operator)
        n_operands = len(tree.operands)

        if op == 'der':
            orig = self.src[tree.operands[0]]
            s = ca.MX.sym("der_"+orig.name(),orig.sparsity())
            self.derivative[orig] = s
            self.nodes[s] = s
            src = s
        elif op in op_map and n_operands == 2:
            lhs = self.src[tree.operands[0]]
            rhs = self.src[tree.operands[1]]
            lhs_op = getattr(lhs,op_map[op])
            src = lhs_op(rhs)
        elif op in ['+'] and n_operands == 1:
            src = self.src[tree.operands[0]]
        elif op in ['-'] and n_operands == 1:
            src = -self.src[tree.operands[0]]
        else:
            raise Exception("unknown")
        self.src[tree] = src

    def exitPrimary(self, tree):
        self.src[tree] = float(tree.value)

    def exitComponentRef(self, tree):
        self.src[tree] = self.nodes[name_flat(tree)]

    def enterSymbol(self, tree):
        s =  ca.MX.sym(name_flat(tree))
        self.nodes[name_flat(tree)] = s
        self.src[tree] = s

    def exitSymbol(self, tree):
        pass

    def exitEquation(self, tree):
        self.src[tree] = self.src[tree.left]-self.src[tree.right]


def generate(ast_tree, model_name):
    ast_tree_new = copy.deepcopy(ast_tree)
    ast_walker = tree.TreeWalker()
    flat_tree = tree.flatten(ast_tree_new, model_name)
    sympy_gen = CasadiGenerator()
    ast_walker.walk(sympy_gen, flat_tree)
    return sympy_gen.results
