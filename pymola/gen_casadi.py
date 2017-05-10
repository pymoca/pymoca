from __future__ import print_function, absolute_import, division, unicode_literals
from collections import namedtuple
from . import ast
from .tree import TreeWalker, flatten

import os
import sys
import copy
import logging
import itertools

import casadi as ca
import numpy as np
from .gen_numpy import NumpyGenerator

logger = logging.getLogger("pymola")

# TODO
#  - Nested for loops
#
#  - DLL export
#  - Metadata export

OP_MAP = {'*': "__mul__",
          '+': "__add__",
          "-": "__sub__",
          "/": "__div__",
          '>': '__gt__',
          '<': '__lt__',
          '<=': '__le__',
          '>=': '__ge__',
          '!=': '__ne__',
          '==': '__eq__',
          "min": "fmin",
          "max": "fmax",
          "abs": "fabs"}


class CasadiSysModel:

    def __init__(self):
        self.states = []
        self.der_states = []
        self.alg_states = []
        self.inputs = []
        self.outputs = []
        self.constants = []
        self.constant_values = []
        self.parameters = []
        self.equations = []
        self.time = ca.MX.sym('time')

    def __str__(self):
        r = ""
        r += "Model\n"
        r += "time: " + str(self.time) + "\n"
        r += "states: " + str(self.states) + "\n"
        r += "der_states: " + str(self.der_states) + "\n"
        r += "alg_states: " + str(self.alg_states) + "\n"
        r += "inputs: " + str(self.inputs) + "\n"
        r += "outputs: " + str(self.outputs) + "\n"
        r += "constants: " + str(self.constants) + "\n"
        r += "constant_values: " + str(self.constant_values) + "\n"
        r += "parameters: " + str(self.parameters) + "\n"
        r += "equations: " + str(self.equations) + "\n"
        return r

    def check_balanced(self):
        n_variables = sum(v.size1() * v.size2() for v in itertools.chain(self.states, self.alg_states))
        n_equations = sum(e.size1() * e.size2() for e in self.equations)
        if n_variables != n_equations:
            logger.warning("System is not balanced.  Number of variables is {}, number of equations is {}.".format(n_variables, n_equations))

    def get_function(self):
        return ca.Function('dae', [self.time, ca.vertcat(*self.states), ca.vertcat(*self.der_states), ca.vertcat(*self.alg_states), ca.vertcat(*self.constants), ca.vertcat(*self.parameters)], [ca.vertcat(*self.equations)])


ForLoopIndexedSymbol = namedtuple('ForLoopSymbol', ['tree', 'indices'])


class ForLoop:

    def __init__(self, generator, tree):
        self.tree = tree
        self.generator = generator
        i = tree.indices[0]
        e = i.expression
        start = e.start.value
        step = e.step.value
        stop = self.generator.get_integer(e.stop)
        self.values = np.arange(start, stop + step, step, dtype=np.int)
        self.index_variable = ca.MX.sym(i.name)
        self.name = i.name
        self.indexed_symbols = {}

    def register_indexed_symbol(self, e, tree, index_expr=None):
        if isinstance(index_expr, ca.MX):
            F = ca.Function('index_expr', [self.index_variable], [index_expr])
            expr = lambda ar : np.array([F(a)[0] for a in ar], dtype=np.int)
            Fmap = F.map("map", "serial", len(self.values), [], [])
            res = Fmap.call([self.values])
            indices = np.array(res[0].T, dtype=np.int)
        else:
            indices = self.values
        self.indexed_symbols[e] = ForLoopIndexedSymbol(tree, indices)


class CasadiGenerator(NumpyGenerator):

    def __init__(self, root, class_name):
        super(CasadiGenerator, self).__init__()
        self.src = {}
        self.model = CasadiSysModel()
        self.nodes = {'time': self.model.time}
        self.derivative = {}
        self.root = root
        self.class_name = class_name
        self.for_loops = []

    def exitClass(self, tree):
        logger.debug('exitClass {}'.format(tree.name))

        states = []
        inputs = []
        outputs = []
        constants = []
        parameters = []
        variables = []
        symbols = sorted(tree.symbols.values(), key=lambda s: s.order)
        for s in symbols:
            if 'constant' in s.prefixes:
                constants.append(s)
            elif 'parameter' in s.prefixes:
                parameters.append(s)
            else:
                states.append(s)
                if 'input' in s.prefixes:
                    inputs.append(s)
                elif 'output' in s.prefixes:
                    outputs.append(s)

        ode_states = []
        alg_states = []
        for s in states:
            if self.src[s] in self.derivative:
                ode_states.append(s)
            else:
                alg_states.append(s)
        self.model.states = [self.get_mx(e) for e in ode_states]
        self.model.der_states = [self.derivative[
            self.get_mx(e)] for e in ode_states]
        self.model.alg_states = [self.get_mx(e) for e in alg_states]
        self.model.constants = [self.get_mx(e) for e in constants]
        self.model.constant_values = [self.get_mx(e.value) for e in constants]
        self.model.parameters = [self.get_mx(e) for e in parameters]
        self.model.inputs = [self.get_mx(e) for e in inputs]
        self.model.outputs = [self.get_mx(e) for e in outputs]
        self.model.equations = [self.get_mx(e) for e in tree.equations]

    def exitExpression(self, tree):
        if isinstance(tree.operator, ast.ComponentRef):
            op = tree.operator.name
        else:
            op = tree.operator

        if op == '*':
            op = 'mtimes' # .* differs from *
        if op.startswith('.'):
            op = op[1:]

        logger.debug('exitExpression')

        n_operands = len(tree.operands)
        if op == 'der':
            orig = self.get_mx(tree.operands[0])
            if orig in self.derivative:
                src = self.derivative[orig]
            else:
                s = ca.MX.sym("der({})".format(orig.name()), orig.sparsity())
                self.derivative[orig] = s
                self.nodes[s] = s
                src = s
        elif op == '-' and n_operands == 1:
            src = -self.get_mx(tree.operands[0])
        elif op == 'mtimes':
            src = self.get_mx(tree.operands[0])
            for i in tree.operands[1:]:
                src = ca.mtimes(src, self.get_mx(i))
        elif op == 'transpose' and n_operands == 1:
            src = self.get_mx(tree.operands[0]).T
        elif op == 'sum' and n_operands == 1:
            v = self.get_mx(tree.operands[0])
            src = ca.sum1(v)
        elif op == 'linspace' and n_operands == 3:
            a = self.get_mx(tree.operands[0])
            b = self.get_mx(tree.operands[1])
            n_steps = self.get_integer(tree.operands[2])
            src = ca.linspace(a, b, n_steps)
        elif op == 'fill' and n_operands == 2:
            val = self.get_mx(tree.operands[0])
            n_row = self.get_integer(tree.operands[1])
            src = val * ca.DM.ones(n_row)
        elif op == 'fill' and n_operands == 3:
            val = self.get_mx(tree.operands[0])
            n_row = self.get_integer(tree.operands[1])
            n_col = self.get_integer(tree.operands[2])
            src = val * ca.DM.ones(n_row, n_col)
        elif op == 'zeros' and n_operands == 1:
            n_row = self.get_integer(tree.operands[0])
            src = ca.DM.zeros(n_row)
        elif op == 'zeros' and n_operands == 2:
            n_row = self.get_integer(tree.operands[0])
            n_col = self.get_integer(tree.operands[1])
            src = ca.DM.zeros(n_row, n_col)
        elif op == 'ones' and n_operands == 1:
            n_row = self.get_integer(tree.operands[0])
            src = ca.DM.ones(n_row)
        elif op == 'ones' and n_operands == 2:
            n_row = self.get_integer(tree.operands[0])
            n_col = self.get_integer(tree.operands[1])
            src = ca.DM.ones(n_row, n_col)
        elif op == 'identity' and n_operands == 1:
            n = self.get_integer(tree.operands[0])
            src = ca.DM.eye(n)
        elif op == 'diagonal' and n_operands == 1:
            diag = self.get_mx(tree.operands[0])
            n = len(diag)
            indices = list(range(n))
            src = ca.DM.triplet(indices, indices, diag, n, n)
        elif op == 'delay' and n_operands == 2:
            expr = self.get_mx(tree.operands[0])
            delay_time = self.get_mx(tree.operands[1])
            assert isinstance(expr, MX)
            src = ca.MX.sym('{}_delayed_{}'.format(
                expr.name, delay_time), expr.size1(), expr.size2())
        elif op in OP_MAP and n_operands == 2:
            lhs = self.get_mx(tree.operands[0])
            rhs = self.get_mx(tree.operands[1])
            lhs_op = getattr(lhs, OP_MAP[op])
            src = lhs_op(rhs)
        elif op in OP_MAP and n_operands == 1:
            lhs = self.get_mx(tree.operands[0])
            lhs_op = getattr(lhs, OP_MAP[op])
            src = lhs_op()
        elif n_operands == 1:
            src = self.get_mx(tree.operands[0])
            src = getattr(src, tree.operator.name)()
        elif n_operands == 2:
            lhs = self.get_mx(tree.operands[0])
            rhs = self.get_mx(tree.operands[1])
            lhs_op = getattr(lhs, tree.operator.name)
            src = lhs_op(rhs)
        else:
            raise Exception("Unknown operator {}({})".format(op, ','.join(n_operands * ['.'])))
        self.src[tree] = src

    def exitIfExpression(self, tree):
        logger.debug('exitIfExpression')

        assert(len(tree.conditions) + 1 == len(tree.expressions))

        src = self.get_mx(tree.expressions[-1])
        for cond_index in range(len(tree.conditions)):
            cond = self.get_mx(tree.conditions[-(cond_index + 1)])
            expr1 = self.get_mx(tree.expressions[-(cond_index + 2)])

            src = ca.if_else(cond, expr1, src)

        self.src[tree] = src

    def exitEquation(self, tree):
        logger.debug('exitEquation')

        self.src[tree] = self.get_mx(tree.left) - self.get_mx(tree.right)

    def enterForEquation(self, tree):
        logger.debug('enterForEquation')

        self.for_loops.append(ForLoop(self, tree))

    def exitForEquation(self, tree):
        logger.debug('exitForEquation')

        f = self.for_loops.pop()
        if len(f.values) > 0:
            indexed_symbols = list(f.indexed_symbols.keys())
            args = [f.index_variable] + indexed_symbols
            expr = ca.vcat([ca.vec(self.get_mx(e)) for e in tree.equations])
            free_vars = ca.symvar(expr)

            arg_names = [arg.name() for arg in args]
            free_vars = [e for e in free_vars if e.name() not in arg_names]
            all_args = args + free_vars
            F = ca.Function('loop_body_' + f.name, all_args, [expr])

            indexed_symbols_full = [self.nodes[
                f.indexed_symbols[k].tree.name][f.indexed_symbols[k].indices - 1] for k in indexed_symbols]
            Fmap = F.map("map", "serial", len(f.values), list(
                range(len(args), len(all_args))), [])
            res = Fmap.call([f.values] + indexed_symbols_full + free_vars)

            self.src[tree] = res[0].T
        else:
            self.src[tree] = ca.MX()

    def exitIfEquation(self, tree):
        logger.debug('exitIfEquation')

        assert(len(tree.equations) % (len(tree.conditions) + 1) == 0)

        equations_per_condition = int(
            len(tree.equations) / (len(tree.conditions) + 1))

        src = ca.vertcat(*[self.get_mx(tree.equations[-(i + 1)])
                           for i in range(equations_per_condition)])
        for cond_index in range(len(tree.conditions)):
            cond = self.get_mx(tree.conditions[-(cond_index + 1)])
            expr1 = ca.vertcat(*[self.get_mx(tree.equations[-equations_per_condition * (
                cond_index + 1) - (i + 1)]) for i in range(equations_per_condition)])

            src = ca.if_else(cond, expr1, src)

        self.src[tree] = src

    def get_integer(self, tree):
        # CasADi needs to know the dimensions of symbols at instantiation.
        # We therefore need a mechanism to evaluate expressions that define dimensions of symbols.
        if isinstance(tree, ast.Primary):
            return int(tree.value)
        if isinstance(tree, ast.ComponentRef):
            s = self.root.find_symbol(self.root.classes[self.class_name], tree)
            assert s.type.name == 'Integer'
            return self.get_integer(s.value)
        if isinstance(tree, ast.Expression):
            # Make sure that the expression has been converted to MX by (re)visiting the
            # relevant part of the AST.
            ast_walker = TreeWalker()
            ast_walker.walk(self, tree)

            # Obtain expression
            expr = self.get_mx(tree)

            # Obtain the symbols it depends on
            deps = [expr.dep(i) for i in range(expr.n_dep())
                    if expr.dep(i).is_symbolic()]

            # Find the values of the symbols
            vals = []
            for dep in deps:
                if dep.is_symbolic():
                    if (len(self.for_loops) > 0) and (dep.name() == self.for_loops[-1].name):
                        vals.append(self.for_loops[-1].index_variable)
                    else:
                        vals.append(self.get_integer(self.root.find_symbol(self.root.classes[self.class_name], 
                                    ast.ComponentRef(name=dep.name())).value))

            # Evaluate the expression
            F = ca.Function('get_integer_{}'.format('_'.join([dep.name().replace('.', '_') for dep in deps])), deps, [expr])
            ret = F.call(vals)
            if ret[0].is_constant():
                return int(ret[0])
            else:
                return ret[0]
        if isinstance(tree, ast.Slice):
            start = self.get_integer(tree.start)
            step = self.get_integer(tree.step)
            stop = self.get_integer(tree.stop)
            return np.arange(start, stop + step, step, dtype=np.int)
        else:
            raise Exception('Unexpected node type {}'.format(i.__class__.__name__))

    def get_symbol(self, tree):
        # Create symbol
        size = [self.get_integer(d) for d in tree.dimensions]
        assert(len(size) <= 2)
        s = ca.MX.sym(tree.name, *size)
        self.nodes[tree.name] = s
        return s

    def get_indexed_symbol(self, tree, s):
        # Check whether we loop over an index of this symbol
        indices = []
        for index in tree.indices:
            if isinstance(index, ast.ComponentRef):                
                for for_loop in self.for_loops:
                    if index.name == for_loop.name:
                        # TODO support nested loops
                        s = ca.MX.sym('{}[{}]'.format(tree.name, for_loop.name))
                        for_loop.register_indexed_symbol(s, tree)
                        return s

            sl = self.get_integer(index)
            if not isinstance(sl, int) and not isinstance(sl, np.ndarray):
                for_loop = self.for_loops[-1]
                s = ca.MX.sym('{}[{}]'.format(tree.name, for_loop.name))
                for_loop.register_indexed_symbol(s, tree, sl)
                return s

            # Modelica indexing starts from one;  Python from zero.
            indices.append(sl - 1)        
        if len(indices) == 1:
            return s[indices[0]]
        elif len(indices) == 2:
            return s[indices[0], indices[1]]
        else:
            raise Exception("Dimensions higher than two are not yet supported")

    def get_component(self, tree):
        # Check special symbols
        if tree.name == 'time':
            return self.model.time
        else:
            for f in reversed(self.for_loops):
                if f.name == tree.name:
                    return f.index_variable

        # Check ordinary symbols
        symbol = self.root.find_symbol(self.root.classes[self.class_name], tree)
        s = self.get_mx(symbol)
        if len(tree.indices) > 0:
            s = self.get_indexed_symbol(tree, s)
        return s

    def get_mx(self, tree):
        # We pull components and symbols from the AST on demand.  
        # This is to ensure that parametrized vector dimensions can be resolved.  Vector
        # dimensions need to be known at CasADi MX creation time.
        if tree not in self.src:
            if isinstance(tree, ast.Symbol):
                s = self.get_symbol(tree)
            elif isinstance(tree, ast.ComponentRef):
                s = self.get_component(tree)
            else:
                raise Exception('Tried to look up expression before it was reached by the tree walker')
            self.src[tree] = s
        return self.src[tree]


def generate(ast_tree, model_name):
    # create a walker
    ast_walker = TreeWalker()

    flat_tree = flatten(ast_tree, model_name)

    casadi_gen = CasadiGenerator(flat_tree, model_name)
    casadi_gen.src.update(casadi_gen.src)
    ast_walker.walk(casadi_gen, flat_tree)
    return casadi_gen.model
