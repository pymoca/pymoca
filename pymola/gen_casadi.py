from __future__ import print_function, absolute_import, division, print_function, unicode_literals
from . import tree
from . import ast

import os
import sys
import copy

import casadi as ca
import numpy as np
from .gen_numpy import NumpyGenerator

# TODO
#  - DLL export
#  - Metadata export

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def hashcompare(self, other):
    return cmp(hash(self), hash(other))

def equality(self, other):
    return hash(self) == hash(other)

ca.MX.__cmp__ = hashcompare
ca.MX.__eq__ = equality


op_map = {'*': "__mul__",
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


def name_flat(tree):
    return tree.name.replace('.', '__')


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

    def get_function(self):
        return ca.Function('check', [self.time] + self.states + self.der_states + self.alg_states + self.inputs + self.outputs + self.constants + self.parameters, self.equations)


class ForLoop:

    def __init__(self, generator, tree):
        self.tree = tree
        self.generator = generator
        i = tree.indices[0]
        e = i.expression
        start = e.start.value
        step = e.step.value
        stop = self.generator.get_int_parameter(e.stop)
        self.values = np.arange(start, stop + step, step)
        self.index_variable = ca.MX.sym(i.name)
        self.name = i.name
        self.indexed_symbols = {}

    def register_indexed_symbol(self, e, tree):
        self.generator.get_src(tree)  # ensure symbol is available
        self.indexed_symbols[e] = tree


class CasadiGenerator(NumpyGenerator):

    def __init__(self, root, class_name):
        super(CasadiGenerator, self).__init__()
        self.src = {}
        self.nodes = {"time": ca.MX.sym("time")}
        self.derivative = {}
        self.root = root
        self.class_name = class_name
        self.for_loops = []

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
            if len(s.prefixes) == 0 or len(s.prefixes) == 1 and 'flow' in s.prefixes:
                states += [s]
            else:
                for prefix in s.prefixes:
                    if prefix == 'constant':
                        constants += [s]
                    elif prefix == 'parameter':
                        if s.type.name != "Integer":
                            parameters += [s]
                    elif prefix == 'input':
                        inputs += [s]
                    elif prefix == 'output':
                        outputs += [s]

        for s in outputs:
            if s not in states:
                variables += [s]

        results = CasadiSysModel()
        ode_states = []
        alg_states = []
        for s in states:
            if self.src[s] in self.derivative:
                ode_states.append(s)
            else:
                alg_states.append(s)
        results.states = [self.get_src(e) for e in ode_states]
        results.der_states = [self.derivative[
            self.get_src(e)] for e in ode_states]
        results.alg_states = [self.get_src(e) for e in alg_states]
        results.constants = [self.get_src(e) for e in constants]
        results.constant_values = [self.get_src(e.value) for e in constants]
        results.parameters = [self.get_src(e) for e in parameters]
        results.inputs = [self.get_src(e) for e in inputs]
        results.outputs = [self.get_src(e) for e in outputs]
        results.equations = [self.get_src(e) for e in tree.equations]
        results.time = self.nodes["time"]
        self.results = results

    def exitExpression(self, tree):
        try:
            op = tree.operator.name
        except:
            op = str(tree.operator)

        if op == "*":
            op = "mtimes"
        if op.startswith("."):
            op = op[1:]

        n_operands = len(tree.operands)
        if op == 'der':
            orig = self.get_src(tree.operands[0])
            if orig in self.derivative:
                src = self.derivative[orig]
            else:
                s = ca.MX.sym("der_" + orig.name(), orig.sparsity())
                self.derivative[orig] = s
                self.nodes[s] = s
                src = s
        elif op in ['-'] and n_operands == 1:
            src = -self.get_src(tree.operands[0])
        elif op == '+':
            src = self.get_src(tree.operands[0])
            for i in tree.operands[1:]:
                src += self.get_src(i)
        elif op == 'mtimes':
            src = self.get_src(tree.operands[0])
            for i in tree.operands[1:]:
                src = ca.mtimes(src, self.get_src(i))
        elif op == 'transpose' and n_operands == 1:
            src = self.get_src(tree.operands[0]).T
        elif op == 'sum' and n_operands == 1:
            v = self.get_src(tree.operands[0])
            src = ca.sum1(v)
        elif op == 'linspace' and n_operands == 3:
            a = self.get_src(tree.operands[0])
            b = self.get_src(tree.operands[1])
            n_steps = self.get_int_parameter(tree.operands[2])
            src = ca.linspace(a, b, n_steps)
        elif op == 'fill' and n_operands == 2:
            val = self.get_src(tree.operands[0])
            n_row = self.get_int_parameter(tree.operands[1])
            src = val * ca.DM.ones(n_row)
        elif op == 'fill' and n_operands == 3:
            val = self.get_src(tree.operands[0])
            n_row = self.get_int_parameter(tree.operands[1])
            n_col = self.get_int_parameter(tree.operands[2])
            src = val * ca.DM.ones(n_row, n_col)
        elif op == 'zeros' and n_operands == 1:
            n_row = self.get_int_parameter(tree.operands[0])
            src = ca.DM.zeros(n_row)
        elif op == 'zeros' and n_operands == 2:
            n_row = self.get_int_parameter(tree.operands[0])
            n_col = self.get_int_parameter(tree.operands[1])
            src = ca.DM.zeros(n_row, n_col)
        elif op == 'ones' and n_operands == 1:
            n_row = self.get_int_parameter(tree.operands[0])
            src = ca.DM.ones(n_row)
        elif op == 'ones' and n_operands == 2:
            n_row = self.get_int_parameter(tree.operands[0])
            n_col = self.get_int_parameter(tree.operands[1])
            src = ca.DM.ones(n_row, n_col)
        elif op == 'identity' and n_operands == 1:
            n = self.get_int_parameter(tree.operands[0])
            src = ca.DM.eye(n)
        elif op == 'diagonal' and n_operands == 1:
            diag = self.get_src(tree.operands[0])
            n = len(diag)
            indices = list(range(n))
            src = ca.DM.triplet(indices, indices, diag, n, n)
        elif op == 'delay' and n_operands == 2:
            expr = self.get_src(tree.operands[0])
            delay_time = self.get_src(tree.operands[1])
            assert isinstance(expr, MX)
            src = ca.MX.sym('{}_delayed_{}'.format(
                expr.name, delay_time), expr.size1(), expr.size2())
        elif op in op_map and n_operands == 2:
            lhs = self.get_src(tree.operands[0])
            rhs = self.get_src(tree.operands[1])
            lhs_op = getattr(lhs, op_map[op])
            src = lhs_op(rhs)
        elif op in op_map and n_operands == 1:
            lhs = self.get_src(tree.operands[0])
            lhs_op = getattr(lhs, op_map[op])
            src = lhs_op()
        elif n_operands == 1:
            src = self.get_src(tree.operands[0])
            src = getattr(src, tree.operator.name)()
        elif n_operands == 2:
            lhs = self.get_src(tree.operands[0])
            rhs = self.get_src(tree.operands[1])
            print(tree)
            lhs_op = getattr(lhs, tree.operator.name)
            src = lhs_op(rhs)
        else:
            raise Exception("unknown")
        self.src[tree] = src

    def exitIfExpression(self, tree):
        assert(len(tree.conditions) + 1 == len(tree.expressions))

        src = self.get_src(tree.expressions[-1])
        for cond_index in range(len(tree.conditions)):
            cond = self.get_src(tree.conditions[-(cond_index + 1)])
            expr1 = self.get_src(tree.expressions[-(cond_index + 2)])

            src = ca.if_else(cond, expr1, src)

        self.src[tree] = src

    def get_symbol_size(self, tree):
        return 1

    def get_indexed_symbol(self, tree):
        s = self.nodes[name_flat(tree)]
        slice = self.get_int_parameter(tree.indices[0])
        print(slice)
        return s[np.array(slice) - 1]
        print("get_indexed_symbol", tree)

    def get_indexed_symbol_loop(self, tree):

        names = []
        for e in tree.indices:
            assert hasattr(e, "name")
            names.append(e.name)

        s = ca.MX.sym(tree.name + str(names), self.get_symbol_size(tree))
        print("self", tree, s, self.for_loops)
        for f in reversed(self.for_loops):
            if f.name in names:
                f.register_indexed_symbol(s, tree)

        return s

    def get_src(self, i):
        # TODO clean up
        if isinstance(i, ast.Symbol):
            return self.get_symbol(i)
        elif isinstance(i, ast.ComponentRef):
            if i in self.src:
                return self.src[i]

            tree = i
            if tree.name == "time":
                self.src[tree] = self.nodes["time"]
                return self.src[tree]

            for f in reversed(self.for_loops):
                if f.name == tree.name:
                    self.src[tree] = f.index_variable
                    return self.src[tree]

            tree = self.root.find_symbol(self.root.classes[self.class_name], i)
            self.src[i] = self.get_symbol(tree)
            tree = i

            if len(tree.indices) > 0 and len(self.for_loops) == 0:
                self.src[tree] = self.get_indexed_symbol(tree)
                return self.src[tree]
            elif len(tree.indices) > 0:
                self.src[tree] = self.get_indexed_symbol_loop(tree)
                return self.src[tree]

            return self.src[i]

        else:
            return self.src[i]

    def get_symbol(self, tree):
        assert isinstance(tree, ast.Symbol)

        try:
            return self.src[tree]
        except KeyError:
            size = [self.get_int_parameter(d) for d in tree.dimensions]
            assert(len(size) <= 2)
            for i in tree.type.indices:
                assert len(size) == 1
                size = [size[0] * self.get_int_parameter(i)]
            s = ca.MX.sym(name_flat(tree), *size)
            self.nodes[name_flat(tree)] = s
            self.src[tree] = s
            return s

    def get_int_parameter(self, i):
        if isinstance(i, ast.Primary):
            return int(i.value)
        if isinstance(i, ast.ComponentRef):
            # TODO dep symbols may not be parsed yet
            s = self.root.find_symbol(self.root.classes[self.class_name], i)
            assert(s.type.name == 'Integer')
            return self.get_int_parameter(s.value)
        if isinstance(i, ast.Expression):
            # Evaluate expression
            ast_walker = tree.TreeWalker()
            ast_walker.walk(self, i)

            # TODO dep symbols may not be parsed yet.
            # TODO parse on demand?  self.get_symbol querying cache, calling
            # exitSymbol() otherwise?  Can then drop OrderedDict as well.
            expr = self.get_src(i)
            deps = [expr.dep(i) for i in range(expr.n_dep())
                    if expr.dep(i).is_symbolic()]
            dep_values = [self.get_int_parameter(self.root.find_symbol(self.root.classes[
                                                 self.class_name], ast.ComponentRef(name=dep.name())).value) for dep in deps if dep.is_symbolic()]
            F = ca.Function('get_int_parameter', deps, [expr])
            ret = F.call(dep_values)
            return int(ret[0])
        if isinstance(i, ast.Slice):
            start = self.get_int_parameter(i.start)
            step = self.get_int_parameter(i.step)
            stop = self.get_int_parameter(i.stop)
            return [int(e) for e in list(np.array(np.arange(start, stop + step, step)))]
        else:
            raise Exception(i)

    def exitEquation(self, tree):
        self.src[tree] = self.get_src(tree.left) - self.get_src(tree.right)

    def enterForEquation(self, tree):
        self.for_loops.append(ForLoop(self, tree))

    def exitForEquation(self, tree):
        f = self.for_loops.pop()

        indexed_symbols = list(f.indexed_symbols.keys())
        args = [f.index_variable] + indexed_symbols
        expr = ca.vcat([ca.vec(self.get_src(e)) for e in tree.equations])
        free_vars = ca.symvar(expr)

        free_vars = [e for e in free_vars if e not in args]
        all_args = args + free_vars
        F = ca.Function('loop_body_' + f.name, all_args, [expr])

        indexed_symbols_full = [self.nodes[
            f.indexed_symbols[k].name] for k in indexed_symbols]
        Fmap = F.map("map", "serial", len(f.values), list(
            range(len(args), len(all_args))), [])
        res = Fmap.call([f.values] + indexed_symbols_full + free_vars)

        self.src[tree] = res[0].T

    def exitIfEquation(self, tree):
        assert(len(tree.equations) % (len(tree.conditions) + 1) == 0)

        equations_per_condition = int(
            len(tree.equations) / (len(tree.conditions) + 1))

        src = ca.vertcat(*[self.get_src(tree.equations[-(i + 1)])
                           for i in range(equations_per_condition)])
        for cond_index in range(len(tree.conditions)):
            cond = self.get_src(tree.conditions[-(cond_index + 1)])
            expr1 = ca.vertcat(*[self.get_src(tree.equations[-equations_per_condition * (
                cond_index + 1) - (i + 1)]) for i in range(equations_per_condition)])

            src = ca.if_else(cond, expr1, src)

        self.src[tree] = src


def generate(ast_tree, model_name):
    # create a walker
    ast_walker = tree.TreeWalker()

    flat_tree = tree.flatten(ast_tree, model_name)

    casadi_gen = CasadiGenerator(flat_tree, model_name)
    casadi_gen.src.update(casadi_gen.src)
    ast_walker.walk(casadi_gen, flat_tree)
    return casadi_gen.results
