from __future__ import print_function, absolute_import, division, print_function, unicode_literals
from . import tree

import os
import sys
import copy

import casadi as ca
import numpy as np
from .gen_numpy import NumpyGenerator

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def hashcompare(self,other):
  return cmp(hash(self),hash(other))


def equality(self,other):
  return hash(self)==hash(other)

ca.MX.__cmp__ = hashcompare
ca.MX.__eq__  = equality


op_map = {  '*':"__mul__",
            '+':"__add__",
            "-":"__sub__",
            "/":"__div__",
            "min":"fmin",
            "max":"fmax",
            "abs":"fabs"}

def name_flat(tree):
    return tree.name.replace('.','__')

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
        r+="Model\n"
        r+="time: " + str(self.time) + "\n"
        r+="states: " + str(self.states) + "\n"
        r+="der_states: " + str(self.der_states) + "\n"
        r+="alg_states: " + str(self.alg_states) + "\n"
        r+="inputs: " + str(self.inputs) + "\n"
        r+="outputs: " + str(self.outputs) + "\n"
        r+="constants: " + str(self.constants) + "\n"
        r+="constant_values: " + str(self.constant_values) + "\n"
        r+="parameters: " + str(self.parameters) + "\n"
        r+="equations: " + str(self.equations) + "\n"
        return r
    def get_function(self):
        return ca.Function('check',[self.time]+self.states+self.der_states+self.alg_states+self.inputs+self.outputs+self.constants+self.parameters,self.equations)

class ForLoop:
    def __init__(self, generator, tree):
        self.tree = tree
        self.generator = generator
        i = tree.indices[0]
        e = i.expression
        start = e.start.value
        step = e.step.value
        stop = self.generator.get_int_parameter(e.stop)
        self.values = np.arange(start, stop+step, step)
        self.index_variable = ca.MX.sym(i.name)
        self.name = i.name
        self.indexed_symbols = {}
    def register_indexed_symbol(self, e, tree):
        self.indexed_symbols[e] = tree

class CasadiGenerator(NumpyGenerator):

    def __init__(self, root, class_name):
        super(CasadiGenerator, self).__init__()
        self.src = {}
        self.nodes = {"time" :ca.MX.sym("time")}
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
            if len(s.prefixes) == 0 or len(s.prefixes)==1 and 'flow' in s.prefixes:
                states += [s]
            else:
                for prefix in s.prefixes:
                    if prefix == 'constant':
                        constants += [s]
                    elif prefix == 'parameter':
                        if s.type.name!="Integer":
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
        results.states = [self.src[e] for e in ode_states]
        results.der_states = [self.derivative[self.src[e]] for e in ode_states]
        results.alg_states = [self.src[e] for e in alg_states]
        results.constants = [self.src[e] for e in constants]
        results.constant_values = [self.src[e.value] for e in constants]
        results.parameters = [self.src[e] for e in parameters]
        results.inputs = [self.src[e] for e in inputs]
        results.outputs = [self.src[e] for e in outputs]
        results.equations = [self.src[e] for e in tree.equations]
        results.time = self.nodes["time"]
        self.results = results


    def exitExpression(self, tree):
        try:
            op = tree.operator.name
        except:
            op = str(tree.operator)

        if op=="*":
            op = "mtimes"
        if op.startswith("."):
            op = op[1:]

        n_operands = len(tree.operands)
        if op == 'der':
            orig = self.src[tree.operands[0]]
            if orig in self.derivative:
                src = self.derivative[orig]
            else:
                s = ca.MX.sym("der_"+orig.name(),orig.sparsity())
                self.derivative[orig] = s
                self.nodes[s] = s
                src = s
        elif op in ['-'] and n_operands == 1:
            src = -self.src[tree.operands[0]]
        elif op == '+':
            src = self.src[tree.operands[0]]
            for i in tree.operands[1:]:
                src += self.src[i]
        elif op == 'mtimes':
            src = self.src[tree.operands[0]]
            for i in tree.operands[1:]:
                src = ca.mtimes(src,self.src[i])
        elif op == 'transpose' and n_operands == 1:
            src = self.src[tree.operands[0]].T
        elif op in op_map and n_operands == 2:
            lhs = self.src[tree.operands[0]]
            rhs = self.src[tree.operands[1]]
            lhs_op = getattr(lhs,op_map[op])
            src = lhs_op(rhs)
        elif op in op_map and n_operands == 1:
            lhs = self.src[tree.operands[0]]
            lhs_op = getattr(lhs,op_map[op])
            src = lhs_op()
        elif n_operands == 1:
            src = self.src[tree.operands[0]]
            src = getattr(src,tree.operator.name)()
        elif n_operands == 2:
            lhs = self.src[tree.operands[0]]
            rhs = self.src[tree.operands[1]]
            print(tree)
            lhs_op = getattr(lhs,tree.operator.name)
            src = lhs_op(rhs)
        else:
            raise Exception("unknown")
        self.src[tree] = src

    def exitSlice(self, tree):
        start = self.src[tree.start]
        step = self.src[tree.step]
        stop = self.src[tree.stop]
        if isinstance(stop, ca.MX):
            stop = self.get_int_parameter(tree.stop)
        print(start, step, stop)
        self.src[tree] = [ int(e) for e in list(np.array(np.arange(start, stop+step, step))-1)]

    def exitComponentRef(self, tree):
        if tree.name=="Real":
            return
        if tree.name=="time":
            self.src[tree] = self.nodes["time"]
            return
        if len(tree.indices)>0 and len(self.for_loops)==0:
            self.src[tree] = self.get_indexed_symbol(tree)
            return
        elif len(tree.indices)>0:
            self.src[tree] = self.get_indexed_symbol_loop(tree)
            return

        try:
            self.src[tree] = self.nodes[name_flat(tree)]
        except:
            for f in reversed(self.for_loops):
                if f.name==tree.name:
                    self.src[tree] = f.index_variable

    def get_symbol_size(self,tree):
        return 1

    def get_indexed_symbol(self,tree):
        s = self.nodes[tree.name]
        slice = self.src[tree.indices[0]]
        print(slice)
        return s[slice]
        print("get_indexed_symbol",tree)

    def get_indexed_symbol_loop(self,tree):

        names = []
        for e in tree.indices:
            assert hasattr(e,"name")
            names.append(e.name)


        s = ca.MX.sym(tree.name+str(names), self.get_symbol_size(tree))
        print("self",tree, s, self.for_loops)
        for f in reversed(self.for_loops):
            if f.name in names:
                f.register_indexed_symbol(s, tree)

        return s


    def get_int_parameter(self, i):
        s = self.root.find_symbol(self.root.classes[self.class_name],i)
        assert(s.type.name=="Integer")
        return int(s.value.value)

    def enterSymbol(self, tree):
        size = [int(d.value) for d in tree.dimensions]
        assert(len(size)<=2)
        for i in tree.type.indices:
            assert len(size)==1
            size=[size[0]*self.get_int_parameter(i)]
        s =  ca.MX.sym(name_flat(tree), *size)
        self.nodes[name_flat(tree)] = s
        self.src[tree] = s

    def exitSymbol(self, tree):
        pass

    def exitEquation(self, tree):
        self.src[tree] = self.src[tree.left]-self.src[tree.right]

    def enterForEquation(self, tree):
        self.for_loops.append(ForLoop(self, tree))

    def exitForEquation(self, tree):
        f = self.for_loops.pop()

        indexed_symbols = list(f.indexed_symbols.keys())
        args = [f.index_variable]+indexed_symbols
        expr = ca.vcat([ca.vec(self.src[e]) for e in tree.equations])
        free_vars = ca.symvar(expr)

        free_vars = [e for e in free_vars if e not in args]
        all_args = args + free_vars
        F = ca.Function('loop_body_'+f.name,all_args,[expr])

        indexed_symbols_full = [self.nodes[f.indexed_symbols[k].name] for k in indexed_symbols]
        Fmap = F.map("map","serial",len(f.values),list(range(len(args),len(all_args))),[])
        res = Fmap.call([f.values]+indexed_symbols_full+free_vars)

        self.src[tree] = res[0].T

def generate(ast_tree, model_name):
    # create a walker
    ast_walker = tree.TreeWalker()

    flat_tree = tree.flatten(ast_tree, model_name)

    casadi_gen = CasadiGenerator(flat_tree, model_name)
    casadi_gen.src.update(casadi_gen.src)
    ast_walker.walk(casadi_gen, flat_tree)
    return casadi_gen.results
