from __future__ import print_function, absolute_import, division, unicode_literals

import logging
from collections import namedtuple, deque, OrderedDict

import casadi as ca
import numpy as np
import itertools
from typing import Union, Dict, Iterable

from pymoca import ast
from pymoca.tree import TreeWalker, TreeListener, flatten

from .alias_relation import AliasRelation
from .model import Model, Variable, DelayArgument
from .mtensor import _MTensor, _new_mx

from ._options import _merge_default_options

logger = logging.getLogger("pymoca")

# TODO
#  - Nested for loops
#  - Delay operator on arbitrary expressions
#  - Pre operator

OP_MAP = {'*': "__mul__",
          '+': "__add__",
          "-": "__sub__",
          "/": "__div__",
          '^': "__pow__",
          '>': '__gt__',
          '<': '__lt__',
          '<=': '__le__',
          '>=': '__ge__',
          '!=': '__ne__',
          '==': '__eq__',
          "min": "fmin",
          "max": "fmax",
          "abs": "fabs",
          "and": "__mul__",
          "or": "__add__"}

ForLoopIndexedSymbol = namedtuple('ForLoopIndexedSymbol', ['tree', 'transpose', 'indices'])


# noinspection PyPep8Naming,PyUnresolvedReferences
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
        self.index_variable = _new_mx(i.name)
        self.name = i.name
        self.indexed_symbols = OrderedDict()

    def register_indexed_symbol(self, e, index_function, transpose, tree, index_expr=None):
        if isinstance(index_expr, ca.MX) and index_expr is not self.index_variable:
            F = ca.Function('index_expr', [self.index_variable], [index_expr])
            # expr = lambda ar: np.array([F(a)[0] for a in ar], dtype=np.int)
            Fmap = F.map("map", self.generator.map_mode, len(self.values), [], [])
            res = Fmap.call([self.values])
            indices = np.array(res[0].T, dtype=np.int)
        else:
            indices = self.values
        self.indexed_symbols[e] = ForLoopIndexedSymbol(tree, transpose, index_function(indices - 1))


Assignment = namedtuple('Assignment', ['left', 'right'])


class GeneratorWalker(TreeWalker):
    """TreeWalker that skips processing of annotations"""

    def skip_child(self, tree: ast.Node, child_name: str) -> bool:
        skip = super().skip_child(tree, child_name)
        if isinstance(tree, ast.Class) and child_name == "annotation":
            return True
        return skip

    def order_keys(self, keys: Iterable[str]):
        # Symbols must come before classes, as we need to access symbol values when creating
        # CasADi interpolant functions.
        return sorted(keys, key=lambda attr: 0 if attr == 'symbols' else 1)


# noinspection PyPep8Naming,PyUnresolvedReferences
class Generator(TreeListener):
    def __init__(self, root: ast.Tree, class_name: str, options: Dict[str, bool]):
        super(Generator, self).__init__()
        self.src = {}
        self.model = Model()
        self.root = root
        c = self.root.classes[class_name]
        self.nodes = {c: {'time': self.model.time}}
        self.derivative = {}
        self.for_loops = deque()
        self.functions = {}
        self.entered_classes = deque()
        self.map_mode = 'inline' if options['unroll_loops'] else 'serial'
        self.function_mode = (True, False) if options['inline_functions'] else (False, True)
        self.delay_counter = 0

        # NOTE: Part of MTensor workaround.
        self._expand_vectors_enabled = options['expand_vectors']

    @property
    def current_class(self):
        return self.entered_classes[-1]

    def _ast_symbols_to_variables(self, ast_symbols, differentiate=False):
        variables = []
        for ast_symbol in ast_symbols:
            mx_symbol = self.get_mx(ast_symbol)
            modelica_shape = mx_symbol._modelica_shape
            if mx_symbol.is_empty():
                continue
            if differentiate:
                mx_symbol = self.get_derivative(mx_symbol)
                mx_symbol._modelica_shape = modelica_shape
            python_type = self.get_python_type(ast_symbol)
            variable = Variable(mx_symbol, python_type)
            if not differentiate:
                for a in ast.Symbol.ATTRIBUTES:
                    v = self.get_mx(getattr(ast_symbol, a))
                    if v is not None:
                        if isinstance(v, ca.DM) and all(x == (None,) for x in modelica_shape):
                            # Scalar numeric type that behaves like an array.
                            # Coerce to Pyhton type to avoid interpretation
                            # issues.
                            v = python_type(v)
                        elif isinstance(v, (float, int)) and not isinstance(v, python_type):
                            # We skip booleans for now, as users likely depend
                            # on them being integer/float-like.
                            try:
                                v = python_type(v)
                            except (OverflowError, ValueError):
                                # Cannot convert NaN/infs to integer
                                pass

                        setattr(variable, a, v)
                variable.prefixes = ast_symbol.prefixes
            variables.append(variable)
        return variables

    def enterClass(self, tree):
        logger.debug('enterClass {}'.format(tree.name))

        self.entered_classes.append(tree)
        self.nodes.setdefault(tree, {})

    def exitClass(self, tree):
        logger.debug('exitClass {}'.format(tree.name))

        if tree.type == 'function':
            # Already handled previously
            self.entered_classes.pop()
            return

        ode_states = []
        alg_states = []
        inputs = []
        constants = []
        parameters = []
        symbols = sorted(tree.symbols.values(), key=lambda x: x.order)
        for s in symbols:
            if 'constant' in s.prefixes:
                constants.append(s)
            elif 'parameter' in s.prefixes:
                parameters.append(s)
            elif 'input' in s.prefixes:
                inputs.append(s)
            elif 'state' in s.prefixes:
                ode_states.append(s)
            else:
                alg_states.append(s)

        self.model.states = self._ast_symbols_to_variables(ode_states)
        self.model.der_states = self._ast_symbols_to_variables(ode_states, differentiate=True)
        self.model.alg_states = self._ast_symbols_to_variables(alg_states)
        self.model.constants = self._ast_symbols_to_variables(constants)
        self.model.parameters = self._ast_symbols_to_variables(parameters)

        # We extend the input list, as it is already populated with delayed states.
        self.model.inputs.extend(self._ast_symbols_to_variables(inputs))

        # The outputs are a list of strings of state names. Specifying
        # multiple aliases of the same state is allowed.
        self.model.outputs = [v.symbol.name() for v in itertools.chain(self.model.states, self.model.alg_states) if 'output' in v.prefixes]

        def discard_empty(l):
            return list(filter(lambda x: not ca.MX(x).is_empty(), l))

        self.model.equations = discard_empty([self.get_mx(e) for e in tree.equations])
        self.model.initial_equations = discard_empty([self.get_mx(e) for e in tree.initial_equations])

        if len(tree.statements) + len(tree.initial_statements) > 0:
            raise NotImplementedError('Statements are currently supported inside functions only')

        self.entered_classes.pop()

    def exitArray(self, tree):
        self.src[tree] = [self.src[e] for e in tree.values]

    def exitPrimary(self, tree):
        self.src[tree] = tree.value

    def exitExpression(self, tree):
        if isinstance(tree.operator, ast.ComponentRef):
            op = tree.operator.name
        else:
            op = tree.operator

        if op == '*':
            op = 'mtimes'  # .* differs from *
        if op.startswith('.'):
            op = op[1:]

        logger.debug('exitExpression')

        n_operands = len(tree.operands)
        if op == 'der':
            v = self.get_mx(tree.operands[0])
            src = self.get_derivative(v)
        elif op == '-' and n_operands == 1:
            src = -self.get_mx(tree.operands[0])
        elif op == 'not' and n_operands == 1:
            src = ca.if_else(self.get_mx(tree.operands[0]), 0, 1, True)
        elif op == 'mtimes':
            assert n_operands >= 2
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
        elif op == 'cat':
            axis = self.get_integer(tree.operands[0])
            assert axis == 1, "Currently only concatenation on first axis is supported"

            entries = []
            for sym in [self.get_mx(op) for op in tree.operands[1:]]:
                if isinstance(sym, list):
                    for e in sym:
                        entries.append(e)
                else:
                    entries.append(sym)
            src = ca.vertcat(*entries)
        elif op == 'delay' and n_operands == 2:
            expr = self.get_mx(tree.operands[0])
            duration = self.get_mx(tree.operands[1])

            src = _new_mx('_pymoca_delay_{}'.format(self.delay_counter), *expr.size())
            self.delay_counter += 1

            for f in self.for_loops:
                syms = set(ca.symvar(expr))
                if syms.intersection(f.indexed_symbols):
                    f.register_indexed_symbol(src, lambda i: i, True, tree.operands[0], f.index_variable)

            self.model.delay_states.append(src.name())
            self.model.inputs.append(Variable(src))

            delay_argument = DelayArgument(expr, duration)
            self.model.delay_arguments.append(delay_argument)
        elif op == '_pymoca_interp1d' and n_operands >= 3 and n_operands <= 4:
            entered_class = self.entered_classes[-1]
            if isinstance(tree.operands[0], ast.ComponentRef):
                xp = self.get_mx(entered_class.symbols[tree.operands[0].name].value)
            else:
                xp = self.get_mx(tree.operands[0])
            if isinstance(tree.operands[1], ast.ComponentRef):
                yp = self.get_mx(entered_class.symbols[tree.operands[1].name].value)
            else:
                yp = self.get_mx(tree.operands[1])
            arg = self.get_mx(tree.operands[2])
            if n_operands == 4:
                assert isinstance(tree.operands[3], ast.Primary)
                mode = tree.operands[3].value
            else:
                mode = 'linear'
            func = ca.interpolant('interpolant', mode, [xp], yp)
            src = func(arg)
        elif op == '_pymoca_interp2d' and n_operands >= 5 and n_operands <= 6:
            entered_class = self.entered_classes[-1]
            if isinstance(tree.operands[0], ast.ComponentRef):
                xp = self.get_mx(entered_class.symbols[tree.operands[0].name].value)
            else:
                xp = self.get_mx(tree.operands[0])
            if isinstance(tree.operands[1], ast.ComponentRef):
                yp = self.get_mx(entered_class.symbols[tree.operands[1].name].value)
            else:
                yp = self.get_mx(tree.operands[1])
            if isinstance(tree.operands[2], ast.ComponentRef):
                zp = self.get_mx(entered_class.symbols[tree.operands[2].name].value)
            else:
                zp = self.get_mx(tree.operands[2])
            arg_1 = self.get_mx(tree.operands[3])
            arg_2 = self.get_mx(tree.operands[4])
            if n_operands == 6:
                assert isinstance(tree.operands[5], ast.Primary)
                mode = tree.operands[5].value
            else:
                mode = 'linear'
            func = ca.interpolant('interpolant', mode, [xp, yp], np.array(zp).ravel(order='F'))
            src = func(ca.vertcat(arg_1, arg_2))
        elif op in OP_MAP and n_operands == 2:
            lhs = ca.MX(self.get_mx(tree.operands[0]))
            rhs = ca.MX(self.get_mx(tree.operands[1]))
            lhs_op = getattr(lhs, OP_MAP[op])
            src = lhs_op(rhs)
        elif op in OP_MAP and n_operands == 1:
            lhs = ca.MX(self.get_mx(tree.operands[0]))
            lhs_op = getattr(lhs, OP_MAP[op])
            src = lhs_op()
        else:
            src = ca.MX(self.get_mx(tree.operands[0]))
            # Check for built-in operations, such as the
            # elementary functions, first.
            if hasattr(src, op) and n_operands <= 2:
                if n_operands == 1:
                    src = ca.MX(self.get_mx(tree.operands[0]))
                    src = getattr(src, op)()
                else:
                    lhs = ca.MX(self.get_mx(tree.operands[0]))
                    rhs = ca.MX(self.get_mx(tree.operands[1]))
                    lhs_op = getattr(lhs, op)
                    src = lhs_op(rhs)
            else:
                func = self.get_function(op)
                src = ca.vertcat(*func.call([self.get_mx(operand) for operand in tree.operands], *self.function_mode))

        self.src[tree] = src

    def exitIfExpression(self, tree):
        logger.debug('exitIfExpression')

        assert (len(tree.conditions) + 1 == len(tree.expressions))

        src = self.get_mx(tree.expressions[-1])
        for cond_index in range(len(tree.conditions)):
            cond = self.get_mx(tree.conditions[-(cond_index + 1)])
            expr1 = self.get_mx(tree.expressions[-(cond_index + 2)])

            src = ca.if_else(cond, expr1, src, True)

        self.src[tree] = src

    def exitEquation(self, tree):
        logger.debug('exitEquation')

        if isinstance(tree.left, list):
            src_left = ca.vertcat(*[self.get_mx(c) for c in tree.left])
        else:
            src_left = self.get_mx(tree.left)

        if isinstance(tree.right, list):
            src_right = ca.vertcat(*[self.get_mx(c) for c in tree.right])
        else:
            src_right = self.get_mx(tree.right)

        src_left = ca.MX(src_left)
        src_right = ca.MX(src_right)

        # According to the Modelica spec,
        # "It is possible to omit left hand side component references and/or truncate the left hand side list in order to discard outputs from a function call."
        if isinstance(tree.right, ast.Expression) and tree.right.operator in self.root.classes:
            if src_left.size1() < src_right.size1():
                src_right = src_right[0:src_left.size1()]
        if isinstance(tree.left, ast.Expression) and tree.left.operator in self.root.classes:
            if src_left.size1() > src_right.size1():
                src_left = src_left[0:src_right.size1()]

        # If dimensions between the lhs and rhs do not match, but the dimensions of lhs
        # and transposed rhs do match, transpose the rhs.
        if src_left.shape != src_right.shape and src_left.shape == src_right.shape[::-1]:
            src_right = ca.transpose(src_right)

        self.src[tree] = src_left - src_right

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
            F = ca.Function('loop_body', all_args, [expr])

            indexed_symbols_full = []
            for k in indexed_symbols:
                s = f.indexed_symbols[k]
                indices = s.indices
                try:
                    i = self.model.delay_states.index(k.name())
                except ValueError:
                    orig_symbol = self.nodes[self.current_class][s.tree.name]
                else:
                    # We are missing a similarly shaped delayed symbol. Make a new one with the appropriate shape.
                    delay_symbol = self.model.delay_arguments[i]

                    # We need to figure out the shape of the expression that
                    # we are delaying. The symbols that can occur in the delay
                    # expression should have been encountered before this
                    # iteration of the loop. The assert statement below covers
                    # this.
                    delay_expr_args = free_vars + all_args[:len(indexed_symbols_full)+1]
                    assert set(ca.symvar(delay_symbol.expr)).issubset(delay_expr_args)

                    f_delay_expr = ca.Function('delay_expr', delay_expr_args, [delay_symbol.expr])
                    f_delay_map = f_delay_expr.map("map", self.map_mode, len(f.values), list(
                        range(len(free_vars))), [])
                    [res] = f_delay_map.call(free_vars + [f.values] + indexed_symbols_full)
                    res = res.T

                    # Make the symbol with the appropriate size, and replace the old symbol with the new one.
                    orig_symbol = _new_mx(k.name(), *res.size())
                    assert res.size1() == 1 or res.size2() == 1, "Slicing does not yet work with 2-D indices"
                    indices = slice(None, None)

                    model_input = next(x for x in self.model.inputs if x.symbol.name() == k.name())
                    model_input.symbol = orig_symbol
                    self.model.delay_arguments[i] = DelayArgument(res, delay_symbol.duration)

                indexed_symbol = orig_symbol[indices]
                if s.transpose:
                    indexed_symbol = ca.transpose(indexed_symbol)
                indexed_symbols_full.append(indexed_symbol)

            Fmap = F.map("map", self.map_mode, len(f.values), list(
                range(len(args), len(all_args))), [])
            res = Fmap.call([f.values] + indexed_symbols_full + free_vars)

            self.src[tree] = res[0].T
        else:
            self.src[tree] = ca.MX()

    def exitIfEquation(self, tree):
        logger.debug('exitIfEquation')

        # Check if every equation block contains the same number of equations
        if len(set((len(x) for x in tree.blocks))) != 1:
            raise Exception("Every branch in an if-equation needs the same number of equations.")

        # NOTE: We currently assume that we always have an else-clause. This
        # is not strictly necessary, see the Modelica Spec on if equations.
        assert tree.conditions[-1] == True

        src = ca.vertcat(*[self.get_mx(e) for e in tree.blocks[-1]])

        for cond_index in range(1, len(tree.conditions)):
            cond = self.get_mx(tree.conditions[-(cond_index + 1)])
            expr1 = ca.vertcat(*[self.get_mx(e) for e in tree.blocks[-(cond_index + 1)]])
            src = ca.if_else(cond, expr1, src, True)

        self.src[tree] = src

    def exitAssignmentStatement(self, tree):
        logger.debug('exitAssignmentStatement')

        all_assignments = []

        expr = self.get_mx(tree.right)
        for component_ref in tree.left:
            all_assignments.append(Assignment(self.get_mx(component_ref), expr))

        self.src[tree] = all_assignments

    def exitIfStatement(self, tree):
        logger.debug('exitIfStatement')

        # We assume an equal number of statements per branch.
        # Furthermore, we assume that every branch assigns to the same variables.
        assert len(set((len(x) for x in tree.blocks))) == 1

        # NOTE: We currently assume that we always have an else-clause. This
        # is not strictly necessary, see the Modelica Spec on if statements.
        assert tree.conditions[-1] == True

        expanded_blocks = OrderedDict()

        for b in tree.blocks:
            block_assignments = []
            for s in b:
                assignments = self.get_mx(s)
                for assignment in assignments:
                    expanded_blocks.setdefault(assignment.left, []).append(assignment.right)

        assert len(set((len(x) for x in expanded_blocks.values()))) == 1

        all_assignments = []

        for lhs, values in expanded_blocks.items():
            # Set default value to else block, and then loop in reverse over all branches
            src = values[-1]
            for cond, rhs in zip(tree.conditions[-2::-1], values[-2::-1]):
                cond = self.get_mx(cond)
                src = ca.if_else(cond, rhs, src, True)

            all_assignments.append(Assignment(lhs, src))

        self.src[tree] = all_assignments

    def enterForStatement(self, tree):
        logger.debug('enterForStatement')

        self.for_loops.append(ForLoop(self, tree))

    def exitForStatement(self, tree):
        logger.debug('exitForStatement')

        f = self.for_loops.pop()
        if len(f.values) > 0:
            indexed_symbols = list(f.indexed_symbols.keys())
            args = [f.index_variable] + indexed_symbols
            expr = ca.vcat([ca.vec(self.get_mx(e.right)) for e in tree.statements])
            free_vars = ca.symvar(expr)

            arg_names = [arg.name() for arg in args]
            free_vars = [e for e in free_vars if e.name() not in arg_names]
            all_args = args + free_vars
            F = ca.Function('loop_body', all_args, [expr])

            indexed_symbols_full = []
            for k in indexed_symbols:
                s = f.indexed_symbols[k]
                orig_symbol = self.nodes[self.current_class][s.tree.name]
                indexed_symbol = orig_symbol[s.indices]
                if s.transpose:
                    indexed_symbol = ca.transpose(indexed_symbol)
                indexed_symbols_full.append(indexed_symbol)

            Fmap = F.map("map", self.map_mode, len(f.values), list(
                range(len(args), len(all_args))), [])
            res = Fmap.call([f.values] + indexed_symbols_full + free_vars)

            # Split into a list of statements
            variables = [assignment.left for statement in tree.statements for assignment in self.get_mx(statement)]
            all_assignments = []
            for i in range(len(f.values)):
                for j, variable in enumerate(variables):
                    all_assignments.append(Assignment(variable, res[0][j, i].T))

            self.src[tree] = all_assignments
        else:
            self.src[tree] = []

    def get_integer(self, tree: Union[ast.Primary, ast.ComponentRef, ast.Expression, ast.Slice]) -> Union[int, ca.MX, np.ndarray]:
        # CasADi needs to know the dimensions of symbols at instantiation.
        # We therefore need a mechanism to evaluate expressions that define dimensions of symbols.
        if isinstance(tree, ast.Primary):
            return None if tree.value is None else int(tree.value)
        if isinstance(tree, ast.ComponentRef):
            s = self.current_class.symbols[tree.name]
            assert (s.type.name == 'Integer')
            return self.get_integer(s.value)
        if isinstance(tree, ast.Expression):
            # Make sure that the expression has been converted to MX by (re)visiting the
            # relevant part of the AST.
            ast_walker = TreeWalker()
            ast_walker.walk(self, tree)

            # Obtain expression
            expr = self.get_mx(tree)

            # Obtain the symbols it depends on
            free_vars = ca.symvar(expr)

            # Find the values of the symbols
            vals = []
            for free_var in free_vars:
                if free_var.is_symbolic():
                    if (len(self.for_loops) > 0) and (free_var.name() == self.for_loops[-1].name):
                        vals.append(self.for_loops[-1].index_variable)
                    else:
                        vals.append(self.get_integer(self.current_class.symbols[free_var.name()].value))

            # Evaluate the expression
            F = ca.Function('get_integer', free_vars, [expr])
            ret = F.call(vals, *self.function_mode)
            if ret[0].is_constant():
                # We managed to evaluate the expression.  Assume the result to be integer.
                return int(ret[0])
            else:
                # Expression depends on other symbols.  Could not extract integer value.
                return ret[0]
        if isinstance(tree, ast.Slice):
            start = self.get_integer(tree.start)
            step = self.get_integer(tree.step)
            stop = self.get_integer(tree.stop)
            return slice(start, stop, step)
        else:
            raise Exception('Unexpected node type {}'.format(tree.__class__.__name__))

    @staticmethod
    def get_python_type(tree):
        if tree.type.name == 'Boolean':
            return bool
        elif tree.type.name == 'Integer':
            return int
        else:
            return float

    def get_shape(self, tree):
        return [[self.get_integer(d) for d in d_list] for d_list in tree.dimensions]

    def get_symbol(self, tree):
        # Create symbol
        shape = self.get_shape(tree)

        if any(isinstance(x, slice) for var_shape in shape for x in var_shape):
            # Symbol has unspecified dimensions. Value is specified, and
            # carries the correct dimensions.

            # We should only get slices as dimensions for a symbol if one of
            # the dimensions is unspecified, i.e. None.
            assert None in (itertools.chain.from_iterable((x.start, x.stop)
                            for var_shape in shape for x in var_shape if isinstance(x, slice)))

            val_shape = np.array(self.src[tree.value]).shape

            # Check if specified dimensions agree between definition and value
            val_dim_i = -1
            for var_i, var_shape in enumerate(shape):
                for dim_i, dim_size in enumerate(var_shape):
                    if dim_size is None:
                        continue

                    val_dim_i += 1
                    if isinstance(dim_size, slice):
                        shape[var_i][dim_i] = val_shape[val_dim_i]
                        continue

                    if val_shape[val_dim_i] != dim_size:
                        raise Exception("Dimension {} of definition and value for symbol {} "
                                        "differs: {} != {}"
                                        .format(val_dim_i + 1, tree.name, dim_size,
                                                val_shape[val_dim_i]))

        tensor_shape = [d for var_shape in shape for d in var_shape if d is not None]
        if len(tensor_shape) > 2:
            # MX does not support this, so we have to use our own wrapper.
            if not self._expand_vectors_enabled:
                raise NotImplementedError("Cannot handle 3D+ arrays without setting 'expand_vectors'")
            s = _MTensor(tree.name, *tensor_shape)
        else:
            s = _new_mx(tree.name, *tensor_shape)

        # Make a notion of the original shape, as MX is always 2D (even for 1D symbols),
        # and for nested classes we want to remember at which symbols to place indices.
        s._modelica_shape = tuple([tuple(var_shape) for var_shape in shape])

        self.nodes[self.current_class][tree.name] = s
        return s

    def get_derivative(self, s):

        # Case 1: s is a constant, e.g. MX(5)
        if ca.MX(s).is_constant():
            return 0

        # Case 2: s is a symbol, e.g. MX(x)
        elif s.is_symbolic():
            if s.name() not in self.derivative:
                if len(self.for_loops) > 0 and s in self.for_loops[-1].indexed_symbols:
                    # Create a new indexed symbol, referencing to the for loop index inside the vector derivative symbol.
                    for_loop_symbol = self.for_loops[-1].indexed_symbols[s]
                    s_without_index = self.get_mx(ast.ComponentRef(name=for_loop_symbol.tree.name))
                    der_s_without_index = self.get_derivative(s_without_index)
                    if ca.MX(der_s_without_index).is_symbolic():
                        return self.get_indexed_symbol(ast.ComponentRef(name=der_s_without_index.name(), indices=for_loop_symbol.tree.indices), der_s_without_index)
                    else:
                        return 0
                else:
                    der_s = _new_mx("der({})".format(s.name()), s.size())
                    # If the derivative contains an expression (e.g. der(x + y)) this method is
                    # called with MX variables that are the result of a ca.symvar call. This
                    # ca.symvar call strips the _modelica_shape field from the MX variable,
                    # therefore we need to find the original MX to get the modelica shape.
                    der_s._modelica_shape = \
                        self.nodes[self.current_class][s.name()]._modelica_shape
                    self.derivative[s.name()] = der_s
                    self.nodes[self.current_class][der_s.name()] = der_s
                    return der_s
            else:
                return self.derivative[s.name()]

        # Case 3: s is an already indexed symbol, e.g. MX(x[1])
        elif s.is_op(ca.OP_GETNONZEROS) and s.dep().is_symbolic():
            slice_info = s.info()['slice']
            dep = s.dep()
            if dep.name() not in self.derivative:
                der_dep = _new_mx("der({})".format(dep.name()), dep.size())
                der_dep._modelica_shape = \
                    self.nodes[self.current_class][dep.name()]._modelica_shape
                self.derivative[dep.name()] = der_dep
                self.nodes[self.current_class][der_dep.name()] = der_dep
                return der_dep[slice_info['start']:slice_info['stop']:slice_info['step']]
            else:
                return self.derivative[dep.name()][slice_info['start']:slice_info['stop']:slice_info['step']]

        # Case 4: s is an expression that requires differentiation, e.g. MX(x2 * x2)
        # Need to do this sort of expansion: der(x1 * x2) = der(x1) * x2 + x1 * der(x2)
        else:
            # Differentiate expression using CasADi
            orig_deps = ca.symvar(s)
            deps = ca.vertcat(*orig_deps)
            J = ca.Function('J', [deps], [ca.jacobian(s, deps)])
            J_sparsity = J.sparsity_out(0)
            der_deps = [self.get_derivative(dep) if J_sparsity.has_nz(0, j) else ca.DM.zeros(dep.size()) for j, dep in enumerate(orig_deps)]
            return ca.mtimes(J(deps), ca.vertcat(*der_deps))

    def get_indexed_symbol(self, tree, s):
        assert len([dim for shape in s._modelica_shape for dim in shape if dim is not None]) <= 2,\
            "Dimensions higher than two are not yet supported"

        assert len(s._modelica_shape) >= len(tree.indices)

        # For nested variables where an equation is defined at one of the nested models,
        # the modelica shape will contain the shape for the whole nested variable, but the indices
        # will only contain the indices for the symbol in the nested model. We only use the last
        # part of _modelica_shape in this case.
        assert tree.indices
        shapes = s._modelica_shape[-len(tree.indices):]

        # Check whether we loop over an index of this symbol
        indices = []
        for_loop = None
        for i, (index_array, shape) in enumerate(zip(tree.indices, shapes)):
            if len(index_array) > len(shape):
                symbol_name = s.name() if len(tree.indices) == 1 \
                    else s.name().split('.')[i] + ' in nested symbol ' + s.name()
                raise ValueError('Too many indices found for symbol {}, check if the symbol has '
                                 'the correct dimensions.'.format(symbol_name))

            for index, dim in zip(index_array, shape):
                if index is None and dim is None:
                    continue

                sl = None

                if isinstance(index, ast.ComponentRef):
                    for f in self.for_loops:
                        if index.name == f.name:
                            # TODO support nested loops
                            for_loop = f
                            sl = for_loop.index_variable

                if sl is None:
                    sl = self.get_integer(index) if index is not None else None

                    if sl is None and dim is not None:
                        sl = slice(None, None, 1)
                    if sl is not None and dim is None:
                        symbol_name = s.name() if len(tree.indices) == 1 \
                            else s.name().split('.')[i] + ' in nested symbol ' + s.name()
                        raise ValueError('Symbol {} was given an index of {} but this symbol '
                                         'is not an array.'.format(symbol_name, sl))
                    elif isinstance(sl, int):
                        # Modelica indexing starts from one;  Python from zero.
                        if sl <= 0 or sl > dim:
                            symbol_name = s.name() if len(tree.indices) == 1 \
                                else s.name().split('.')[i] + ' in nested symbol ' + s.name()
                            raise ValueError("Index {} of symbol {} is out of bounds. "
                                             "Index should be in range [1,{}] "
                                             "(Modelica uses 1-based indexing)."
                                             .format(sl, symbol_name, dim))
                        sl = sl - 1
                    elif isinstance(sl, slice):
                        # Modelica indexing starts from one;  Python from zero.
                        sl = slice(None if sl.start is None else sl.start - 1, sl.stop, sl.step)
                    else:
                        for_loop = self.for_loops[-1]

                indices.append(sl)

        if for_loop is not None:
            if isinstance(indices[0], ca.MX):
                if len(indices) > 1:
                    s = s[:, indices[1]]
                    indexed_symbol = _new_mx('{}[{},{}]'.format(tree.name, for_loop.name, indices[1]), s.size2())
                    index_function = lambda i : (i, indices[1])
                else:
                    indexed_symbol = _new_mx('{}[{}]'.format(tree.name, for_loop.name))
                    index_function = lambda i : i

                # If the indexed symbol is empty, we know we do not have to
                # map the for loop over it
                if np.prod(s.shape) != 0:
                    for_loop.register_indexed_symbol(indexed_symbol, index_function, True, tree, indices[0])
            else:
                s = ca.transpose(s[indices[0], :])
                indexed_symbol = _new_mx('{}[{},{}]'.format(tree.name, indices[0], for_loop.name), s.size2())
                index_function = lambda i: (indices[0], i)
                if np.prod(s.shape) != 0:
                    for_loop.register_indexed_symbol(indexed_symbol, index_function, False, tree, indices[1])
            return indexed_symbol
        else:
            if len(indices) == 1:
                return s[indices[0]]
            else:
                return s[indices[0], indices[1]]

    def get_component(self, tree):
        # Check special symbols
        if tree.name == 'time':
            return self.model.time
        else:
            for f in reversed(self.for_loops):
                if f.name == tree.name:
                    return f.index_variable

        # Check ordinary symbols
        symbol = self.current_class.symbols[tree.name]
        s = self.get_mx(symbol)
        if len([index for index_array in tree.indices
                for index in index_array if index is not None]) > 0:
            s = self.get_indexed_symbol(tree, s)
        return s

    def get_mx(self, tree: Union[ast.Symbol, ast.ComponentRef, ast.Expression]) -> ca.MX:
        """
        We pull components and symbols from the AST on demand.
        This is to ensure that parametrized vector dimensions can be resolved.  Vector
        dimensions need to be known at CasADi MX creation time.
        :param tree:
        :return:
        """
        if tree not in self.src:
            if isinstance(tree, ast.Symbol):
                s = self.get_symbol(tree)
            elif isinstance(tree, ast.ComponentRef):
                s = self.get_component(tree)
            else:
                raise Exception('Tried to look up expression before it was reached by the tree walker')
            self.src[tree] = s
        return self.src[tree]

    def get_function(self, function_name):
        if function_name in self.functions:
            return self.functions[function_name]

        try:
            tree = self.root.classes[function_name]
        except KeyError:
            raise Exception('Unknown function {}'.format(function_name))

        inputs = []
        outputs = []
        tmp = []
        for s in tree.symbols.values():
            src = self.get_mx(s)
            if 'input' in s.prefixes:
                inputs.append(src)
            elif 'output' in s.prefixes:
                outputs.append(src)
            else:
                tmp.append(src)

        # Store current variable values
        values = {}
        for variable in inputs:
            values[variable] = variable

        # Process statements in order
        for statement in tree.statements:
            src = self.get_mx(statement)
            for assignment in src:
                [values[assignment.left]] = ca.substitute([assignment.right], list(values.keys()), list(values.values()))

        output_expr = ca.substitute([values[output] for output in outputs], tmp, [values[t] for t in tmp])
        func = ca.Function(tree.name, inputs, output_expr)
        self.functions[function_name] = func

        return func


def generate(ast_tree: ast.Tree, model_name: str, options: Dict[str, bool]=None) -> Model:
    """
    :param ast_tree: AST to generate from
    :param model_name: class to generate
    :param options: dictionary of generator options
    :return: casadi model
    """
    options = _merge_default_options(options)


    component_ref = ast.ComponentRef.from_string(model_name)
    ast_walker = GeneratorWalker()
    flat_tree = flatten(ast_tree, component_ref)
    component_ref_tuple = component_ref.to_tuple()
    casadi_gen = Generator(flat_tree, component_ref_tuple[-1], options)
    ast_walker.walk(casadi_gen, flat_tree)
    return casadi_gen.model
