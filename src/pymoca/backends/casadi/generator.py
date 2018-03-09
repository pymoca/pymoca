from __future__ import print_function, absolute_import, division, unicode_literals

import logging
from collections import namedtuple, deque, OrderedDict

import casadi as ca
import numpy as np
import itertools
from typing import Union, Dict

from pymoca import ast
from pymoca.tree import TreeWalker, TreeListener, flatten

from .alias_relation import AliasRelation
from .model import Model, Variable, DelayedState

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
          "abs": "fabs"}

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
        self.index_variable = ca.MX.sym(i.name)
        self.name = i.name
        self.indexed_symbols = {}

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
        self.map_mode = 'inline' if options.get('unroll_loops', True) else 'serial'
        self.function_mode = (True, False) if options.get('inline_functions', True) else (False, True)

    @property
    def current_class(self):
        return self.entered_classes[-1]

    def _ast_symbols_to_variables(self, ast_symbols, differentiate=False):
        variables = []
        for ast_symbol in ast_symbols:
            mx_symbol = self.get_mx(ast_symbol)
            if mx_symbol.is_empty():
                continue
            if differentiate:
                mx_symbol = self.get_derivative(mx_symbol)
            python_type = self.get_python_type(ast_symbol)
            variable = Variable(mx_symbol, python_type)
            if not differentiate:
                for a in ast.Symbol.ATTRIBUTES:
                    v = self.get_mx(getattr(ast_symbol, a))
                    if v is not None:
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
            delay_time = self.get_mx(tree.operands[1])
            if not isinstance(expr, ca.MX) or not expr.is_symbolic():
                # TODO
                raise NotImplementedError('Currently, delay() is only supported with a variable as argument.')
            src = ca.MX.sym('{}_delayed_{}'.format(
                expr.name(), delay_time), *expr.size())
            delayed_state = DelayedState(src.name(), expr.name(), delay_time)
            self.model.delayed_states.append(delayed_state)
            self.model.inputs.append(Variable(src))
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
            src = self.get_mx(tree.operands[0])
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
                orig_symbol = self.nodes[self.current_class][s.tree.name]
                indexed_symbol = orig_symbol[s.indices]
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
        return [self.get_integer(d) for d in tree.dimensions]

    def get_symbol(self, tree):
        # Create symbol
        shape = self.get_shape(tree)
        assert(len(shape) <= 2)

        if any(isinstance(x, slice) for x in shape):
            # Symbol has unspecified dimensions. Value is specified, and
            # carries the correct dimensions.

            # We should only get slices as dimensions for a symbol if one of
            # the dimensions is unspecified, i.e. None.
            assert None in itertools.chain.from_iterable((x.start, x.stop) for x in shape if isinstance(x, slice))

            val_shape = np.array(self.src[tree.value]).shape

            # Check if specified dimensions agree between definition and value
            for dim_i, dim_size in enumerate(shape):
                if isinstance(dim_size, slice):
                    continue

                if val_shape[dim_i] != dim_size:
                    raise Exception("Dimension {} of definition and value for symbol {} differs: {} != {}".format(
                        dim_i + 1, tree.name, dim_size, val_shape[dim_i]))

            shape = val_shape

        s = ca.MX.sym(tree.name, *shape)
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
                    der_s = ca.MX.sym("der({})".format(s.name()), s.size())
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
                der_dep = ca.MX.sym("der({})".format(dep.name()), dep.size())
                self.derivative[dep.name()] = der_dep
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
        # Check whether we loop over an index of this symbol
        indices = []
        for_loop = None
        for index in tree.indices:
            sl = None

            if isinstance(index, ast.ComponentRef):
                for f in self.for_loops:
                    if index.name == f.name:
                        # TODO support nested loops
                        for_loop = f
                        sl = for_loop.index_variable

            if sl is None:
                sl = self.get_integer(index)
                if isinstance(sl, int):
                    # Modelica indexing starts from one;  Python from zero.
                    sl = sl - 1
                elif isinstance(sl, slice):
                    # Modelica indexing starts from one;  Python from zero.
                    sl = slice(None if sl.start is None else sl.start - 1, sl.stop, sl.step)
                else:
                    for_loop = self.for_loops[-1]

            indices.append(sl)

        assert len(indices) <= 2, "Dimensions higher than two are not yet supported"

        if for_loop is not None:
            if isinstance(indices[0], ca.MX):
                if len(indices) > 1:
                    s = s[:, indices[1]]
                    assert s.size2() > 0, "Second dimension of matrix is zero"
                    indexed_symbol = ca.MX.sym('{}[{},{}]'.format(tree.name, for_loop.name, indices[1]), s.size2())
                    index_function = lambda i : (i, indices[1])
                else:
                    indexed_symbol = ca.MX.sym('{}[{}]'.format(tree.name, for_loop.name))
                    index_function = lambda i : i
                for_loop.register_indexed_symbol(indexed_symbol, index_function, True, tree, indices[0])
            else:
                s = ca.transpose(s[indices[0], :])
                assert s.size2() > 0, "Second dimension of matrix is zero"
                indexed_symbol = ca.MX.sym('{}[{},{}]'.format(tree.name, indices[0], for_loop.name), s.size2())
                index_function = lambda i: (indices[0], i)
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
        if len(tree.indices) > 0:
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
    if options is None:
        options = {}
    component_ref = ast.ComponentRef.from_string(model_name)
    ast_walker = TreeWalker()
    flat_tree = flatten(ast_tree, component_ref)
    component_ref_tuple = component_ref.to_tuple()
    casadi_gen = Generator(flat_tree, component_ref_tuple[-1], options)
    ast_walker.walk(casadi_gen, flat_tree)
    return casadi_gen.model
