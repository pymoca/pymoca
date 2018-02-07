from __future__ import print_function, absolute_import, division, unicode_literals

import logging
from collections import namedtuple, deque, OrderedDict

import casadi as ca
import numpy as np
import itertools
import functools
from typing import Union, Dict

from pymola import ast
from pymola.tree import TreeWalker, TreeListener, flatten

from .alias_relation import AliasRelation
from .model import Model, Variable, DelayedState

logger = logging.getLogger("pymola")

# TODO
#  - Nested for loops
#  - Delay operator on arbitrary expressions
#  - Pre operator
#  - JSM style array definitions: array[:, :, 3]
#  - Type annotations (what functions return what type. Especially get_indexed_symbol, etc)
#  - Derivative as individual nodes, or one big node with array size of orig symbol
#  - Simplify other tests as well
OP_MAP = {'*': "__mul__",
          '+': "__add__",
          "-": "__sub__",
          "/": "__truediv__",
          '^': "__pow__"}

COMPARE_OP_MAP = {'>': '__gt__',
                  '<': '__lt__',
                  '<=': '__le__',
                  '>=': '__ge__',
                  '!=': '__ne__',
                  '==': '__eq__'}


class MXArray(np.ndarray):
    pass


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
        self.name = i.name
        self.current_value = None
        self.orig_keys = set(self.generator.src.keys())
        self.tmp_src = {}


Assignment = namedtuple('Assignment', ['left', 'right'])


def _safe_ndarray(x):
    if isinstance(x, ca.MX):
        # We want the single MX symbol in an array. If we try to do
        # np.array(x) directly in this case, we get an AttributeError:
        # "'MX' object has no attribute 'full'"
        return np.array([x], dtype=object)
    elif np.isscalar(x):
        return np.array([x], dtype=object)
    else:
        return np.array(x, dtype=object)


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
            mx_symbols = self.get_mx(ast_symbol)

            for ind, mx_symbol in np.ndenumerate(mx_symbols):
                if mx_symbol.is_empty():
                    continue
                if differentiate:
                    mx_symbol = self.get_derivative(_safe_ndarray(mx_symbol))[0]
                python_type = self.get_python_type(ast_symbol)
                variable = Variable(mx_symbol, python_type)
                if not differentiate:
                    for a in ast.Symbol.ATTRIBUTES:
                        v = self.get_mx(getattr(ast_symbol, a))
                        if v is not None:
                            if not np.isscalar(v):
                                # TODO: Why are some values still lists?
                                v = np.array(v)
                                v = v[ind]
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

        def expand_ndarray(l):
            new_l = []
            for e in l:
                if isinstance(e, np.ndarray):
                    for e_sub in e:
                        new_l.append(e_sub)
                else:
                    new_l.append(e)
            return new_l

        def discard_empty(l):
            return list(filter(lambda x: not ca.MX(x).is_empty(), l))

        self.model.equations = [self.get_mx(e) for e in tree.equations]
        self.model.initial_equations = [self.get_mx(e) for e in tree.initial_equations]

        self.model.equations = expand_ndarray(self.model.equations)
        self.model.initial_equations = expand_ndarray(self.model.initial_equations)

        self.model.equations = discard_empty(self.model.equations)
        self.model.initial_equations = discard_empty(self.model.initial_equations)

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
            # TODO: Probably does not work right for higher order (3D+) matrices.
            # See book.xogeny.com/behavior/arrays/functions/ under "Multiplication (* and .*)"
            assert n_operands >= 2
            src = self.get_mx(tree.operands[0])
            for i in tree.operands[1:]:
                rhs = self.get_mx(i)
                src = np.dot(src, rhs)
        elif op == 'transpose' and n_operands == 1:
            src = np.transpose(self.get_mx(tree.operands[0]))
        elif op == 'sum' and n_operands == 1:
            v = self.get_mx(tree.operands[0])
            src = sum(v)
        elif op == 'linspace' and n_operands == 3:
            a = self.get_mx(tree.operands[0])
            b = self.get_mx(tree.operands[1])
            n_steps = self.get_integer(tree.operands[2])
            src = np.linspace(a, b, n_steps)
        elif op == 'fill' and n_operands >= 2:
            val = self.get_mx(tree.operands[0])
            inds = tuple((self.get_integer(x) for x in tree.operands[1:]))
            src = np.full(inds, val)
        elif op == 'zeros':
            inds = tuple((self.get_integer(x) for x in tree.operands))
            src = np.zeros(inds)
        elif op == 'ones':
            inds = tuple((self.get_integer(x) for x in tree.operands))
            src = np.ones(inds)
        elif op == 'identity' and n_operands == 1:
            n = self.get_integer(tree.operands[0])
            src = np.eye(n)
        elif op == 'diagonal' and n_operands == 1:
            diag = self.get_mx(tree.operands[0])
            src = np.diag(diag)
        elif op in ("min", "max", "abs") and n_operands == 1:
            ca_op = getattr(ca.MX, "f" + op)
            src = _safe_ndarray(self.get_mx(tree.operands[0]))
            src = functools.reduce(ca_op, src.flat)
        elif op in ("min", "max", "abs") and n_operands == 2:
            ca_op = getattr(ca.MX, "f" + op)

            lhs = _safe_ndarray(self.get_mx(tree.operands[0]))
            rhs = _safe_ndarray(self.get_mx(tree.operands[1]))

            # Modelica Spec: Operation only allowed on scalars
            assert np.prod(lhs.shape) == 1 and np.prod(rhs.shape)

            src = ca_op(lhs[0], rhs[0])
        elif op == 'cat':
            axis = self.get_integer(tree.operands[0]) - 1
            entries  = []

            for sym in [self.get_mx(op) for op in tree.operands[1:]]:
                if isinstance(sym, list):
                    for e in sym:
                        entries.append(_safe_ndarray(e))
                else:
                    entries.append(_safe_ndarray(sym))

            src = np.concatenate(entries, axis)
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
            lhs = _safe_ndarray(self.get_mx(tree.operands[0]))
            rhs = _safe_ndarray(self.get_mx(tree.operands[1]))
            lhs_op = getattr(lhs, OP_MAP[op])
            src = lhs_op(rhs)
        elif op in OP_MAP and n_operands == 1:
            lhs = _safe_ndarray(self.get_mx(tree.operands[0]))
            lhs_op = getattr(lhs, OP_MAP[op])
            src = lhs_op()
        elif op in COMPARE_OP_MAP and n_operands == 2:
            lhs = _safe_ndarray(self.get_mx(tree.operands[0]))
            rhs = _safe_ndarray(self.get_mx(tree.operands[1]))

            # Only allowed for scalar expressions
            assert lhs.shape == (1,) and rhs.shape == (1,)

            lhs_op = getattr(lhs[0], COMPARE_OP_MAP[op])
            src = _safe_ndarray(lhs_op(rhs[0]))
        else:
            src = self.get_mx(tree.operands[0])
            # Check for built-in operations, such as the
            # elementary functions, first.

            if hasattr(ca.MX, op) and n_operands <= 2:
                f = getattr(ca.MX, op)

                if n_operands == 1:
                    src = np.array([f(x) for x in src])
                else:
                    lhs = self.get_mx(tree.operands[0])
                    rhs = self.get_mx(tree.operands[1])

                    src = np.array([f(a, b) for a, b in zip(lhs, rhs)])
            else:
                # TODO(Array)
                function = self.get_function(op)
                src = ca.vertcat(*function.call([self.get_mx(operand) for operand in tree.operands], *self.function_mode))

        self.src[tree] = src

    def exitIfExpression(self, tree):
        logger.debug('exitIfExpression')

        assert (len(tree.conditions) + 1 == len(tree.expressions))

        src = _safe_ndarray(self.get_mx(tree.expressions[-1]))

        for cond_index in range(len(tree.conditions)):
            cond = self.get_mx(tree.conditions[-(cond_index + 1)])
            assert cond.shape == (1,), "The expression of an if or elseif-clause must be a scalar Boolean expression"

            expr1 = _safe_ndarray(self.get_mx(tree.expressions[-(cond_index + 2)]))

            for ind, s_i in np.ndenumerate(src):
                src[ind] = ca.if_else(cond[0], expr1[ind], src[ind], True)

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

        src_left = _safe_ndarray(src_left)
        src_right = _safe_ndarray(src_right)

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

        if not self.for_loops or tree is not self.for_loops[-1].tree:
            self.for_loops.append(ForLoop(self, tree))

            f = self.for_loops[-1]

            for i in f.values[:-1]:
                f.current_value = i
                ast_walker = TreeWalker()
                ast_walker.walk(self, tree)

                # Reset the parsed tree to a previous state. We want to throw
                # away all equations inside the current for block.
                for x in list(self.src.keys()):
                    if x not in f.orig_keys:
                        f.tmp_src.setdefault(x, []).append(self.src[x])
                        del self.src[x]
            if len(f.values) > 0:
                f.current_value = f.values[-1]
            else:
                # TODO: Because we are using a listener, and not a visitor, we
                # will still go into the equations inside this ForEquation. To
                # avoid exceptions, we have to set the index to some valid
                # integer.
                f.current_value = 1
        else:
            pass

    def exitForEquation(self, tree):
        logger.debug('exitForEquation')

        f = self.for_loops[-1]

        if len(f.values) == 0:
            self.src[tree] = np.array([])
        elif f.current_value == f.values[-1]:
            for x in list(self.src.keys()):
                if x not in f.orig_keys:
                    f.tmp_src.setdefault(x, []).append(self.src[x])
            # A ForEquation is never a right-hand side, so we can flatten the
            # equations it contains into a 1-D array
            self.src[tree] = np.concatenate([np.concatenate(f.tmp_src[x]) for x in tree.equations])
            f = self.for_loops.pop()

    def exitIfEquation(self, tree):
        logger.debug('exitIfEquation')

        # Check if every equation block contains the same number of equations
        if len(set((len(x) for x in tree.blocks))) != 1:
            raise Exception("Every branch in an if-equation needs the same number of equations.")

        # NOTE: We currently assume that we always have an else-clause. This
        # is not strictly necessary, see the Modelica Spec on if equations.
        assert tree.conditions[-1] == True

        src = np.concatenate([self.get_mx(e) for e in tree.blocks[-1]])

        for cond_index in range(1, len(tree.conditions)):
            cond = self.get_mx(tree.conditions[-(cond_index + 1)])

            assert cond.shape == (1,), "The expression of an if or elseif-clause must be a scalar Boolean expression"

            expr1 = np.concatenate([self.get_mx(e) for e in tree.blocks[-(cond_index + 1)]])

            for ind, s_i in np.ndenumerate(src):
                src[ind] = ca.if_else(cond[0], expr1[ind], src[ind], True)

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
                    rhs = _safe_ndarray(assignment.right)
                    for ind, s_i in np.ndenumerate(assignment.left):
                        expanded_blocks.setdefault(assignment.left[ind], []).append(rhs[ind])

        assert len(set((len(x) for x in expanded_blocks.values()))) == 1

        all_assignments = []

        for lhs, values in expanded_blocks.items():
            # Set default value to else block, and then loop in reverse over all branches
            src = values[-1]
            for cond, rhs in zip(tree.conditions[-2::-1], values[-2::-1]):
                cond = self.get_mx(cond)

                assert cond.shape == (1,), "The expression of an if or elseif-clause must be a scalar Boolean expression"

                src = ca.if_else(cond[0], rhs, src, True)

            # Pack into ndarray again
            all_assignments.append(Assignment(_safe_ndarray(lhs), _safe_ndarray(src)))

        self.src[tree] = all_assignments

    def enterForStatement(self, tree):
        logger.debug('enterForStatement')

        self.for_loops.append(ForLoop(self, tree))

    def exitForStatement(self, tree):
        logger.debug('exitForStatement')

        all_assignments = []

        f = self.for_loops.pop()
        if len(f.values) > 0:
            for assignments in [self.get_mx(e) for e in tree.statements]:
                for assignment in assignments:
                    rhs = _safe_ndarray(assignment.right)
                    for ind, s_i in np.ndenumerate(assignment.left):
                        all_assignments.append(Assignment(
                            _safe_ndarray(assignment.left[ind]),
                            _safe_ndarray(rhs)))

            self.src[tree] = all_assignments
        else:
            self.src[tree] = []

    def get_integer(self, tree: Union[ast.Primary, ast.ComponentRef, ast.Expression, ast.Slice]) -> Union[int, ca.MX, np.ndarray]:
        # CasADi needs to know the dimensions of symbols at instantiation.
        # We therefore need a mechanism to evaluate expressions that define dimensions of symbols.
        if isinstance(tree, ast.Primary):
            return None if tree.value == None else int(tree.value)
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

            assert expr.shape == (1,)
            expr = expr[0]

            if not isinstance(expr, ca.MX):
                return int(expr)

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

    def get_python_type(self, tree):
        if tree.type.name == 'Boolean':
            return bool
        elif tree.type.name == 'Integer':
            return int
        else:
            return float

    def get_shape(self, tree):
        return [self.get_integer(d) for d in tree.dimensions]

    def get_symbol(self, tree):
        try:
            return self.nodes[self.current_class][tree.name]
        except KeyError:
            pass

        # Create symbol
        shape = self.get_shape(tree)

        # FIXME: Even scalars have dimensions. We cannot distinct between 1-D array of length 1, and scalars this way.
        # Real a;
        # Real a[1];
        # are therefore indistinguishable. But do we choose a symbol a, or a[1]?
        # We currently assume "a", and not "a[1]".

        if np.prod(shape) == 1:
            s = np.array([ca.MX.sym(tree.name)])
        else:
            s = np.ndarray(shape, dtype=object)

            for ind in np.ndindex(s.shape):
                ind_str = ",".join((str(x+1) for x in ind))
                name = "{}[{}]".format(tree.name, ind_str)
                s[ind] = ca.MX.sym(name)

        self.nodes[self.current_class][tree.name] = s

        return s

    def get_derivative(self, s):
        o = np.ndarray(s.shape, dtype=object)

        for ind, s_i in np.ndenumerate(s):
            if ca.MX.is_constant(s_i):
                o[ind] = 0
            elif s_i.is_symbolic():
                if s_i.name() not in self.derivative:
                    der_s_i = ca.MX.sym("der({})".format(s_i.name()))
                    self.derivative[s_i.name()] = der_s_i
                    self.nodes[self.current_class][der_s_i.name()] = _safe_ndarray(der_s_i)
                    o[ind] = der_s_i
                else:
                    o[ind] = self.derivative[s_i.name()]
            else:
                # TODO(Tjerk): Test case that ends up here?
                # Differentiate expression using CasADi
                orig_deps = ca.symvar(s_i)
                deps = ca.vertcat(*orig_deps)
                J = ca.Function('J', [deps], [ca.jacobian(s_i, deps)])
                J_sparsity = J.sparsity_out(0)
                der_deps = [self.get_derivative(dep) if J_sparsity.has_nz(0, j) else ca.DM.zeros(dep.size()) for j, dep in enumerate(orig_deps)]
                o[ind] = ca.mtimes(J(deps), ca.vertcat(*der_deps))

            return o

    def get_indexed_symbol(self, tree, s):
        # Check whether we loop over an index of this symbol
        indices = []
        for_loop = None

        for index in tree.indices:
            sl = None

            if isinstance(index, ast.ComponentRef):
                for f in reversed(self.for_loops):
                    if index.name == f.name:
                        sl = f.current_value - 1

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

        if len(indices) == 1:
            return _safe_ndarray(s[indices[0]])
        else:
            return _safe_ndarray(s[indices[0], indices[1]])

    def get_component(self, tree):
        # Check special symbols
        if tree.name == 'time':
            return np.array([self.model.time], dtype=object)
        else:
            for f in reversed(self.for_loops):
                if tree.name == f.name:
                    return f.current_value

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
            for ind, v_i in np.ndenumerate(variable):
                values[v_i] = v_i

        # Process statements in order
        for statement in tree.statements:
            src = self.get_mx(statement)
            for assignment in src:
                for ind, v_i in np.ndenumerate(assignment.left):
                    [values[assignment.left]] = ca.substitute([assignment.right], list(values.keys()), list(values.values()))

        output_expr = ca.substitute([values[output] for output in outputs], tmp, [values[t] for t in tmp])
        function = ca.Function(tree.name, inputs, output_expr)
        self.functions[function_name] = function

        return function


def generate(ast_tree: ast.Tree, model_name: str, options: Dict[str, bool] = {}) -> Model:
    """
    :param ast_tree: AST to generate from
    :param model_name: class to generate
    :param options: dictionary of generator options
    :return: casadi model
    """
    component_ref = ast.ComponentRef.from_string(model_name)
    ast_walker = TreeWalker()
    flat_tree = flatten(ast_tree, component_ref)
    component_ref_tuple = component_ref.to_tuple()
    casadi_gen = Generator(flat_tree, component_ref_tuple[-1], options)
    ast_walker.walk(casadi_gen, flat_tree)
    return casadi_gen.model
