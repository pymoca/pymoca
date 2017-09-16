from collections import namedtuple, OrderedDict
import casadi as ca
import numpy as np
import itertools
import logging
import re

from pymola import ast
from .alias_relation import AliasRelation

logger = logging.getLogger("pymola")

CASADI_COMPARISON_DEPTH = 100


class Variable:
    def __init__(self, symbol, python_type=float, aliases=[]):
        self.symbol = symbol
        self.python_type = python_type
        self.aliases = aliases

        # Default attribute values
        self.value = np.nan
        self.start = 0
        self.min = -np.inf
        self.max = np.inf
        self.nominal = 1
        self.fixed = False

    def __str__(self):
        return self.symbol.name()

    def __repr__(self):
        return '{}[{},{}]:{}'.format(self.symbol.name(), self.symbol.size1(), self.symbol.size2(), self.python_type.__name__)

    def to_dict(self):
        d = {}
        d['name'] = self.symbol.name()
        d['shape'] = (self.symbol.size1(), self.symbol.size2())
        d['python_type'] = self.python_type
        d['aliases'] = self.aliases
        return d

    @classmethod
    def from_dict(cls, d):
        variable = cls(ca.MX.sym(d['name'], *d['shape']), d['python_type'])
        variable.aliases = d['aliases']
        return variable


DelayedState = namedtuple('DelayedState', ['name', 'origin', 'delay'])


# noinspection PyUnresolvedReferences
class Model:
    def __init__(self):
        self.states = []
        self.der_states = []
        self.alg_states = []
        self.inputs = []
        self.outputs = []
        self.constants = []
        self.parameters = []
        self.equations = []
        self.initial_equations = []
        self.time = ca.MX.sym('time')
        self.delayed_states = []

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
        r += "parameters: " + str(self.parameters) + "\n"
        r += "equations: " + str(self.equations) + "\n"
        r += "initial equations: " + str(self.initial_equations) + "\n"
        return r

    def check_balanced(self):
        n_states = sum(v.symbol.size1() * v.symbol.size2() for v in itertools.chain(self.states, self.alg_states))
        n_equations = sum(e.size1() * e.size2() for e in self.dae_residual_function.mx_out())
        if n_states == n_equations:
            logger.info("System is balanced.")
        else:
            logger.warning(
                "System is not balanced.  "
                "Number of states is {}, number of equations is {}.".format(
                    n_states, n_equations))

    def _symbols(self, l):
        return [v.symbol for v in l]

    def simplify(self, options):
        if options.get('replace_parameter_expressions', False):
            logger.info("Replacing parameter expressions")

            simple_parameters, symbols, values = [], [], []
            for p in self.parameters:
                value = ca.MX(p.value)
                if value.is_constant():
                    simple_parameters.append(p)
                else:
                    symbols.append(p.symbol)
                    values.append(value)

            self.parameters = simple_parameters

            if len(values) > 0:
                # Resolve expressions that include other, non-simple parameter
                # expressions.
                converged = False
                while not converged:
                    new_values = ca.substitute(values, symbols, values)
                    converged = ca.is_equal(ca.veccat(*values), ca.veccat(*new_values), CASADI_COMPARISON_DEPTH)
                    values = new_values

                if len(self.equations) > 0:
                    self.equations = ca.substitute(self.equations, symbols, values)
                if len(self.initial_equations) > 0:
                    self.initial_equations = ca.substitute(self.initial_equations, symbols, values)

                # Replace parameter expressions in metadata
                for variable in itertools.chain(self.states, self.alg_states, self.inputs, self.parameters, self.constants):
                    for attribute in ast.Symbol.ATTRIBUTES:
                        value = getattr(variable, attribute)
                        if isinstance(value, ca.MX) and not value.is_constant():
                            [value] = ca.substitute([value], symbols, values)
                            setattr(variable, attribute, value)

        if options.get('replace_constant_expressions', False):
            logger.info("Replacing constant expressions")

            simple_constants, symbols, values = [], [], []
            for c in self.constants:
                value = ca.MX(c.value)
                if value.is_constant():
                    simple_constants.append(c)
                else:
                    symbols.append(c.symbol)
                    values.append(value)

            self.constants = simple_constants

            if len(values) > 0:
                # Resolve expressions that include other, non-simple parameter
                # expressions.
                converged = False
                while not converged:
                    new_values = ca.substitute(values, symbols, values)
                    converged = ca.is_equal(ca.veccat(*values), ca.veccat(*new_values), CASADI_COMPARISON_DEPTH)
                    values = new_values

                if len(self.equations) > 0:
                    self.equations = ca.substitute(self.equations, symbols, values)
                if len(self.initial_equations) > 0:
                    self.initial_equations = ca.substitute(self.initial_equations, symbols, values)

                # Replace constant expressions in metadata
                for variable in itertools.chain(self.states, self.alg_states, self.inputs, self.parameters, self.constants):
                    for attribute in ast.Symbol.ATTRIBUTES:
                        value = getattr(variable, attribute)
                        if isinstance(value, ca.MX) and not value.is_constant():
                            [value] = ca.substitute([value], symbols, values)
                            setattr(variable, attribute, value)

        if options.get('eliminate_constant_assignments', False):
            logger.info("Elimating constant variable assignments")

            alg_states = OrderedDict([(s.symbol.name(), s) for s in self.alg_states])

            reduced_equations = []
            for eq in self.equations:
                if eq.n_dep() == 2 and (eq.is_op(ca.OP_SUB) or eq.is_op(ca.OP_ADD)):
                    if eq.dep(0).is_symbolic() and eq.dep(0).name() in alg_states and eq.dep(1).is_constant():
                        variable = eq.dep(0)
                        value = eq.dep(1)
                    elif eq.dep(1).is_symbolic() and eq.dep(1).name() in alg_states and eq.dep(0).is_constant():
                        variable = eq.dep(1)
                        value = eq.dep(0)
                    else:
                        variable = None
                        value = None

                    if variable is not None:
                        constant = alg_states.pop(variable.name())

                        if eq.is_op(ca.OP_SUB):
                            constant.value = value
                        else:
                            constant.value = -value

                        self.constants.append(constant)

                        # Skip this equation
                        continue

                # Keep this equation
                reduced_equations.append(eq)

            # Eliminate alias variables
            self.alg_states = list(alg_states.values())
            self.equations = reduced_equations

        if options.get('replace_parameter_values', False):
            logger.info("Replacing parameter values")

            # N.B. Any parameter expression elimination must be done first.
            unspecified_parameters, symbols, values = [], [], []
            for p in self.parameters:
                if ca.MX(p.value).is_constant() and ca.MX(p.value).is_regular():
                    symbols.append(p.symbol)
                    values.append(p.value)
                else:
                    unspecified_parameters.append(p)

            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, symbols, values)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, symbols, values)
            self.parameters = unspecified_parameters

            # Replace parameter values in metadata
            for variable in itertools.chain(self.states, self.alg_states, self.inputs, self.parameters, self.constants):
                for attribute in ast.Symbol.ATTRIBUTES:
                    value = getattr(variable, attribute)
                    if isinstance(value, ca.MX) and not value.is_constant():
                        [value] = ca.substitute([value], symbols, values)
                        setattr(variable, attribute, value)

        if options.get('replace_constant_values', False):
            logger.info("Replacing constant values")

            # N.B. Any parameter expression elimination must be done first.
            symbols = self._symbols(self.constants)
            values = [v.value for v in self.constants]
            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, symbols, values)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, symbols, values)
            self.constants = []

            # Replace constant values in metadata
            for variable in itertools.chain(self.states, self.alg_states, self.inputs, self.parameters, self.constants):
                for attribute in ast.Symbol.ATTRIBUTES:
                    value = getattr(variable, attribute)
                    if isinstance(value, ca.MX) and not value.is_constant():
                        [value] = ca.substitute([value], symbols, values)
                        setattr(variable, attribute, value)

        if options.get('eliminable_variable_expression', None) is not None:
            logger.info("Elimating variables that match the regular expression {}".format(options['eliminable_variable_expression']))

            p = re.compile(options['eliminable_variable_expression'])

            alg_states = OrderedDict([(s.symbol.name(), s) for s in self.alg_states])

            variables = []
            values = []

            reduced_equations = []
            for eq in self.equations:
                if eq.n_dep() == 2 and (eq.is_op(ca.OP_SUB) or eq.is_op(ca.OP_ADD)):
                    if eq.dep(0).is_symbolic() and eq.dep(0).name() in alg_states and p.match(eq.dep(0).name()):
                        variable = eq.dep(0)
                        value = eq.dep(1)
                    elif eq.dep(1).is_symbolic() and eq.dep(1).name() in alg_states and p.match(eq.dep(1).name()):
                        variable = eq.dep(1)
                        value = eq.dep(0)
                    else:
                        variable = None
                        value = None

                    if variable is not None:
                        del alg_states[variable.name()]

                        variables.append(variable)
                        if eq.is_op(ca.OP_SUB):
                            values.append(value)
                        else:
                            values.append(-value)

                        # Skip this equation
                        continue

                # Keep this equation
                reduced_equations.append(eq)

            # Eliminate alias variables
            self.alg_states = list(alg_states.values())
            self.equations = reduced_equations

            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, variables, values)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, variables, values)

            # TODO update aliases

        if options.get('expand_vectors', False):
            logger.info("Expanding vectors")

            symbols = []
            values = []

            for l in ['states', 'der_states', 'alg_states', 'inputs', 'parameters', 'constants']:
                old_vars = getattr(self, l)
                new_vars = []
                for old_var in old_vars:
                    if old_var.symbol.numel() > 1:
                        rows = []
                        for i in range(old_var.symbol.size1()):
                            cols = []
                            for j in range(old_var.symbol.size2()):
                                if old_var.symbol.size1() > 1 and old_var.symbol.size2() > 1:
                                    component_symbol = ca.MX.sym('{}[{},{}]'.format(old_var.symbol.name(), i, j))
                                elif old_var.symbol.size1() > 1:
                                    component_symbol = ca.MX.sym('{}[{}]'.format(old_var.symbol.name(), i))
                                elif old_var.symbol.size2() > 1:
                                    component_symbol = ca.MX.sym('{}[{}]'.format(old_var.symbol.name(), j))
                                else:
                                    raise AssertionError
                                component_var = Variable(component_symbol, old_var.python_type)
                                for attribute in ast.Symbol.ATTRIBUTES:
                                    value = ca.MX(getattr(old_var, attribute))
                                    if value.size1() == old_var.symbol.size1() and value.size2() == old_var.symbol.size2():
                                        setattr(component_var, attribute, value[i, j])
                                    elif value.size1() == old_var.symbol.size1():
                                        setattr(component_var, attribute, value[i])
                                    elif value.size2() == old_var.symbol.size2():
                                        setattr(component_var, attribute, value[j])
                                    else:
                                        assert value.size1() == 1
                                        assert value.size2() == 1
                                        setattr(component_var, attribute, value)
                                cols.append(component_var)
                            rows.append(cols)
                        symbols.append(old_var.symbol)
                        values.append(ca.vertcat(*[ca.horzcat(*[v.symbol for v in row]) for row in rows]))
                        new_vars.extend(itertools.chain.from_iterable(rows))
                    else:
                        new_vars.append(old_var)
                setattr(self, l, new_vars)

            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, symbols, values)
                self.equations = list(itertools.chain.from_iterable(ca.vertsplit(ca.vec(eq)) for eq in self.equations))
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, symbols, values)
                self.initial_equations = list(itertools.chain.from_iterable(ca.vertsplit(ca.vec(eq)) for eq in self.initial_equations))

            # Replace values in metadata
            for variable in itertools.chain(self.states, self.alg_states, self.inputs, self.parameters, self.constants):
                for attribute in ast.Symbol.ATTRIBUTES:
                    value = getattr(variable, attribute)
                    if isinstance(value, ca.MX) and not value.is_constant():
                        [value] = ca.substitute([value], symbols, values)
                        setattr(variable, attribute, value)

        if options.get('detect_aliases', False):
            logger.info("Detecting aliases")

            states = OrderedDict([(s.symbol.name(), s) for s in self.states])
            der_states = OrderedDict([(s.symbol.name(), s) for s in self.der_states])
            alg_states = OrderedDict([(s.symbol.name(), s) for s in self.alg_states])
            inputs = OrderedDict([(s.symbol.name(), s) for s in self.inputs])

            all_states = OrderedDict()
            all_states.update(states)
            all_states.update(der_states)
            all_states.update(alg_states)
            all_states.update(inputs)

            alias_rel = AliasRelation()

            # For now, we only eliminate algebraic states.
            do_not_eliminate = set(list(der_states) + list(states) + list(inputs))

            reduced_equations = []
            for eq in self.equations:
                if eq.n_dep() == 2 and (eq.is_op(ca.OP_SUB) or eq.is_op(ca.OP_ADD)):
                    if eq.dep(0).is_symbolic() and eq.dep(1).is_symbolic():
                        if eq.dep(0).name() in alg_states:
                            alg_state = eq.dep(0)
                            other_state = eq.dep(1)
                        elif eq.dep(1).name() in alg_states:
                            alg_state = eq.dep(1)
                            other_state = eq.dep(0)
                        else:
                            alg_state = None
                            other_state = None

                        # If both states are algebraic, we need to decide which to eliminate
                        if eq.dep(0).name() in alg_states and eq.dep(1).name() in alg_states:
                            # Most of the time it does not matter which one we eliminate.
                            # The exception is if alg_state has already been aliased to a
                            # variable in do_not_eliminate. If this is the case, setting the
                            # states in the default order will cause the new canonical variable
                            # to be other_state, unseating (and eliminating) the current
                            # canonical variable (which is in do_not_eliminate).
                            if alias_rel.canonical_signed(alg_state.name())[0] in do_not_eliminate:
                                # swap the states
                                other_state, alg_state = alg_state, other_state

                        if alg_state is not None:
                            # Check to see if we are linking two entries in do_not_eliminate
                            if alias_rel.canonical_signed(alg_state.name())[0] in do_not_eliminate and \
                               alias_rel.canonical_signed(other_state.name())[0] in do_not_eliminate:
                                # Don't do anything for now, we only eliminate alg_states
                                pass

                            else:
                                # Eliminate alg_state by aliasing it to other_state
                                if eq.is_op(ca.OP_SUB):
                                    alias_rel.add(other_state.name(), alg_state.name())
                                else:
                                    alias_rel.add(other_state.name(), '-' + alg_state.name())

                                # To keep equations balanced, drop this equation
                                continue

                # Keep this equation
                reduced_equations.append(eq)

            # Eliminate alias variables
            variables, values = [], []
            for canonical, aliases in alias_rel:
                canonical_state = all_states[canonical]

                python_type = canonical_state.python_type
                start = canonical_state.start
                m, M = canonical_state.min, canonical_state.max
                nominal = canonical_state.nominal
                fixed = canonical_state.fixed

                for alias in aliases:
                    if alias[0] == '-':
                        sign = -1
                        alias = alias[1:]
                    else:
                        sign = 1

                    alias_state = all_states[alias]

                    variables.append(alias_state.symbol)
                    values.append(sign * canonical_state.symbol)

                    # If any of the aliases has a nonstandard type, apply it to
                    # the canonical state as well
                    if alias_state.python_type != float:
                        python_type = alias_state.python_type

                    # If any of the aliases has a nondefault start value, apply it to
                    # the canonical state as well
                    alias_state_start = ca.MX(alias_state.start)
                    if alias_state_start.is_regular() and not alias_state_start.is_zero():
                        start = sign * alias_state.start

                    # The intersection of all bound ranges applies
                    m = ca.fmax(m, alias_state.min if sign == 1 else -alias_state.max)
                    M = ca.fmin(M, alias_state.max if sign == 1 else -alias_state.min)

                    # Take the largest nominal of all aliases
                    nominal = ca.fmax(nominal, alias_state.nominal)

                    # If any of the aliases is fixed, the canonical state is as well
                    fixed = ca.fmax(fixed, alias_state.fixed)

                    del all_states[alias]

                canonical_state.aliases = aliases
                canonical_state.python_type = python_type
                canonical_state.start = start
                canonical_state.min = m
                canonical_state.max = M
                canonical_state.nominal = nominal
                canonical_state.fixed = fixed

            self.states = [v for k, v in all_states.items() if k in states]
            self.der_states = [v for k, v in all_states.items() if k in der_states]
            self.alg_states = [v for k, v in all_states.items() if k in alg_states]
            self.inputs = [v for k, v in all_states.items() if k in inputs]
            self.equations = reduced_equations

            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, variables, values)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, variables, values)

        if options.get('reduce_affine_expression', False):
            logger.info("Collapsing model into an affine expression")

            for equation_list in ['equations', 'initial_equations']:
                equations = getattr(self, equation_list)
                if len(equations) > 0:
                    states = ca.veccat(*self._symbols(itertools.chain(self.states, self.der_states, self.alg_states, self.inputs)))
                    constants = ca.veccat(*self._symbols(self.constants))
                    parameters = ca.veccat(*self._symbols(self.parameters))

                    equations = ca.veccat(*equations)

                    Af = ca.Function('Af', [states, constants, parameters], [ca.jacobian(equations, states)])
                    bf = ca.Function('bf', [states, constants, parameters], [equations])

                    # Work around CasADi issue #172
                    if len(self.constants) == 0:
                        constants = 0
                    else:
                        logger.warning('Not all constants have been eliminated.  As a result, the affine DAE expression will use a symbolic matrix, as opposed to a numerical sparse matrix.')
                    if len(self.parameters) == 0:
                        parameters = 0
                    else:
                        logger.warning('Not all parameters have been eliminated.  As a result, the affine DAE expression will use a symbolic matrix, as opposed to a numerical sparse matrix.')

                    A = Af(0, constants, parameters)
                    b = bf(0, constants, parameters)

                    equations = [ca.reshape(ca.mtimes(A, states), equations.shape) + b]
                    setattr(self, equation_list, equations)

        if options.get('expand_mx', False):
            logger.info("Expanding MX graph")
            
            if len(self.equations) > 0:
                self.equations = ca.matrix_expand(self.equations)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.matrix_expand(self.initial_equations)

        logger.info("Finished model simplification")

    @property
    def dae_residual_function(self):
        return ca.Function('dae_residual', [self.time, ca.veccat(*self._symbols(self.states)), ca.veccat(*self._symbols(self.der_states)),
                                            ca.veccat(*self._symbols(self.alg_states)), ca.veccat(*self._symbols(self.inputs)), ca.veccat(*self._symbols(self.constants)),
                                            ca.veccat(*self._symbols(self.parameters))], [ca.veccat(*self.equations)] if len(self.equations) > 0 else [])

    # noinspection PyUnusedLocal
    @property
    def initial_residual_function(self):
        return ca.Function('initial_residual', [self.time, ca.veccat(*self._symbols(self.states)), ca.veccat(*self._symbols(self.der_states)),
                                            ca.veccat(*self._symbols(self.alg_states)), ca.veccat(*self._symbols(self.inputs)), ca.veccat(*self._symbols(self.constants)),
                                            ca.veccat(*self._symbols(self.parameters))], [ca.veccat(*self.initial_equations)] if len(self.initial_equations) > 0 else [])

    # noinspection PyPep8Naming
    @property
    def variable_metadata_function(self):
        out = []
        zero, one = ca.MX(0), ca.MX(1) # Recycle these common nodes as much as possible.
        for variable_list in [self.states, self.alg_states, self.inputs, self.parameters, self.constants]:
            attribute_lists = [[] for i in range(len(ast.Symbol.ATTRIBUTES))]
            for variable in variable_list:
                for attribute_list_index, attribute in enumerate(ast.Symbol.ATTRIBUTES):
                    value = ca.MX(getattr(variable, attribute))
                    if value.is_zero():
                        value = zero
                    elif (value - 1).is_zero():
                        value = one
                    value = value if value.numel() != 1 else ca.repmat(value, *variable.symbol.size())
                    attribute_lists[attribute_list_index].append(value)
            out.append(ca.horzcat(*[ca.veccat(*attribute_list) for attribute_list in attribute_lists]))
        return ca.Function('variable_metadata', [ca.veccat(*self._symbols(self.parameters))], out) 
