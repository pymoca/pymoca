from collections import namedtuple, OrderedDict
import casadi as ca
import numpy as np
import itertools
import logging
import re

from pymoca import ast
from .alias_relation import AliasRelation
from .mtensor import _MTensor

logger = logging.getLogger("pymoca")

CASADI_COMPARISON_DEPTH = 100
CASADI_ATTRIBUTES = [attr for attr in ast.Symbol.ATTRIBUTES if not attr == 'unit']


class Variable:
    def __init__(self, symbol, python_type=float, aliases=None):
        if aliases is None:
            aliases = []
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


DelayArgument = namedtuple('DelayArgument', ['expr', 'duration'])


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
        self.delay_states = []
        self.delay_arguments = []
        self._expand_mx_func = lambda x: x

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

    def _post_checks(self):
        # We do not support delayMax yet, so delay durations can only depend
        # on constants, parameters and fixed inputs.
        if self.delay_states:
            delay_durations = ca.veccat(*(x.duration for x in self.delay_arguments))
            disallowed_duration_symbols = ca.vertcat(self.time,
                ca.veccat(*self._symbols(self.states)),
                ca.veccat(*self._symbols(self.der_states)),
                ca.veccat(*self._symbols(self.alg_states)),
                ca.veccat(*(x.symbol for x in self.inputs if not x.fixed)))

            if ca.depends_on(delay_durations, disallowed_duration_symbols):
                raise ValueError(
                    "Delay durations can only depend on parameters, constants and fixed inputs.")

    @staticmethod
    def _symbols(l):
        return [v.symbol for v in l]

    def _substitute_delay_arguments(self, delay_arguments, symbols, values):
        exprs = ca.substitute([ca.MX(argument.expr) for argument in delay_arguments], symbols, values)
        durations = ca.substitute([ca.MX(argument.duration) for argument in delay_arguments], symbols, values)
        return [DelayArgument(expr, duration) for expr, duration in zip(exprs, durations)]

    def simplify(self, options):
        if options.get('replace_parameter_expressions', False):
            logger.info("Replacing parameter expressions")

            simple_parameters, symbols, values = [], [], []
            for p in self.parameters:
                if isinstance(p.value, list):
                    p.value = np.array(p.value)

                    if not np.issubdtype(p.value.dtype, np.number):
                        raise NotImplementedError(
                            "Only parameters arrays with numeric values can be simplified")

                    simple_parameters.append(p)
                else:
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
                if len(self.delay_arguments) > 0:
                    self.delay_arguments = self._substitute_delay_arguments(self.delay_arguments, symbols, values)

                # Replace parameter expressions in metadata
                for variable in itertools.chain(self.states, self.alg_states, self.inputs, self.parameters, self.constants):
                    for attribute in CASADI_ATTRIBUTES:
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
                if len(self.delay_arguments) > 0:
                    self.delay_arguments = self._substitute_delay_arguments(self.delay_arguments, symbols, values)

                # Replace constant expressions in metadata
                for variable in itertools.chain(self.states, self.alg_states, self.inputs, self.parameters, self.constants):
                    for attribute in CASADI_ATTRIBUTES:
                        value = getattr(variable, attribute)
                        if isinstance(value, ca.MX) and not value.is_constant():
                            [value] = ca.substitute([value], symbols, values)
                            setattr(variable, attribute, value)

        if options.get('eliminate_constant_assignments', False):
            logger.info("Elimating constant variable assignments")

            alg_states = OrderedDict([(s.symbol.name(), s) for s in self.alg_states])

            reduced_equations = []
            for eq in self.equations:
                if eq.is_symbolic() and eq.name() in alg_states:
                    constant = alg_states.pop(eq.name())
                    constant.value = 0.0

                    self.constants.append(constant)

                    # Skip this equation
                    continue

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
                for attribute in CASADI_ATTRIBUTES:
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
            if len(self.delay_arguments) > 0:
                self.delay_arguments = self._substitute_delay_arguments(self.delay_arguments, symbols, values)
            self.constants = []

            # Replace constant values in metadata
            for variable in itertools.chain(self.states, self.alg_states, self.inputs, self.parameters, self.constants):
                for attribute in CASADI_ATTRIBUTES:
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
                if eq.is_symbolic() and eq.name() in alg_states and p.match(eq.name()):
                    variables.append(eq)
                    values.append(0.0)
                    del alg_states[eq.name()]
                    # Skip this equation
                    continue

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
            if len(self.delay_arguments) > 0:
                self.delay_arguments = self._substitute_delay_arguments(self.delay_arguments, variables, values)

        if options.get('expand_vectors', False):
            logger.info("Expanding vectors")

            symbols = []
            values = []

            for l in ['states', 'der_states', 'alg_states', 'inputs', 'parameters', 'constants']:
                old_vars = getattr(self, l)
                new_vars = []
                for old_var in old_vars:
                    # For delayed states we do not have any reliable shape
                    # information available due to it being an arbitrary
                    # expression, so we just always expand.
                    if (set(old_var.symbol._modelica_shape) != {(None,)}
                            or old_var.symbol.name() in self.delay_states):
                        expanded_symbols = []

                        # Prepare a component_name_format in which the different possible
                        # combinations of indices can later be inserted.
                        # For example: a.b[{}].c[{},{}]
                        if old_var.symbol.name() in self.delay_states:
                            # For delay states we use the _modelica_shape as is
                            iterator_shape = old_var.symbol._modelica_shape
                            component_name_format = \
                                old_var.symbol.name() + '[' + \
                                ','.join('{}' for _ in range(len(iterator_shape))) + ']'
                        else:
                            # For (nested) array symbols the _modelica_shape contains a tuple of
                            # tuples containing the shape for each of the nested symbols.
                            # For example, symbol name a.b.c and shape ((None,),(3,),(1,3)) means
                            # a was a scalar, b a 1d array of size [3] and c a 2d array of size
                            # [1,3].
                            modelica_shape = old_var.symbol._modelica_shape
                            symbol_names = old_var.symbol.name().split('.')
                            assert len(symbol_names) == len(modelica_shape)
                            component_name_format = ''
                            for i, symbol_name in enumerate(symbol_names):
                                if i != 0:
                                    component_name_format += '.'
                                component_name_format += symbol_name
                                if modelica_shape[i] != (None,):
                                    component_name_format += \
                                        '[' + \
                                        ','.join('{}' for _ in range(len(modelica_shape[i]))) + \
                                        ']'

                            iterator_shape = tuple([d for var_shape in modelica_shape
                                                    for d in var_shape if d is not None])

                        # Generate symbols for each possible combination of indices
                        for ind in np.ndindex(iterator_shape):
                            component_symbol = ca.MX.sym(component_name_format
                                                         .format(*tuple(i + 1 for i in ind)))
                            component_var = Variable(component_symbol, old_var.python_type)
                            for attribute in CASADI_ATTRIBUTES:
                                # Can't convert 3D arrays to MX, so we convert to nparray instead
                                value = getattr(old_var, attribute)
                                if not isinstance(value, ca.MX) and not np.isscalar(value):
                                    value = np.array(value)
                                else:
                                    value = ca.MX(getattr(old_var, attribute))

                                if np.prod(value.shape) == 1:
                                    setattr(component_var, attribute, value)
                                else:
                                    setattr(component_var, attribute, value[ind])
                            expanded_symbols.append(component_var)

                        s = old_var.symbol._mx if isinstance(old_var.symbol, _MTensor) else old_var.symbol
                        symbols.append(s)
                        values.append(ca.reshape(ca.vertcat(*[x.symbol for x in expanded_symbols]), *tuple(reversed(s.shape))).T)
                        new_vars.extend(expanded_symbols)

                        # Replace variable in delay expressions and durations if needed
                        try:
                            assert len(self.delay_states) == len(self.delay_arguments)

                            i = self.delay_states.index(old_var.symbol.name())
                        except ValueError:
                            pass
                        else:
                            delay_state = self.delay_states.pop(i)
                            delay_argument = self.delay_arguments.pop(i)

                            for ind in np.ndindex(old_var.symbol._modelica_shape):
                                new_name = '{}[{}]'.format(delay_state, ",".join(str(i+1) for i in ind))

                                self.delay_states.append(new_name)
                                self.delay_arguments.append(
                                    DelayArgument(delay_argument.expr[ind], delay_argument.duration))

                        # Replace variable in list of outputs if needed
                        try:
                            i = self.outputs.index(old_var.symbol.name())
                        except ValueError:
                            pass
                        else:
                            self.outputs.pop(i)
                            for new_s in reversed(expanded_symbols):
                                self.outputs.insert(i, new_s.symbol.name())
                    else:
                        new_vars.append(old_var)

                setattr(self, l, new_vars)

            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, symbols, values)
                self.equations = list(itertools.chain.from_iterable(ca.vertsplit(ca.vec(eq)) for eq in self.equations))
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, symbols, values)
                self.initial_equations = list(itertools.chain.from_iterable(ca.vertsplit(ca.vec(eq)) for eq in self.initial_equations))
            if len(self.delay_arguments) > 0:
                self.delay_arguments = self._substitute_delay_arguments(self.delay_arguments, symbols, values)

            # Make sure that the naming in the main loop and the delay argument loop match
            input_names = [v.symbol.name() for v in self.inputs]
            assert set(self.delay_states).issubset(input_names)

            # Replace values in metadata
            for variable in itertools.chain(self.states, self.alg_states, self.inputs, self.parameters, self.constants):
                for attribute in CASADI_ATTRIBUTES:
                    value = getattr(variable, attribute)
                    if isinstance(value, ca.MX) and not value.is_constant():
                        [value] = ca.substitute([value], symbols, values)
                        setattr(variable, attribute, value)

        if options.get('factor_and_simplify_equations', False):
            # Operations that preserve the equivalence of an equation
            # TODO: There may be more, but this is the most frequent set
            unary_ops = ca.OP_NEG, ca.OP_FABS, ca.OP_SQRT
            binary_ops = ca.OP_MUL, ca.OP_DIV

            # Recursive factor and simplify function
            def factor_and_simplify(eq):
                # These are ops that can simply be dropped
                if eq.n_dep() == 1 and eq.op() in unary_ops:
                    return factor_and_simplify(eq.dep())

                # These are binary ops and can get a little tricky
                # For now, we just drop constant divisors or multipliers
                elif eq.n_dep() == 2 and eq.op() in binary_ops:
                    if eq.dep(1).is_constant():
                        return factor_and_simplify(eq.dep(0))
                    elif eq.dep(0).is_constant() and eq.op() == ca.OP_MUL:
                        return factor_and_simplify(eq.dep(1))

                # If no hits, return unmodified
                return eq

            # Do the simplifications
            simplified_equations = [factor_and_simplify(eq) for eq in self.equations]

            # Debugging output
            if logger.getEffectiveLevel() == logging.DEBUG:
                changed_equations = [(o,s) for o, s in zip(self.equations, simplified_equations) if o is not s]
                for orig, simp in changed_equations:
                    logger.debug('Equation {} simplified to {}'.format(orig, simp))

            # Store changes
            self.equations = simplified_equations

        if options.get('detect_aliases', False):
            logger.info("Detecting aliases")

            states = OrderedDict([(s.symbol.name(), s) for s in self.states])
            der_states = OrderedDict([(s.symbol.name(), s) for s in self.der_states])
            alg_states = OrderedDict([(s.symbol.name(), s) for s in self.alg_states])
            inputs = OrderedDict([(s.symbol.name(), s) for s in self.inputs])
            parameters = OrderedDict([(s.symbol.name(), s) for s in self.parameters])

            all_states = OrderedDict()
            all_states.update(states)
            all_states.update(der_states)
            all_states.update(alg_states)
            all_states.update(inputs)
            all_states.update(parameters)

            alias_rel = AliasRelation()

            # For now, we only eliminate algebraic states.
            do_not_eliminate = set(list(der_states) + list(states) + list(inputs) + list(parameters))

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
            self.parameters = [v for k, v in all_states.items() if k in parameters]
            self.equations = reduced_equations

            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, variables, values)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, variables, values)
            if len(self.delay_arguments) > 0:
                self.delay_arguments = self._substitute_delay_arguments(self.delay_arguments, variables, values)

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
                    if len(self.constants) == 0 or not ca.depends_on(equations, constants):
                        constants = 0
                    else:
                        logger.warning('Not all constants have been eliminated.  As a result, the affine DAE expression will use a symbolic matrix, as opposed to a numerical sparse matrix.')
                    if len(self.parameters) == 0 or not ca.depends_on(equations, parameters):
                        parameters = 0
                    else:
                        logger.warning('Not all parameters have been eliminated.  As a result, the affine DAE expression will use a symbolic matrix, as opposed to a numerical sparse matrix.')

                    A = Af(0, constants, parameters)
                    b = bf(0, constants, parameters)

                    # Replace veccat'ed states with brand new state vectors so as to avoid the value copy operations induced by veccat.
                    self._states_vector = ca.MX.sym('states_vector', sum([s.numel() for s in self._symbols(self.states)]))
                    self._der_states_vector = ca.MX.sym('der_states_vector', sum([s.numel() for s in self._symbols(self.der_states)]))
                    self._alg_states_vector = ca.MX.sym('alg_states_vector', sum([s.numel() for s in self._symbols(self.alg_states)]))
                    self._inputs_vector = ca.MX.sym('inputs_vector', sum([s.numel() for s in self._symbols(self.inputs)]))

                    states_vector = ca.vertcat(self._states_vector, self._der_states_vector, self._alg_states_vector, self._inputs_vector)
                    equations = [ca.reshape(ca.mtimes(A, states_vector), equations.shape) + b]
                    setattr(self, equation_list, equations)

        if options.get('expand_mx', False):
            logger.info("Expanded MX functions will be returned")
            self._expand_mx_func = lambda x: x.expand()

        logger.info("Finished model simplification")

    @property
    def dae_residual_function(self):
        if hasattr(self, '_states_vector'):
            return self._expand_mx_func(ca.Function('dae_residual', [self.time, self._states_vector, self._der_states_vector,
                                                self._alg_states_vector, self._inputs_vector, ca.veccat(*self._symbols(self.constants)),
                                                ca.veccat(*self._symbols(self.parameters))], [ca.veccat(*self.equations)] if len(self.equations) > 0 else []))
        else:
            return self._expand_mx_func(ca.Function('dae_residual', [self.time, ca.veccat(*self._symbols(self.states)), ca.veccat(*self._symbols(self.der_states)),
                                                ca.veccat(*self._symbols(self.alg_states)), ca.veccat(*self._symbols(self.inputs)), ca.veccat(*self._symbols(self.constants)),
                                                ca.veccat(*self._symbols(self.parameters))], [ca.veccat(*self.equations)] if len(self.equations) > 0 else []))

    # noinspection PyUnusedLocal
    @property
    def initial_residual_function(self):
        if hasattr(self, '_states_vector'):
            return self._expand_mx_func(ca.Function('initial_residual', [self.time, self._states_vector, self._der_states_vector,
                                                self._alg_states_vector, self._inputs_vector, ca.veccat(*self._symbols(self.constants)),
                                                ca.veccat(*self._symbols(self.parameters))], [ca.veccat(*self.initial_equations)] if len(self.initial_equations) > 0 else []))
        else:
            return self._expand_mx_func(ca.Function('initial_residual', [self.time, ca.veccat(*self._symbols(self.states)), ca.veccat(*self._symbols(self.der_states)),
                                                ca.veccat(*self._symbols(self.alg_states)), ca.veccat(*self._symbols(self.inputs)), ca.veccat(*self._symbols(self.constants)),
                                                ca.veccat(*self._symbols(self.parameters))], [ca.veccat(*self.initial_equations)] if len(self.initial_equations) > 0 else []))

    # noinspection PyPep8Naming
    @property
    def variable_metadata_function(self):
        in_var = ca.veccat(*self._symbols(self.parameters))
        out = []
        is_affine = True
        zero, one = ca.MX(0), ca.MX(1) # Recycle these common nodes as much as possible.
        for variable_list in [self.states, self.alg_states, self.inputs, self.parameters, self.constants]:
            attribute_lists = [[] for i in range(len(CASADI_ATTRIBUTES))]
            for variable in variable_list:
                for attribute_list_index, attribute in enumerate(CASADI_ATTRIBUTES):
                    value = ca.MX(getattr(variable, attribute))
                    if value.is_zero():
                        value = zero
                    elif value.is_one():
                        value = one
                    value = value if value.numel() != 1 else ca.repmat(value, *variable.symbol.size())
                    attribute_lists[attribute_list_index].append(value)
            expr = ca.horzcat(*[ca.veccat(*attribute_list) for attribute_list in attribute_lists])
            if len(self.parameters) > 0 and isinstance(expr, ca.MX):
                f = ca.Function('f', [in_var], [expr])
                # NOTE: This is not a complete list of operations that can be
                # handled in an affine expression. That said, it should
                # capture the most common ways variable attributes are
                # expressed as a function of parameters.
                allowed_ops = {ca.OP_INPUT, ca.OP_OUTPUT, ca.OP_CONST,
                               ca.OP_SUB, ca.OP_ADD, ca.OP_SUB, ca.OP_MUL, ca.OP_DIV, ca.OP_NEG}
                f_ops = {f.instruction_id(k) for k in range(f.n_instructions())}
                contains_unallowed_ops = not f_ops.issubset(allowed_ops)
                zero_hessian = ca.jacobian(ca.jacobian(expr, in_var), in_var).is_zero()
                if contains_unallowed_ops or not zero_hessian:
                    is_affine = False
            out.append(expr)
        if len(self.parameters) > 0 and is_affine:
            # Rebuild variable metadata as a single affine expression, if all
            # subexpressions are affine.
            in_var_ = ca.MX.sym('in_var', in_var.shape)
            out_ = []
            for o in out:
                Af = ca.Function('Af', [in_var], [ca.jacobian(o, in_var)])
                bf = ca.Function('bf', [in_var], [o])

                A = Af(0)
                A = ca.sparsify(A)

                b = bf(0)
                b = ca.sparsify(b)

                o_ = ca.reshape(ca.mtimes(A, in_var_), o.shape) + b
                out_.append(o_)
            out = out_
            in_var = in_var_

        return self._expand_mx_func(ca.Function('variable_metadata', [in_var], out))

    # noinspection PyPep8Naming
    @property
    def delay_arguments_function(self):
        # We cannot assume that we can ca.horzcat/vertcat all delay arguments
        # and expressions due to shape differences, so instead we flatten our
        # delay expressions and durations into list, i.e. [delay_expr_1,
        # delay_duration_1, delay_expr_2, delay_duration_2, ...].

        out_arguments = list(itertools.chain.from_iterable(self.delay_arguments))

        if hasattr(self, '_states_vector'):
            return self._expand_mx_func(ca.Function(
                'delay_arguments',
                [self.time,
                 self._states_vector,
                 self._der_states_vector,
                 self._alg_states_vector,
                 self._inputs_vector,
                 ca.veccat(*self._symbols(self.constants)),
                 ca.veccat(*self._symbols(self.parameters))],
                out_arguments))
        else:
            return self._expand_mx_func(ca.Function(
                'delay_arguments',
                [self.time,
                 ca.veccat(*self._symbols(self.states)),
                 ca.veccat(*self._symbols(self.der_states)),
                 ca.veccat(*self._symbols(self.alg_states)),
                 ca.veccat(*self._symbols(self.inputs)),
                 ca.veccat(*self._symbols(self.constants)),
                 ca.veccat(*self._symbols(self.parameters))],
                out_arguments))
