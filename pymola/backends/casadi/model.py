from collections import namedtuple, OrderedDict
import casadi as ca
import numpy as np
import itertools
import logging
import re

from pymola import ast
from .alias_relation import AliasRelation

logger = logging.getLogger("pymola")


class Variable:
    def __init__(self, symbol, python_type=float, aliases=[]):
        self.symbol = symbol
        self.python_type = python_type
        self.aliases = aliases

        # Default attribute values
        self.value = np.nan
        self.start = np.nan
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
                if ca.MX(p.value).is_constant():
                    simple_parameters.append(p)
                else:
                    symbols.append(p.symbol)
                    values.append(p.value)   

            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, symbols, values)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, symbols, values)
            self.parameters = simple_parameters

            # Replace parameter expressions in metadata
            for variable in itertools.chain(self.states, self.alg_states, self.inputs, self.parameters, self.constants):
                for attribute in ast.Symbol.ATTRIBUTES:
                    value = getattr(variable, attribute)
                    if isinstance(value, ca.MX):
                        [value] = ca.substitute([value], symbols, values)
                        setattr(variable, attribute, value)

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
                    if isinstance(value, ca.MX):
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
                    if isinstance(value, ca.MX):
                        [value] = ca.substitute([value], symbols, values)
                        setattr(variable, attribute, value)

        if options.get('eliminable_variable_expression', None) is not None:
            logger.info("Elimating variables that match the regular expression {}".format(options['eliminable_variable_expression']))

            p = re.compile(options['eliminable_variable_expression'])

            alg_states = OrderedDict({s.symbol.name() : s for s in self.alg_states})

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

        if options.get('detect_aliases', False):
            logger.info("Detecting aliases")

            states = OrderedDict({s.symbol.name() : s for s in self.states})
            der_states = OrderedDict({s.symbol.name() : s for s in self.der_states})
            alg_states = OrderedDict({s.symbol.name() : s for s in self.alg_states})
            inputs = OrderedDict({s.symbol.name() : s for s in self.inputs})
            outputs = OrderedDict({s.symbol.name() : s for s in self.outputs})

            all_states = {}
            all_states.update(states)
            all_states.update(der_states)
            all_states.update(alg_states)
            all_states.update(inputs)

            alias_rel = AliasRelation()

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

                        if alg_state is not None:
                            # Add alias
                            if eq.is_op(ca.OP_SUB):
                                alias_rel.add(other_state.name(), alg_state.name())
                            else:
                                alias_rel.add(other_state.name(), '-' + alg_state.name())

                            # Skip this equation
                            continue

                # Keep this equation
                reduced_equations.append(eq)

            # Eliminate alias variables
            variables, values = [], []
            for canonical, aliases in alias_rel:
                canonical_state = all_states[canonical]
                setattr(canonical_state, 'aliases', aliases)
                for alias in aliases:
                    if alias[0] == '-':
                        sign = -1
                        alias = alias[1:]
                    else:
                        sign = 1
                    variables.append(all_states[alias].symbol)
                    values.append(sign * canonical_state.symbol)

                    del all_states[alias]
                    if alias in outputs:
                        outputs[alias].symbol = sign * canonical_state.symbol

            self.states = [v for k, v in all_states.items() if k in states]
            self.der_states = [v for k, v in all_states.items() if k in der_states]
            self.alg_states = [v for k, v in all_states.items() if k in alg_states]
            self.inputs = [v for k, v in all_states.items() if k in inputs]
            self.outputs = list(outputs.values())
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

                    Af = ca.Function('Af', [constants, parameters], [ca.jacobian(equations, states)])
                    A = Af(constants, parameters)

                    bf = ca.Function('bf', [states, constants, parameters], [equations])
                    b = bf(0, constants, parameters)

                    equations = [ca.reshape(ca.mtimes(A, states), equations.shape) + b]
                    setattr(self, equation_list, equations)

        if options.get('expand', False):
            logger.info("Expanding MX graph")
            
            if len(self.equations) > 0:
                self.equations = ca.matrix_expand(self.equations)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.matrix_expand(self.initial_equations)

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
        for variable_list in [self.states, self.alg_states, self.inputs, self.parameters, self.constants]:
            attribute_lists = [[] for i in range(len(ast.Symbol.ATTRIBUTES))]
            for variable in variable_list:
                for attribute_list_index, attribute in enumerate(ast.Symbol.ATTRIBUTES):
                    value = ca.MX(getattr(variable, attribute))
                    value = value if value.numel() != 1 else ca.repmat(value, *variable.symbol.size())
                    attribute_lists[attribute_list_index].append(value)
            out.append(ca.horzcat(*[ca.veccat(*attribute_list) for attribute_list in attribute_lists]))
        return ca.Function('variable_metadata', [ca.veccat(*self._symbols(self.parameters))], out) 
