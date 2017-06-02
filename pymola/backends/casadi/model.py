from collections import namedtuple, OrderedDict
import casadi as ca
import itertools
import logging
import re

from .alias_relation import AliasRelation

logger = logging.getLogger("pymola")


class Variable:
    def __init__(self, symbol, python_type=float, aliases=[]):
        self.symbol = symbol
        self.python_type = python_type
        self.aliases = aliases

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

    @staticmethod
    def from_dict(cls, d):
        variable = cls(MX.sym(d['name'], *d['shape']), d['python_type'])
        variable.aliases = d['aliases']
        return variable


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
        n_inputs = sum(v.symbol.size1() * v.symbol.size2() for v in self.inputs)
        n_equations = sum(e.size1() * e.size2() for e in self.dae_residual_function.mx_out())
        if n_states - n_inputs == n_equations:
            logger.info("System is balanced.")
        else:
            logger.warning(
                "System is not balanced.  "
                "Number of states minus inputs is {}, number of equations is {}.".format(
                    n_states - n_inputs, n_equations))

    def _symbols(self, l):
        return [v.symbol for v in l]

    def simplify(self, options):
        if options.get('replace_constants', False):
            logger.info("Replacing constants")

            symbols = self._symbols(self.constants)
            values = [v.value for v in self.constants]
            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, symbols, values)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, symbols, values)
            self.constants = []

        if options.get('replace_parameter_expressions', False):
            logger.info("Replacing parameter expressions")

            simple_parameters, symbols, values = [], [], []
            for p in self.parameters:
                is_composite = isinstance(p.value, ca.MX) and not p.value.is_constant()
                if is_composite:
                    symbols.append(p.symbol)
                    values.append(p.value)
                else:
                    simple_parameters.append(p)

            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, symbols, values)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, symbols, values)
            self.parameters = simple_parameters

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
                        del alg_states[variable]

                        variables.append(variable.symbol)
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
            alg_states = OrderedDict({s.symbol.name() : s for s in self.alg_states})
            inputs = OrderedDict({s.symbol.name() : s for s in self.inputs})
            outputs = OrderedDict({s.symbol.name() : s for s in self.outputs})

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
                try:
                    canonical_state = alg_states[canonical]
                except KeyError:
                    canonical_state = states[canonical]
                setattr(canonical_state, 'aliases', aliases)
                for alias in aliases:
                    if alias[0] == '-':
                        sign = -1
                        alias = alias[1:]
                    else:
                        sign = 1
                    variables.append(alg_states[alias].symbol)
                    values.append(sign * canonical_state.symbol)

                    del alg_states[alias]
                    if alias in inputs:
                        inputs[alias].symbol = sign * canonical_state.symbol
                    if alias in outputs:
                        outputs[alias].symbol = sign * canonical_state.symbol

            self.alg_states = list(alg_states.values())
            self.inputs = list(inputs.values())
            self.outputs = list(outputs.values())
            self.equations = reduced_equations

            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, variables, values)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, variables, values)

        if options.get('expand', False):
            logger.info("Expanding MX graph")
            
            if len(self.equations) > 0:
                self.equations = ca.matrix_expand(self.equations)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.matrix_expand(self.initial_equations)

    @property
    def dae_residual_function(self):
        return ca.Function('dae_residual', [self.time, ca.veccat(*self._symbols(self.states)), ca.veccat(*self._symbols(self.der_states)),
                                            ca.veccat(*self._symbols(self.alg_states)), ca.veccat(*self._symbols(self.constants)),
                                            ca.veccat(*self._symbols(self.parameters))], [ca.veccat(*self.equations)])

    # noinspection PyUnusedLocal
    @property
    def initial_residual_function(self):
        return ca.Function('initial_residual', [self.time, ca.veccat(*self._symbols(self.states)), ca.veccat(*self._symbols(self.der_states)),
                                            ca.veccat(*self._symbols(self.alg_states)), ca.veccat(*self._symbols(self.constants)),
                                            ca.veccat(*self._symbols(self.parameters))], [ca.veccat(*self.initial_equations)] if len(self.initial_equations) > 0 else [])

    VARIABLE_METADATA = ['value', 'start', 'min', 'max', 'nominal', 'fixed']

    # noinspection PyPep8Naming
    @property
    def variable_metadata_function(self):
        v, s, m, M, n, f = [], [], [], [], [], []
        for v in itertools.chain(self.states, self.alg_states, self.parameters, self.constants):
            # TODO
            v_ = v.value if hasattr(v.value, '__iter__') else np.full(v.symbol.size(), v.value if hasattr(v, 'value') else np.nan)
            s_ = v.start if hasattr(v.start, '__iter__') else np.full(v.symbol.size(), v.start if hasattr(v, 'start') is not None else np.nan)
            m_ = v.min if hasattr(v.min, '__iter__') else np.full(v.symbol.size(), v.min if hasattr(v, 'min') is not None else -np.inf)
            M_ = v.max if hasattr(v.max, '__iter__') else np.full(v.symbol.size(), v.max if hasattr(v, 'max') is not None else np.inf)
            n_ = v.nominal if hasattr(v.nominal, '__iter__') else np.full(v.symbol.size(),
                                                                          v.nominal if v.nominal is not None else 1)
            f_ = v.fixed if hasattr(v.fixed, '__iter__') else np.full(v.symbol.size(), v.fixed)

            v.append(v_)
            s.append(s_)
            m.append(m_)
            M.append(M_)
            n.append(n_)
            f.append(f_)
        out = ca.horzcat(ca.veccat(*v), ca.veccat(*s), ca.veccat(*m), ca.veccat(*M), ca.veccat(*n), ca.veccat(*f))
        return ca.Function('variable_metadata', [ca.veccat(*self.parameters)],
                            [out[:len(self.states), :], out[len(self.states):len(self.states) + len(self.alg_states), :], out[len(self.states) + len(self.alg_states):len(self.states) + len(self.alg_states) + len(self.parameters), :], out[len(self.states) + len(self.alg_states) + len(self.parameters):, :]])
