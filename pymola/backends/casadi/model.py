from collections import namedtuple, OrderedDict
import casadi as ca
import itertools
import logging
import re

from .alias_relation import AliasRelation

logger = logging.getLogger("pymola")

VariableMetadata = namedtuple('VariableMetadata', ['start', 'min', 'max', 'nominal', 'fixed'])


# noinspection PyUnresolvedReferences
class CasadiSysModel:
    def __init__(self):
        self.states = []
        self.state_metadata = []
        self.der_states = []
        self.alg_states = []
        self.alg_state_metadata = []
        self.inputs = []
        self.outputs = []
        self.constants = []
        self.constant_values = []
        self.parameters = []
        self.parameter_values = []
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
        r += "constant_values: " + str(self.constant_values) + "\n"
        r += "parameters: " + str(self.parameters) + "\n"
        r += "equations: " + str(self.equations) + "\n"
        r += "initial equations: " + str(self.initial_equations) + "\n"
        return r

    def check_balanced(self):
        n_states = sum(v.size1() * v.size2() for v in itertools.chain(self.states, self.alg_states))
        n_inputs = sum(v.size1() * v.size2() for v in self.inputs)
        n_equations = sum(e.size1() * e.size2() for e in self.equations)
        if n_states - n_inputs == n_equations:
            logger.info("System is balanced.")
        else:
            logger.warning(
                "System is not balanced.  "
                "Number of states minus inputs is {}, number of equations is {}.".format(
                    n_states - n_inputs, n_equations))

    def simplify(self, options):
        if options.get('replace_constants', False):
            logger.info("Replacing constants")

            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, self.constants, self.constant_values)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, self.constants, self.constant_values)
            self.constants = []
            self.constant_values = []

        if options.get('replace_parameter_expressions', False):
            logger.info("Replacing parameter expressions")

            composite_parameters, simple_parameters = [], []
            composite_parameter_values, simple_parameter_values = [], []
            for e, v in zip(self.parameters, self.parameter_values):
                is_composite = isinstance(v, ca.MX) and not v.is_constant()
                (simple_parameters, composite_parameters)[is_composite].append(e)
                (simple_parameter_values, composite_parameter_values)[is_composite].append(
                    float(v) if not is_composite and isinstance(v, ca.MX) else v)

            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, composite_parameters, composite_parameter_values)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, composite_parameters, composite_parameter_values)
            self.parameters = simple_parameters
            self.parameter_values = simple_parameter_values

        if options.get('eliminable_variable_expression', None) is not None:
            logger.info("Elimating variables that match the regular expression {}".format(options['eliminable_variable_expression']))

            p = re.compile(options['eliminable_variable_expression'])

            alg_states = OrderedDict({s.name() : s for s in self.alg_states})

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

            states = OrderedDict({s.name() : s for s in self.states})
            alg_states = OrderedDict({s.name() : s for s in self.alg_states})
            inputs = OrderedDict({s.name() : s for s in self.inputs})
            outputs = OrderedDict({s.name() : s for s in self.outputs})

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
                    variables.append(alg_states[alias])
                    values.append(sign * canonical_state)

                    del alg_states[alias]
                    if alias in inputs:
                        inputs[alias] = sign * canonical_state
                    if alias in outputs:
                        outputs[alias] = sign * canonical_state

            self.alg_states = list(alg_states.values())
            self.inputs = list(inputs.values())
            self.outputs = list(outputs.values())
            self.equations = reduced_equations

            if len(self.equations) > 0:
                self.equations = ca.substitute(self.equations, variables, values)
            if len(self.initial_equations) > 0:
                self.initial_equations = ca.substitute(self.initial_equations, variables, values)

    def dae_residual_function(self, group_arguments=True):
        if group_arguments:
            return ca.Function('dae_residual', [self.time, ca.vertcat(*self.states), ca.vertcat(*self.der_states),
                                                ca.vertcat(*self.alg_states), ca.vertcat(*self.constants),
                                                ca.vertcat(*self.parameters)], [ca.vertcat(*self.equations)])
        else:
            return ca.Function('dae_residual', [
                self.time] + self.states + self.der_states + self.alg_states + self.constants + self.parameters,
                               self.equations)

    # noinspection PyUnusedLocal
    def initial_residual_function(self, group_arguments=True):
        if group_arguments:
            return ca.Function('initial_residual', [self.time, ca.vertcat(*self.states), ca.vertcat(*self.der_states),
                                                ca.vertcat(*self.alg_states), ca.vertcat(*self.constants),
                                                ca.vertcat(*self.parameters)], [ca.vertcat(*self.initial_equations)])
        else:
            return ca.Function('initial_residual', [
                self.time] + self.states + self.der_states + self.alg_states + self.constants + self.parameters,
                               self.initial_equations)

    # noinspection PyPep8Naming
    def state_metadata_function(self, group_arguments=True):
        s, m, M, n, f = [], [], [], [], []
        for e, v in zip(itertools.chain(self.states, self.alg_states),
                        itertools.chain(self.state_metadata, self.alg_state_metadata)):
            s_ = v.start if hasattr(v.start, '__iter__') else np.full(e.size(), v.start if v.start is not None else np.nan)
            m_ = v.min if hasattr(v.min, '__iter__') else np.full(e.size(), v.min if v.min is not None else -np.inf)
            M_ = v.max if hasattr(v.max, '__iter__') else np.full(e.size(), v.max if v.max is not None else np.inf)
            n_ = v.nominal if hasattr(v.nominal, '__iter__') else np.full(e.size(),
                                                                          v.nominal if v.nominal is not None else 1)
            f_ = v.fixed if hasattr(v.fixed, '__iter__') else np.full(e.size(), v.fixed)

            s.append(s_)
            m.append(m_)
            M.append(M_)
            n.append(n_)
            f.append(f_)
        out = ca.horzcat(ca.vertcat(*s), ca.vertcat(*m), ca.vertcat(*M), ca.vertcat(*n), ca.vertcat(*f))
        if group_arguments:
            return ca.Function('state_metadata', [ca.vertcat(*self.parameters)],
                               [out[:len(self.state_metadata), :], out[len(self.state_metadata):, :]])
        else:
            return ca.Function('state_metadata', self.parameters, [out])