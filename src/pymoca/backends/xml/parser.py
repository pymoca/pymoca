import os
from collections import OrderedDict
from typing import Union

import casadi as ca
# noinspection PyPackageRequirements

from lxml import etree

from .model import HybridDae

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.path.join(FILE_PATH, 'ModelicaXML', 'schemas')


class XMLParser:

    def __init__(self, schema_dir, schema_file):
        orig_path = os.path.abspath(os.curdir)
        os.chdir(schema_dir)
        with open(schema_file, 'r') as f:
            schema = etree.XMLSchema(etree.XML(f.read().encode('utf-8')))
        os.chdir(orig_path)
        self._parser = etree.XMLParser(
            schema=schema,
            remove_comments=True,
            remove_blank_text=False)

    def parse(self, txt: str):
        if not isinstance(txt, str):
            raise ValueError('txt must be a str')
        xml_file = txt.encode('utf-8')
        return etree.fromstring(xml_file, self._parser)


Sym = Union[ca.MX, ca.SX]


# noinspection PyProtectedMember,PyPep8Naming
class ModelListener:
    """ Converts ModelicaXML file to Hybrid DAE"""

    def __init__(self, sym: Sym = ca.SX, verbose=False):
        self.depth = 0
        self.model = {}
        self.scope_stack = []
        self.verbose = verbose
        self.sym = sym

        # Define an operator map that can be used as
        # self.op_map[n_operations][operator](*args)
        self.op_map = {
            1: {
                'der': self.der,
                '-': lambda x: -1 * x,
                'abs': ca.fabs,
                'sin': ca.sin,
                'cos': ca.cos,
                'tan': ca.tan,
            },
            2: {
                '+': lambda x, y: x + y,
                '-': lambda x, y: x - y,
                '*': lambda x, y: x * y,
                '/': lambda x, y: x / y,
                '^': lambda x, y: ca.power(x, y),
                '>': lambda x, y: x > y,
                '<': lambda x, y: x < y,
                '<=': lambda x, y: x <= y,
                '>=': lambda x, y: x >= y,
                'reinit': self.reinit,
                'sample': self.sample,
                'and': ca.logic_and,
                'or': ca.logic_or,
                'not': ca.logic_not,
                'noise_gaussian': lambda mean, std: self.noise_gaussian(mean, std),
                'noise_uniform': lambda lower, upper: self.noise_uniform(lower, upper),
            },
        }

    @property
    def scope(self):
        return self.scope_stack[-1]

    def call(self, tag_name: str, *args, **kwargs):
        """Convenience method for calling methods with walker."""
        if hasattr(self, tag_name):
            getattr(self, tag_name)(*args, **kwargs)

    # ------------------------------------------------------------------------
    # OPERATORS
    # ------------------------------------------------------------------------

    def der(self, x: Sym):
        """Get the derivative of the variable, create it if it doesn't exist."""
        name = 'der({:s})'.format(x.name())
        if name not in self.scope['dvar'].keys():
            self.scope['dvar'][name] = self.sym.sym(name, *x.shape)
            self.scope['states'].append(x.name())
        return self.scope['dvar'][name]

    def cond(self, expr):
        c = self.sym.sym('c_{:d}'.format(len(self.scope['c'])))
        self.scope['c'][c] = expr
        return c

    def pre_cond(self, x: Sym):
        name = 'pre({:s})'.format(x.name())
        if name not in self.scope['pre_c'].keys():
            self.scope['pre_c'][name] = self.sym.sym(name, *x.shape)
        return self.scope['pre_c'][name]

    def edge(self, c):
        """rising edge"""
        return ca.logic_and(c, ca.logic_not(self.pre_cond(c)))

    @staticmethod
    def reinit(x_old, x_new):
        return 'reinit', x_old, x_new

    @staticmethod
    def sample(t_start, period):
        print('sample', t_start, period)
        return 'sample', t_start, period

    def noise_gaussian(self, mean, std):
        """Create a gaussian noise variable"""
        assert std > 0
        ng = self.sym.sym('ng_{:d}'.format(len(self.scope['ng'])))
        self.scope['ng'].append(ng)
        return mean + std*ng

    def noise_uniform(self, lower_bound, upper_bound):
        """Create a uniform noise variable"""
        assert upper_bound > lower_bound
        nu = self.sym.sym('nu_{:d}'.format(len(self.scope['nu'])))
        self.scope['nu'].append(nu)
        return lower_bound + nu*(upper_bound - lower_bound)

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------

    @staticmethod
    def get_attr(e, name, default):
        if name in e.attrib.keys():
            return e.attrib[name]
        else:
            return default

    def get_var(self, name):
        """Get the variable in the current scope"""
        if name == 'time':
            return self.scope['time']
        else:
            return self.scope['var'][name]

    def log(self, *args, **kwargs):
        """Convenience function for printing indenting debug output."""
        if self.verbose:
            print('   ' * self.depth, *args, **kwargs)

    # ------------------------------------------------------------------------
    # Listener Methods
    # ------------------------------------------------------------------------

    def enter_every_before(self, tree: etree._Element):
        # initialize model to None
        self.model[tree] = None

        # print name to log
        self.log(tree.tag, '{')

        # increment depth
        self.depth += 1

    def exit_every_after(self, tree: etree._Element):
        # decrement depth
        self.depth -= 1

        # self.log('tree:', etree.tostring(tree))

        # print model
        if self.model[tree] is not None:
            self.log('model:', self.model[tree])

        # print name to log
        self.log('}', tree.tag)

    # noinspection PyUnusedLocal
    def enter_classDefinition(self, tree: etree._Element):
        # we don't know if variables are states
        # yet, we need to wait until equations are parsed
        self.scope_stack.append({
            'time': self.sym.sym('time'),
            'sample_times': [],
            'var': OrderedDict(),  # variables
            'states': [],  # list of which variables are states (based on der call)
            'dvar': OrderedDict(),  # derivative of variables
            'eqs': [],  # equations
            'when_eqs': [],  # when equations
            'c': {},  # conditions
            'pre_c': {},  # pre conditions
            'p': [],  # parameters and constants
            'prop': {},  # properties for variables
            'ng': [],  # gaussian
            'nu': [],  # uniform
        })

    def exit_classDefinition(self, tree: etree._Element):  # noqa: too-complex
        dae = HybridDae()
        dae.t = self.scope['time']
        self.model[tree] = dae

        # handle component declarations
        for var_name, v in self.scope['var'].items():
            variability = self.scope['prop'][var_name]['variability']
            if variability == 'continuous':
                if var_name in self.scope['states']:
                    dae.x = ca.vertcat(dae.x, v)
                    dae.dx = ca.vertcat(dae.dx, self.der(v))
                else:
                    dae.y = ca.vertcat(dae.y, v)
            elif variability == 'discrete':
                dae.m = ca.vertcat(dae.m, v)
            elif variability == 'parameter':
                dae.p = ca.vertcat(dae.p, v)
            elif variability == 'constant':
                dae.p = ca.vertcat(dae.p, v)
            else:
                raise ValueError('unknown variability', variability)

        for eq in self.scope['eqs']:
            if isinstance(eq, self.sym):
                dae.f_x = ca.vertcat(dae.f_x, eq)

        # build reinit expression and discrete equations
        dae.f_i = dae.x
        dae.f_m = dae.m
        for eq in self.scope['when_eqs']:
            w = eq['cond']
            for then_eq in eq['then']:
                if isinstance(then_eq, tuple):
                    if then_eq[0] == 'reinit':
                        sub_var = then_eq[1]
                        sub_expr = ca.if_else(self.edge(w), then_eq[2], sub_var)
                        dae.f_i = ca.substitute(dae.f_i, sub_var, sub_expr)
                elif isinstance(then_eq, self.sym):
                    # this is a discrete variable assignment
                    # so it should be a casadi subtraction y = x
                    assert then_eq.is_op(ca.OP_SUB) and then_eq.n_dep() == 2
                    sub_var = then_eq.dep(0)
                    sub_expr = ca.if_else(self.edge(w), then_eq.dep(1), sub_var)
                    dae.f_m = ca.substitute(dae.f_m, sub_var, sub_expr)

        dae.t = self.scope['time']
        dae.prop.update(self.scope['prop'])
        c_dict = self.scope['c']
        for k in c_dict.keys():
            dae.c = ca.vertcat(dae.c, k)
            dae.pre_c = ca.vertcat(dae.pre_c, self.pre_cond(k))
            dae.f_c = ca.vertcat(dae.f_c, c_dict[k])

        for l, r in [('f_c', 'c'), ('c', 'pre_c'), ('dx', 'x'), ('f_m', 'm')]:
            vl = getattr(dae, l)
            vr = getattr(dae, r)
            if vl.shape != vr.shape:
                raise ValueError(
                    '{:s} and {:s} must have the same shape:'
                    '\n{:s}: {:s}\t{:s}: {:s}'.format(
                        l, r, l, str(dae.f_m), r, str(dae.m)))

        dae.ng = ca.vertcat(*self.scope['ng'])
        dae.nu = ca.vertcat(*self.scope['nu'])

        n_eq = dae.f_x.shape[0] + dae.f_m.shape[0]
        n_var = dae.x.shape[0] + dae.m.shape[0] + dae.y.shape[0]
        if n_eq != n_var:
            raise ValueError(
                'must have equal number of equations '
                '{:d} and unknowns {:d}\n:{:s}'.format(
                    n_eq, n_var, str(dae)))
        self.scope_stack.pop()

    def enter_component(self, tree: etree._Element):
        self.model[tree] = {
            'start': None,
            'fixed': None,
            'value': None,
            'variability': self.get_attr(tree, 'variability', 'continuous'),
            'visibility': self.get_attr(tree, 'visibility', 'public'),
        }
        self.scope_stack.append(self.model[tree])

    def exit_component(self, tree: etree._Element):
        var_scope = self.scope_stack.pop()
        name = tree.attrib['name']
        shape = (1, 1)
        sym = self.sym.sym(name, *shape)
        self.scope['prop'][name] = var_scope
        self.scope['var'][name] = sym

    def exit_local(self, tree: etree._Element):
        name = tree.attrib['name']
        self.model[tree] = self.get_var(name)

    def exit_operator(self, tree: etree._Element):
        op = tree.attrib['name']
        self.model[tree] = self.op_map[len(tree)][op](*[self.model[e] for e in tree])

    def exit_if(self, tree: etree._Element):
        assert len(tree) == 3
        cond = self.model[tree[0]]
        then_eq = self.model[tree[1]]
        else_eq = self.model[tree[2]]
        c = self.cond(cond)
        if len(then_eq) != len(else_eq):
            raise SyntaxError("then and else equations must have same number of statements")
        self.model[tree] = ca.if_else(c, then_eq[0], else_eq[0])

    def exit_apply(self, tree: etree._Element):
        op = tree.attrib['builtin']
        self.model[tree] = self.op_map[len(tree)][op](*[self.model[e] for e in tree])

    def exit_equal(self, tree: etree._Element):
        assert len(tree) == 2
        self.model[tree] = self.model[tree[0]] - self.model[tree[1]]

    def exit_equation(self, tree: etree._Element):
        self.model[tree] = [self.model[c] for c in tree]
        self.scope['eqs'].extend(self.model[tree])

    def exit_modifier(self, tree: etree._Element):
        props = {}
        for e in tree:
            props.update(self.model[e])
        self.model[tree] = props
        self.scope.update(props)

    def exit_item(self, tree: etree._Element):
        assert len(tree) == 1
        self.model[tree] = {
            tree.attrib['name']: self.model[tree[0]]
        }

    def exit_real(self, tree: etree._Element):
        self.model[tree] = float(tree.attrib["value"])

    def exit_true(self, tree: etree._Element):
        self.model[tree] = True

    def exit_false(self, tree: etree._Element):
        self.model[tree] = False

    def exit_modelica(self, tree: etree._Element):
        # get all class definitions as a list
        self.model[tree] = [self.model[c] for c in tree[0]]

    def exit_when(self, tree: etree._Element):
        assert len(tree) == 2
        cond = self.model[tree[0]]
        then = self.model[tree[1]]
        self.model[tree] = {
            'cond': self.cond(cond),
            'then': then
        }
        self.scope['when_eqs'].append(self.model[tree])

    def exit_cond(self, tree: etree._Element):
        assert len(tree) == 1
        self.model[tree] = self.model[tree[0]]

    def exit_then(self, tree: etree._Element):
        self.model[tree] = [self.model[c] for c in tree]

    def exit_else(self, tree: etree._Element):
        self.model[tree] = [self.model[c] for c in tree]


# noinspection PyProtectedMember
def walk(e: etree._Element, l: ModelListener) -> None:
    tag = e.tag
    l.call('enter_every_before', e)
    l.call('enter_' + tag, e)
    l.call('enter_every_after', e)
    for c in e.getchildren():
        walk(c, l)
    l.call('exit_every_before', e)
    l.call('exit_' + tag, e)
    l.call('exit_every_after', e)


def parse(model_txt: str, verbose: bool = False) -> HybridDae:
    parser = XMLParser(SCHEMA_DIR, 'Modelica.xsd')
    root = parser.parse(model_txt)
    listener = ModelListener(verbose=verbose)
    walk(root, listener)
    return listener.model[root][0]


def parse_file(file_path: str, verbose: bool = False) -> HybridDae:
    with open(file_path, 'r') as f:
        txt = f.read()
    return parse(txt, verbose)
