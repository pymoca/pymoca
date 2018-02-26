import numpy as np
import logging
import casadi as ca

from pymola.tree import TreeListener, TreeWalker, flatten
import pymola.ast as ast
from pymola.parser import parse

from .model import Dae


class CasadiListener(TreeListener):
    """
    Modelica DAE Creation Steps
    -------------------------------------
    see Intro to Modelica/ Fritzson

    1. Constant variables are collected into paremter vector $p$. All other constants can be replaced by their values.
    2. Variables declared as input on root model are put in the input vector $u$.
    3. Variables whose derivatives appear in the model are put in the state vector $x$.
    4. All other variables are collected into a vector y of algebraic variables (their derivatives do not appear in the model).
    """

    def __init__(self, sym_class=ca.MX, print_depth: int = -1, log_level=logging.WARNING):
        super().__init__()
        
        # unary operation translation
        self._unary_ops = {
            '+': lambda x: x,
            '-': lambda x: -x,
            'sin': lambda x: ca.sin(x),
            'cos': lambda x: ca.cos(x),
            'tan': lambda x: ca.tan(x),
            'sqrt': lambda x: ca.sqrt(x),
            'der': lambda x: self.der(x)
        }
        
        # binary opeation translation
        self._binary_ops = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '^': lambda x, y: ca.power(x, y)
        }
        self._indent = "--"
        self._depth = 0
        self._print_depth = print_depth
        self._expr = {}  # type: Dict[ast.Node, Union[ca.MX, ca.SX]]
        self._vars = {}
        self.model = Dae()
        self.model.sym_class = sym_class # type: Class
        self._logger = logging.getLogger('CasadiListener')
        self._logger.setLevel(log_level)
        self._time = self.sym('time')


    def sym(self, *args, **kwargs):
        name = args[0]
        self._logger.info('%s creating symbol %s', self._depth*self._indent, name)
        if name in self._vars.keys():
            raise RuntimeError("symbols {:s} already present".format(name))
        v = getattr(self.model.sym_class, 'sym')(*args, **kwargs)
        self._vars[name] = v
        return v

    def get_sym(self, name):
        if name == 'time':
            return self._time
        else:
            return self._vars[name]

    def der(self, s):
        return self.get_sym('der({:s})'.format(s.name()))

    def const(self, *args, **kwargs):
        name = args[0]
        return self.model.sym_class(*args, **kwargs)
        
    def get(self, tree):
        """
        Get the expression representation of the tree.
        """
        if tree not in self._expr.keys():
            raise RuntimeError("no model for node of type", type(tree))
        return self._expr[tree]
    
    def set(self, tree: ast.Node, value):
        """
        Set the expression representation of the tree.
        """
        self._expr[tree] = value

    def enterEvery(self, tree: ast.Node):
        if self._depth <= self._print_depth:
            self._logger.info('%s%s', self._depth*self._indent, tree.__class__.__name__)
        self._depth += 1

    def exitEvery(self, tree: ast.Node):
        self._depth -= 1

    def exitTree(self, tree: ast.Tree):
        assert len(tree.classes) == 1
        name = list(tree.classes.keys())[0]
        cls = tree.classes[name]
        self.set(tree, self.get(cls))
        equations = self.get(tree)
        dae_eq = []
        alg_eq = []
        for eq in equations:
            if ca.depends_on(eq, self.model.der_state):
                dae_eq.append(eq)
            else:
                alg_eq.append(eq)
        self.model.dae_eq = ca.vertcat(*dae_eq)
        self.model.alg_eq = ca.vertcat(*alg_eq)
        self.model.time = self._time

    def exitClass(self, tree: ast.Class):
        eqs = [self.get(eq) for eq in tree.equations]
        self.set(tree, eqs)

    def exitEquationSection(self, tree: ast.EquationSection):
        pass
        
    def exitEquation(self, tree: ast.Equation):
        self.set(tree, self.get(tree.left) - self.get(tree.right))

    def exitExpression(self, tree: ast.Expression):
        if len(tree.operands) == 1:
            right = self.get(tree.operands[0])
            assert isinstance(right, self.model.sym_class)
            if isinstance(tree.operator, ast.ComponentRef):
                self.set(tree, self._unary_ops[tree.operator.name](right))
            elif isinstance(tree.operator, str):
                self.set(tree, self._unary_ops[tree.operator](right))
            else:
                raise NotImplementedError("unary op not implemented for type", type(tree.operator))
        elif len(tree.operands) == 2:
            left = self.get(tree.operands[0])
            right = self.get(tree.operands[1])
            assert isinstance(left, self.model.sym_class)
            assert isinstance(right, self.model.sym_class)
            if isinstance(tree.operator, str):
                self.set(tree, self._binary_ops[tree.operator](left, right))
            else:
                raise NotImplementedError("binary op not implemented for type", type(tree.operator))

        if tree not in self._expr.keys():
            print('skip expression, #ops: ', len(tree.operands), 'operand:', tree.operator, 'type:', type(tree.operands[0]))

    def exitSymbol(self, tree: ast.Symbol):
        dim = [d.value for d in tree.dimensions]
        s = self.sym(tree.name, *dim)
        type_prefixes = ['parameter', 'constant', 'state', 'input', 'output']
        type_prefix_list = set(type_prefixes).intersection(tree.prefixes)
        if len(type_prefix_list) == 1:
            type_prefix = list(type_prefix_list)[0]
        elif len(type_prefix_list) == 0:
            type_prefix = "alg_state"
        else:
            raise RuntimeError("ambiguous prefxies", tree.prefixes)

        if (type_prefix == "input" or type_prefix == "output") and self._depth > 2:
            type_prefix = "alg_state"

        if tree.name not in ca.vertsplit(getattr(self.model, type_prefix)):
            v = getattr(self.model, type_prefix)
            setattr(self.model, type_prefix,
                ca.vertcat(getattr(self.model, type_prefix), s))
            if type_prefix == "state":
                s_dot = self.sym('der({:s})'.format(tree.name, *dim))
                self.model.der_state = ca.vertcat(self.model.der_state, s_dot)

        self.model.property[s] = {
            'comment': tree.comment,
            'tree.prefixes': tree.prefixes,
            'start': tree.start.value,
            'value': tree.value.value,
            'min': tree.min.value,
            'max': tree.max.value,
            'nominal': tree.nominal.value,
            'dimentaions': dim
        }
            
    def exitComponentRef(self, tree: ast.ComponentRef):
        #TODO need to handle type and function refs properly
        if tree.name in self._unary_ops.keys() or tree.name == "Real":
            return
        try:
            self.set(tree, self.get_sym(tree.name))
        except KeyError as e:
            print('key error', e)

    def exitPrimary(self, tree: ast.Primary):
        if tree.value is None:
            return
        try:
            self.set(tree, self.const(tree.value))
        except:
            self.set(tree, tree.value)

def generate(model_txt, name, sym_class=ca.SX, print_depth=10, log_level=logging.WARN):
    model_ast = flatten(parse(model_txt), ast.ComponentRef(name=name))
    walker = TreeWalker()
    listener = CasadiListener(
        sym_class=sym_class, print_depth=print_depth, log_level=log_level)
    walker.walk(listener, model_ast)
    model = listener.model
    if not model.balanced:
        print('WARNING: model not balanced')
    return model
