import copy
import os

from lxml import etree, objectify

from pymoca import ast
from pymoca.tree import TreeListener, TreeWalker, flatten

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
# noinspection PyUnresolvedReferences
BUILTINS = dir(__builtins__) + ['psi']

E = objectify.E


class XmlGenerator(TreeListener):

    def __init__(self):
        super().__init__()
        self.xml = {}

    def exitEquation(self, tree: ast.Equation):
        self.xml[tree] = E(
            'equal',
            self.xml[tree.left],
            self.xml[tree.right],
        )

    def exitExpression(self, tree: ast.Expression):
        if isinstance(tree.operator, ast.ComponentRef):
            op_name = tree.operator.name
        else:
            op_name = tree.operator
        if len(tree.operands) == 1:
            self.xml[tree] = E(
                'operator', name=op_name,
                *[self.xml[c] for c in tree.operands])
        else:
            self.xml[tree] = E(
                'apply', builtin=op_name,
                *[self.xml[c] for c in tree.operands])

    def exitPrimary(self, tree: ast.Primary):
        self.xml[tree] = E('real', value=str(tree.value))

    def exitComponentRef(self, tree: ast.ComponentRef):
        self.xml[tree] = E('local', name=tree.name)

    def exitSymbol(self, tree: ast.Symbol):  # noqa: C901
        variability = None
        for v_type in ['discrete', 'continuous', 'parameter', 'constant']:
            if v_type in tree.prefixes:
                variability = v_type
                break
        items = []
        for f in ['start', 'value']:
            val = getattr(tree, f).value
            if val is None:
                continue
            if val == 0 and f == 'start':
                # this is already default
                continue
            items.append(
                E('item', E('real', value=str(val)), name=f))

        for f in ['fixed']:
            val = getattr(tree, f).value
            if val is None:
                continue
            if val:
                val_xml = E('true')
            else:
                if f in ['fixed']:
                    # this is already default
                    continue
                val_xml = E('false')
            items.append(
                E('item', val_xml, name=f))

        modifier = E(
            'modifier',
            * items,
        )
        e = E(
            'component',
            E('builtin', name=tree.type.name),
            modifier,
            name=tree.name,
        )  # type: etree._Element
        if variability is not None:
            e.attrib['variability'] = variability
        self.xml[tree] = e

    def exitClassModification(self, tree: ast.ClassModification):
        self.xml[tree] = E(
            'modifier',
            *[self.xml[arg] for arg in tree.arguments]
        )

    def exitFunction(self, tree: ast.Function):
        self.xml[tree] = E(
            'apply',
            builtin=tree.name,
            *[self.xml[arg] for arg in tree.arguments]
        )

    def exitWhenEquation(self, tree: ast.WhenEquation):
        self.xml[tree] = E(
            'when',
            E(
                'cond', self.xml[tree.conditions[0]],
            ),
            E(
                'then', *[self.xml[b] for b in tree.blocks[0]],
            ),
        )

    def exitClass(self, tree: ast.Class):
        self.xml[tree] = E(
            'classDefinition',
            E(
                'class',
                *[self.xml[s] for s in tree.symbols.values()],
                E(
                    'equation',
                    *[self.xml[s] for s in tree.equations]
                ),
                kind='model',
            ),
            name=tree.name
        )

    def exitTree(self, tree: ast.Tree):
        self.xml[tree] = E(
            'modelica',
            E(
                'declarations',
                *[self.xml[c] for c in tree.classes.values()]
            ),
            format="1.0"
        )


def generate(ast_tree: ast.Tree, model_name: str):
    """
    :param ast_tree: AST to generate from
    :param model_name: class to generate
    :return: sympy source code for model
    """
    component_ref = ast.ComponentRef.from_string(model_name)
    ast_tree_new = copy.deepcopy(ast_tree)
    ast_walker = TreeWalker()
    flat_tree = flatten(ast_tree_new, component_ref)
    gen = XmlGenerator()
    ast_walker.walk(gen, flat_tree)
    return etree.tostring(gen.xml[flat_tree], pretty_print=True).decode('utf-8')
