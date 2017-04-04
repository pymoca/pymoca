#!/usr/bin/env python
"""
Tools for tree walking and visiting etc.
"""

from __future__ import print_function, absolute_import, division, print_function, unicode_literals
import copy

from . import ast


class TreeWalker(object):

    def walk(self, listener, tree):
        name = tree.__class__.__name__
        if hasattr(listener, 'enterEvery'):
            getattr(listener, 'enterEvery')(tree)
        if hasattr(listener, 'enter' + name):
            getattr(listener, 'enter' + name)(tree)
        for child_name in tree.ast_spec.keys():
            self.handle_walk(self, listener, tree.__dict__[child_name])
        if hasattr(listener, 'exitEvery'):
            getattr(listener, 'exitEvery')(tree)
        if hasattr(listener, 'exit' + name):
            getattr(listener, 'exit' + name)(tree)

    @classmethod
    def handle_walk(cls, walker, listener, tree):
        if isinstance(tree, ast.Node):
            walker.walk(listener, tree)
        elif isinstance(tree, dict):
            for k in tree.keys():
                cls.handle_walk(walker, listener, tree[k])
        elif isinstance(tree, list):
            for i in range(len(tree)):
                cls.handle_walk(walker, listener, tree[i])
        else:
            pass


class TreeVisitor(object):
    """
    TODO this class hasn't been tested
    """

    def visit(self, visitor, tree):
        name = self.__class__.__name__
        if hasattr(visitor, 'visit' + name):
            return visitor.__getattribute__('visit' + name)(tree)
        else:
            res = []
            for k in tree.ast_spec.keys():
                res += [self.handle_visit(visitor, tree[k])]
            if len(res) == 1:
                res = res[0]
            return res

    @classmethod
    def handle_visit(cls, visitor, tree):
        res = []
        if isinstance(tree, ast.Node):
            res += visitor.visit(visitor, tree)
        elif isinstance(tree, dict):
            for k in tree.keys():
                res += [cls.handle_visit(tree[k], visitor)]
        elif isinstance(tree, list):
            for c in tree:
                res += [cls.handle_visit(c, visitor)]
        if len(res) == 1:
            res = res[0]
        return res


class TreeListener(object):

    def __init__(self):
        self.context = {}

    def enterEvery(self, tree):
        self.context[type(tree).__name__] = tree

    def exitEvery(self, tree):
        self.context[type(tree).__name__] = None

    def enterFile(self, tree):
        pass

    def exitFile(self, tree):
        pass

    def enterClass(self, tree):
        pass

    def exitClass(self, tree):
        pass

    def enterImportAsClause(self, tree):
        pass

    def exitImportAsClause(self, tree):
        pass

    def enterImportFromClause(self, tree):
        pass

    def exitImportFromClause(self, tree):
        pass

    def enterElementModification(self, tree):
        pass

    def exitElementModification(self, tree):
        pass

    def enterClassModification(self, tree):
        pass

    def exitClassModification(self, tree):
        pass

    def enterExtendsClause(self, tree):
        pass

    def exitExtendsClause(self, tree):
        pass

    def enterIfExpression(self, tree):
        pass

    def exitIfExpression(self, tree):
        pass

    def enterExpression(self, tree):
        pass

    def exitExpression(self, tree):
        pass

    def enterIfEquation(self, tree):
        pass

    def exitIfEquation(self, tree):
        pass

    def enterForIndex(self, tree):
        pass

    def exitForIndex(self, tree):
        pass

    def enterForEquation(self, tree):
        pass

    def exitForEquation(self, tree):
        pass

    def enterEquation(self, tree):
        pass

    def exitEquation(self, tree):
        pass

    def enterConnectClause(self, tree):
        pass

    def exitConnectClause(self, tree):
        pass

    def enterSymbol(self, tree):
        pass

    def exitSymbol(self, tree):
        pass

    def enterComponentClause(self, tree):
        pass

    def exitComponentClause(self, tree):
        pass

    def enterArray(self, tree):
        pass

    def exitArray(self, tree):
        pass

    def enterSlice(self, tree):
        pass

    def exitSlice(self, tree):
        pass

    def enterPrimary(self, tree):
        pass

    def exitPrimary(self, tree):
        pass

    def enterComponentRef(self, tree):
        pass

    def exitComponentRef(self, tree):
        pass


def flatten(root, class_name, instance_name=''):
    """
    This function takes and flattens it so that all subclasses instances
    are replaced by the their equations and symbols with name mangling
    of the instance name passed.
    :param root: The root of the tree that contains all class definitions
    :param class_name: The class we want to flatten
    :param instance_name:
    :return: flat_class, the flattened class of type Class
    """

    # extract the original class of interest
    orig_class = root.classes[class_name]

    # create the returned class
    flat_class = ast.Class(
        name=class_name,
        equations=copy.deepcopy(orig_class.equations),
    )

    # flat file
    flat_file = ast.File()
    flat_file.classes[class_name] = flat_class

    # append period to non empty instance_name
    if instance_name != '':
        instance_prefix = instance_name + '.'
    else:
        instance_prefix = instance_name

    def modify_class(c, modification):
        for argument in modification.arguments:
            if argument.name in ast.Symbol.ATTRIBUTES:
                setattr(c, argument.name, argument.modifications[0])
            else:
                sym = flat_class.symbols[argument.name]

                for modification in argument.modifications:
                    if isinstance(modification, ast.ClassModification):
                        modify_class(sym, modification)
                    else:
                        sym.value = modification   

    # pull in parent classes
    for extends in orig_class.extends:
        c = root.find_class(extends.component)

        # recursively call flatten on the parent class
        flat_parent_file = flatten(root, c.name, instance_name=instance_name)
        flat_parent_class = flat_parent_file.classes[c.name]

        # add parent class members symbols and equations
        for parent_sym_name, parent_sym in flat_parent_class.symbols.items():
            flat_class.symbols[instance_prefix + parent_sym_name] = copy.deepcopy(parent_sym)
        flat_class.equations += copy.deepcopy(flat_parent_class.equations)

        # carry out modifications
        modify_class(c, extends.class_modification)

    # create a walker
    ast_walker = TreeWalker()

    # for all symbols in the original class
    for sym_name, sym in orig_class.symbols.items():
        # if the symbol type is a class
        try:
            class_data = root.find_class(sym.type)
        except KeyError:
            # append original symbol to flat class
            flat_class.symbols[instance_prefix + sym_name] = copy.deepcopy(sym)
            continue
        if class_data.type == 'connector':
            continue

        # recursively call flatten on the sub class
        flat_sub_file = flatten(root, sym.type.name, instance_name=sym_name)
        flat_sub_class = flat_sub_file.find_class(sym.type)

        # add sub_class members symbols and equations
        for sub_sym_name, sub_sym in flat_sub_class.symbols.items():
            flat_class.symbols[instance_prefix + sub_sym_name] = copy.deepcopy(sub_sym)
        flat_class.equations += copy.deepcopy(flat_sub_class.equations)

    # walker for expanding connect equations
    # ast_walker.walk(ConnectExpanderListener(), flat_class)

    return flat_file


class Instatiator(TreeListener):
    """
    Instantiates all classes that are not connectors.
    """

    def __init__(self, classes, scope=[]):
        super(Instatiator, self).__init__()
        self.classes = classes
        self.res = {}
        self.scope = scope

    @staticmethod
    def get_scoped_name(scope, name):
        scope = copy.deepcopy(scope)
        scope += [str(name)]
        return '.'.join(scope)

    def enterComponentRef(self, tree):
        name = self.get_scoped_name(self.scope, tree.name)
        tree.name = name

    def exitSymbol(self, tree):
        """
        Set the result of classes to their definitions if it is
        not a connector, otherwise set the result to the symbol
        """
        if tree.type in self.classes:
            class_def = self.classes[tree.type]
            if class_def.type == 'connector':
                self.res[tree] = tree
            else:
                self.res[tree] = class_def
        else:
            self.res[tree] = tree

    def exitEquation(self, tree):
        self.res[tree] = tree

    def exitClass(self, tree):
        """
        For each nested class definition, expand its attributes,
        fore each equation and symbol, copy them to the result
        """
        c = ast.Class()
        for key, val in tree.symbols.items():
            name = self.get_scoped_name(self.scope, key)
            root = self.res[val]
            if type(root) == ast.Class:
                walker = TreeWalker()
                instantiator = Instatiator(
                    classes=self.classes,
                    scope=self.scope + [key]
                )
                walker.walk(instantiator, root)
                c.symbols.update(instantiator.res[root].symbols)
                c.equations.extend(instantiator.res[root].equations)
            elif type(root) == ast.Symbol:
                root.name = name
                c.symbols[name] = root
            else:
                raise RuntimeError('unhandled type', type(root))
        c.equations.extend(tree.equations)
        self.res[tree] = c


class ConnectExpander(TreeListener):

    def __init__(self, classes):
        super(ConnectExpander, self).__init__()
        self.scope = []
        self.classes = classes

    def enterConnectClause(self, tree):
        left_class = self.context['Class'].symbols[tree.left.name]
        right_class =self.context['Class'].symbols[tree.right.name]
        assert left_class.type == 'connector'
        assert right_class.type == 'connector'
        assert left_class == right_class