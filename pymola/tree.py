#!/usr/bin/env python
"""
Tools for tree walking and visiting etc.
"""

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
            for child in self.getChildren():
                res += [self.handle_visit(child, visitor)]
            if len(res) == 1:
                res = res[0]
            return res

    @classmethod
    def handle_visit(cls, var, visitor):
        res = []
        if isinstance(var, ast.Node):
            res += var.visit(visitor)
        elif isinstance(var, dict):
            for k in var.keys():
                res += [cls.handle_visit(var[k], visitor)]
        elif isinstance(var, list):
            for c in var:
                res += [cls.handle_visit(c, visitor)]
        if len(res) == 1:
            res = res[0]
        return res


class TreeListener(object):

    def enterFile(self, tree):
        print('walked file')

    def exitFile(self, tree):
        pass

    def enterClass(self, tree):
        pass

    def exitClass(self, tree):
        pass

    def enterExperssion(self, tree):
        pass

    def exitExpression(self, tree):
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
        symbols=[],
        equations=orig_class.equations
    )

    # append period to non empty instance_name
    if instance_name != '':
        instance_name += '.'

    # for all symbols in the original class
    for sym_name, sym in orig_class.symbols.items():
        # if the symbol type is a class
        if sym.type in root.classes:
            # recursively call flatten on the sub class
            flat_sub_class = flatten(root, sym.type, instance_name=sym_name)
            # add sub_class members symbols and equations
            for sub_sym_name, sub_sym in flat_sub_class.symbols.items():
                flat_class.symbols[instance_name + sub_sym_name] = sub_sym
            flat_class.equations += flat_sub_class.equations

        # else if the symbols is not a class name
        else:
            # append original symbol to flat class
            flat_class.symbols[instance_name + sym_name] = sym
    return flat_class


class ComponentRenameListener(object):

    def __init__(self, prefix):
        self.prefix = prefix

    def enterClass(self, tree):
        print('class', tree.name)

    def enterSymbol(self, tree):
        tree.name = '{:s}.{:s}'.format(self.prefix, tree.name)

    def enterComponentRef(self, tree):
        tree.name = '{:s}.{:s}'.format(self.prefix, tree.name)