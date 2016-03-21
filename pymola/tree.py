#!/usr/bin/env python
"""
Tools for tree walking and visiting etc.
"""

from __future__ import print_function, absolute_import, division, print_function, unicode_literals

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
    def enterFile(self, tree):
        pass

    def exitFile(self, tree):
        pass

    def enterClass(self, tree):
        pass

    def exitClass(self, tree):
        pass

    def enterExpression(self, tree):
        pass

    def exitExpression(self, tree):
        pass

    def exitConnectClause(self, tree):
        pass

    def exitSymbol(self, tree):
        pass

    def exitComponentClause(self, tree):
        pass

    def exitPrimary(self, tree):
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
        equations=orig_class.equations,
        states = orig_class.states,
        inputs = orig_class.inputs,
        outputs = orig_class.outputs,
        parameters = orig_class.parameters,
        constants = orig_class.constants,
        variables = orig_class.variables,
    )

    # flat file
    flat_file = ast.File()
    flat_file.classes[class_name] = flat_class

    # append period to non empty instance_name
    if instance_name != '':
        instance_prefix = instance_name + '.'
    else:
        instance_prefix = instance_name

    # for all symbols in the original class
    for sym_name, sym in orig_class.symbols.items():
        # if the symbol type is a class
        if sym.type in root.classes:
            # recursively call flatten on the sub class
            flat_sub_file = flatten(root, sym.type, instance_name=sym_name)
            flat_sub_class = flat_sub_file.classes[sym.type]

            # add sub_class members symbols and equations
            for sub_sym_name, sub_sym in flat_sub_class.symbols.items():
                flat_class.symbols[instance_prefix + sub_sym_name] = sub_sym
            flat_class.equations += flat_sub_class.equations
            flat_class.states += flat_sub_class.states
            #flat_class.inputs += flat_sub_class.inputs
            #flat_class.outputs += flat_sub_class.outputs
            flat_class.variables += flat_sub_class.variables
            flat_class.parameters += flat_sub_class.parameters
            flat_class.constants += flat_sub_class.constants

        # else if the symbols is not a class name
        else:
            # append original symbol to flat class
            flat_class.symbols[instance_prefix + sym_name] = sym


    # walker for renaming components
    if instance_name != '':
        ast_walker = TreeWalker()
        ast_walker.walk(ComponentRenameListener(instance_name), flat_class)

    return flat_file


class ComponentRenameListener(object):
    def __init__(self, prefix):
        self.prefix = prefix

    def enterSymbol(self, tree):
        tree.name = '{:s}.{:s}'.format(self.prefix, tree.name)

    def enterComponentRef(self, tree):
        tree.name = '{:s}.{:s}'.format(self.prefix, tree.name)