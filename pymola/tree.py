#!/usr/bin/env python
"""
Tools for tree walking and visiting etc.
"""

from __future__ import print_function, absolute_import, division, unicode_literals
import logging
import copy

from . import ast

logger = logging.getLogger("pymola")

# TODO class name spaces
# TODO dot notation
# TODO find_symbol, find_class
# TODO Flatten function vs. conversion classes


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

    def flatten_symbol(sym, instance_prefix):
        sym_copy = copy.deepcopy(sym)
        sym_copy.name = instance_prefix + sym.name
        return sym_copy

    def flatten_expression(expression, instance_prefix):
        expression_copy = copy.deepcopy(expression)

        class ExpressionFlattener(TreeListener):
            def __init__(self, instance_prefix):
                self.instance_prefix = instance_prefix
                self.d = 0

                super(ExpressionFlattener, self).__init__()

            def enterComponentRef(self, tree):
                self.d += 1

                # TODO: Handle array indices in name flattening
                if self.d == 1:
                    tree.name = self.instance_prefix + tree.name
                    c = tree
                    while len(c.child) > 0:
                        c = tree.child[0]
                        tree.name += '.' + c.name
                    tree.child = []

            def exitComponentRef(self, tree):
                self.d -= 1

        w = TreeWalker()
        w.walk(ExpressionFlattener(instance_prefix), expression_copy)

        return expression_copy


    # pull in parent classes
    for extends in orig_class.extends:
        c = root.find_class(extends.component)

        # recursively call flatten on the parent class
        flat_parent_file = flatten(root, c.name, instance_name=instance_name)
        flat_parent_class = flat_parent_file.classes[c.name]

        # add parent class members symbols, equations and statements
        for parent_sym_name, parent_sym in flat_parent_class.symbols.items():
            flat_sym = flatten_symbol(parent_sym, instance_prefix)
            flat_class.symbols[flat_sym.name] = flat_sym
        flat_class.equations += [flatten_expression(e, instance_prefix) for e in flat_parent_class.equations]
        flat_class.statements += [flatten_expression(e, instance_prefix) for e in flat_parent_class.statements]

        # carry out modifications
        modify_class(c, extends.class_modification)

    # for all symbols in the original class
    for sym_name, sym in orig_class.symbols.items():
        # if the symbol type is a class
        try:
            class_data = root.find_class(sym.type)
            if class_data.type == 'connector':
                flat_class.symbols[instance_prefix + sym_name] = copy.deepcopy(sym)
                for sym_name2, sym2 in class_data.symbols.items():
                    flat_class.symbols[instance_prefix + sym_name + "." + sym_name2] = copy.deepcopy(sym2)

            # recursively call flatten on the sub class
            flat_sub_file = flatten(root, sym.type.name, instance_name=sym_name)
            flat_sub_class = flat_sub_file.find_class(sym.type) # TODO

            # add sub_class members symbols and equations
            for sub_sym_name, sub_sym in flat_sub_class.symbols.items():
                flat_sym = flatten_symbol(sub_sym, instance_prefix)
                flat_class.symbols[flat_sym.name] = flat_sym
            flat_class.equations += [flatten_expression(e, instance_prefix) for e in flat_sub_class.equations]
            flat_class.statements += [flatten_expression(e, instance_prefix) for e in flat_sub_class.statements]

        except KeyError:
            # append original symbol to flat class
            flat_sym = flatten_symbol(sym, instance_prefix)
            flat_class.symbols[flat_sym.name] = flat_sym

    # for all equations in original class
    flow_connections = {}
    for equation in orig_class.equations:
        flat_equation = flatten_expression(equation, instance_prefix)
        if isinstance(equation, ast.ConnectClause):
            # expand connector
            connect_equations = []
            sym_left = root.find_symbol(orig_class, equation.left)
            sym_right = root.find_symbol(orig_class, equation.right)

            try:
                class_left = root.find_class(sym_left.type)
                class_right = root.find_class(sym_right.type)

                assert(class_left == class_right)

                flat_file_left = flatten(root, class_left.name)
                flat_class_left = flat_file_left.classes[class_left.name]

                for connector_variable in flat_class_left.symbols.values():
                    left_name = flat_equation.left.name + '.' + connector_variable.name
                    right_name = flat_equation.right.name + '.' + connector_variable.name
                    if len(connector_variable.prefixes) == 0:
                        left = ast.ComponentRef(name=left_name)
                        right = ast.ComponentRef(name=right_name)
                        connect_equation = ast.Equation(left=left, right=right)
                        connect_equations.append(connect_equation)
                    elif connector_variable.prefixes == ['flow']:
                        left_connected_variables = flow_connections.get(left_name, set())
                        right_connected_variables = flow_connections.get(right_name, set())

                        connected_variables = left_connected_variables.union(right_connected_variables)
                        if left_name not in connected_variables:
                            connected_variables.add(left_name)
                        if right_name not in connected_variables:
                            connected_variables.add(right_name)

                        connected_variables = frozenset(connected_variables)
                        for connected_variable in connected_variables:
                            flow_connections[connected_variable] = connected_variables
                    else:
                        raise Exception("Unsupported connector variable prefixes {}".format(connector_variable.prefixes))
            except KeyError:
                logger.debug("Connector class not defined.  Assuming it to be an elementary type.")

                connect_equation = ast.Equation(left=flat_equation.left, right=flat_equation.right)
                connect_equations.append(connect_equation)

            # TODO if flow in prefixes:  flow_equalities[port_a] = flow_equalities[port_b] = flow_variables,

            flat_class.equations += connect_equations
        else:
            # flatten equation
            flat_class.equations += [flat_equation]

    flat_class.statements += [flatten_expression(e, instance_prefix) for e in orig_class.statements]

    # add flow equations
    if len(flow_connections) > 0:
        # TODO Flatten first
        logger.warning("Note: Connections between connectors with flow variables are not supported across levels of the class hierarchy")
    for connected_variables in set(flow_connections.values()):
        operands = [ast.ComponentRef(name=variable) for variable in connected_variables]
        connect_equation = ast.Equation(left=ast.Expression(operator='+', operands=operands), right=ast.Primary(value=0))
        flat_class.equations += [connect_equation]

    return flat_file
