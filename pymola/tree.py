#!/usr/bin/env python
"""
Tools for tree walking and visiting etc.
"""

from __future__ import print_function, absolute_import, division, unicode_literals
import logging
import copy

from . import ast

logger = logging.getLogger("pymola")

# TODO Flatten function vs. conversion classes


class TreeWalker(object):

    def walk(self, listener, tree):
        name = tree.__class__.__name__
        if hasattr(listener, 'enterEvery'):
            getattr(listener, 'enterEvery')(tree)
        if hasattr(listener, 'enter' + name):
            getattr(listener, 'enter' + name)(tree)
            #massive hack to make sure symbols get processesd first
        for child_name in reversed(sorted(tree.ast_spec.keys())):
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


def flatten_class(root, orig_class, instance_name):
    """
    This function takes and flattens it so that all subclasses instances
    are replaced by the their equations and symbols with name mangling
    of the instance name passed.
    :param root: The root of the tree that contains all class definitions
    :param orig_class: The class we want to flatten
    :param instance_name:
    :return: flat_class, the flattened class of type Class
    """

    CLASS_SEPARATOR = '.'

    # create the returned class
    flat_class = ast.Class(
        name=orig_class.name,
    )

    # append period to non empty instance_name
    if instance_name != '':
        instance_prefix = instance_name + CLASS_SEPARATOR
    else:
        instance_prefix = instance_name

    def modify_class(class_or_sym, modification):
        class_or_sym = copy.deepcopy(class_or_sym)
        for argument in modification.arguments:
            if isinstance(argument, ast.ElementModification):
                if argument.component.name in ast.Symbol.ATTRIBUTES:
                    setattr(class_or_sym, argument.component.name, argument.modifications[0])
                else:
                    sym = root.find_symbol(class_or_sym, argument.component)
                    for modification in argument.modifications:
                        if isinstance(modification, ast.ClassModification):
                            modify_class(sym, modification)
                        else:
                            sym.value = modification
            elif isinstance(argument, ast.ComponentClause):
                for new_sym in argument.symbol_list:
                    orig_sym = class_or_sym.symbols[sym.name]
                    orig_sym.__dict__.update(new_sym.__dict__)
            elif isinstance(argument, ast.ShortClassDefinition):
                for sym in class_or_sym.symbols.values():
                    if len(sym.type.child) == 0 and sym.type.name == argument.name:
                        sym.type = argument.component
                # TODO class modifications to short class definition
            else:
                raise Exception('Unsupported class modification argument {}'.format(argument))
        return class_or_sym

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
                        tree.name += CLASS_SEPARATOR + c.name
                    tree.child = []

            def exitComponentRef(self, tree):
                self.d -= 1

        w = TreeWalker()
        w.walk(ExpressionFlattener(instance_prefix), expression_copy)

        return expression_copy

    def pull_functions(expression, instance_prefix):
        expression_copy = copy.deepcopy(expression)

        class FunctionPuller(TreeListener):
            def __init__(self, instance_prefix, root, function_set):
                self.instance_prefix = instance_prefix
                self.root = root
                self.function_set = function_set

                super(FunctionPuller, self).__init__()

            def exitExpression(self, tree):
                if isinstance(tree.operator, ast.ComponentRef) and \
                                tree.operator.name in self.root.classes:
                    self.function_set.add(tree.operator.name)

        w = TreeWalker()
        function_set = set()
        w.walk(FunctionPuller(instance_prefix, root, function_set), expression_copy)

        return function_set

    # pull in parent classes
    for extends in orig_class.extends:
        c = root.find_class(extends.component)

        # carry out modifications
        c = modify_class(c, extends.class_modification)

        # recursively call flatten on the parent class
        flat_parent_class = flatten_class(root, c, instance_name)

        # set visibility
        for sym in flat_parent_class.symbols.values():
            sym.visibility = min(sym.visibility, extends.visibility)

        # add parent class members symbols, equations and statements
        flat_class.symbols.update(flat_parent_class.symbols)
        flat_class.equations += flat_parent_class.equations
        flat_class.statements += flat_parent_class.statements

    # for all symbols in the original class
    for sym_name, sym in orig_class.symbols.items():
        try:
            c = root.find_class(sym.type)
        except KeyError:
            # append original symbol to flat class
            flat_sym = flatten_symbol(sym, instance_prefix)
            flat_class.symbols[flat_sym.name] = flat_sym
        else:
            # the symbol type is a class
            if sym.class_modification is not None:
                c = modify_class(c, sym.class_modification)

            # recursively call flatten on the contained class
            flat_sub_class = flatten_class(root, c, instance_prefix + sym_name)

            # add sub_class members symbols and equations
            flat_class.symbols.update(flat_sub_class.symbols)
            flat_class.equations += flat_sub_class.equations
            flat_class.statements += flat_sub_class.statements

            # we keep connectors in the class hierarchy, as we may refer to them further
            # up using connect() clauses
            if c.type == 'connector':
                flat_sym = flatten_symbol(sym, instance_prefix)
                flat_class.symbols[flat_sym.name] = flat_sym

    # for all equations in original class
    flow_connections = {}
    for equation in orig_class.equations:
        flat_equation = flatten_expression(equation, instance_prefix)
        if isinstance(equation, ast.ConnectClause):
            # expand connector
            connect_equations = []

            sym_left = root.find_symbol(flat_class, flat_equation.left)
            sym_right = root.find_symbol(flat_class, flat_equation.right)

            try:
                class_left = root.find_class(sym_left.type)
                class_right = root.find_class(sym_right.type)
            except KeyError:
                logger.warning("Connector class {} or {} not defined.  Assuming it to be an elementary type.".format(sym_left.type, sym_right.type))

                connect_equation = ast.Equation(left=flat_equation.left, right=flat_equation.right)
                connect_equations.append(connect_equation)
            else:
                assert(class_left == class_right)

                flat_class_left = flatten_class(root, class_left, '')

                for connector_variable in flat_class_left.symbols.values():
                    left_name = flat_equation.left.name + CLASS_SEPARATOR + connector_variable.name
                    right_name = flat_equation.right.name + CLASS_SEPARATOR + connector_variable.name
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
        operands = [ast.ComponentRef(name=variable) for variable in sorted(connected_variables)]
        connect_equation = ast.Equation(left=ast.Expression(operator='+', operands=operands), right=ast.Primary(value=0))
        flat_class.equations += [connect_equation]

    # TODO: Also drag along any functions we need
    # function_set = set()
    # for eq in flat_class.equations + flat_class.statements:
    #     function_set |= pull_functions(eq, instance_prefix)

    # for f in function_set:
    #     if f not in flat_file.classes:
    #         flat_file.classes.update(flatten(root, f, instance_name).classes)

    return flat_class

def flatten(root, class_name):
    """
    This function takes and flattens it so that all subclasses instances
    are replaced by the their equations and symbols with name mangling
    of the instance name passed.
    :param root: The root of the tree that contains all class definitions
    :param class_name: The class we want to flatten
    :return: flat_file, a File containing the flattened class
    """

    # flatten class
    flat_class = flatten_class(root, root.classes[class_name], '')

    # strip connector symbols
    for i, sym in list(flat_class.symbols.items()):
        try:
            c = root.find_class(sym.type)
        except KeyError:
            pass
        else:
            del flat_class.symbols[i]

    # flat file
    flat_file = ast.File()
    flat_file.classes[class_name] = flat_class

    return flat_file
