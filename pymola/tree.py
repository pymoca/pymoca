#!/usr/bin/env python
"""
Tools for tree walking and visiting etc.
"""

from __future__ import print_function, absolute_import, division, unicode_literals

import copy
import logging
import sys
from collections import OrderedDict
from typing import Union

from . import ast

CLASS_SEPARATOR = '.'

logger = logging.getLogger("pymola")


# TODO Flatten function vs. conversion classes


# noinspection PyPep8Naming
class TreeListener(object):
    """
    Defines interface for tree listeners.
    """

    def __init__(self):
        self.context = {}

    def enterEvery(self, tree: ast.Node) -> None:
        self.context[type(tree).__name__] = tree

    def exitEvery(self, tree: ast.Node):
        self.context[type(tree).__name__] = None

    def enterFile(self, tree: ast.File) -> None:
        pass

    def exitFile(self, tree: ast.File) -> None:
        pass

    def enterClass(self, tree: ast.Class) -> None:
        pass

    def exitClass(self, tree: ast.Class) -> None:
        pass

    def enterImportAsClause(self, tree: ast.ImportAsClause) -> None:
        pass

    def exitImportAsClause(self, tree: ast.ImportAsClause) -> None:
        pass

    def enterImportFromClause(self, tree: ast.ImportFromClause) -> None:
        pass

    def exitImportFromClause(self, tree: ast.ImportFromClause) -> None:
        pass

    def enterElementModification(self, tree: ast.ElementModification) -> None:
        pass

    def exitElementModification(self, tree: ast.ElementModification) -> None:
        pass

    def enterClassModification(self, tree: ast.ClassModification) -> None:
        pass

    def exitClassModification(self, tree: ast.ClassModification) -> None:
        pass

    def enterExtendsClause(self, tree: ast.ExtendsClause) -> None:
        pass

    def exitExtendsClause(self, tree: ast.ExtendsClause) -> None:
        pass

    def enterIfExpression(self, tree: ast.IfExpression) -> None:
        pass

    def exitIfExpression(self, tree: ast.IfExpression) -> None:
        pass

    def enterExpression(self, tree: ast.Expression) -> None:
        pass

    def exitExpression(self, tree: ast.Expression) -> None:
        pass

    def enterIfEquation(self, tree: ast.IfEquation) -> None:
        pass

    def exitIfEquation(self, tree: ast.IfEquation) -> None:
        pass

    def enterForIndex(self, tree: ast.ForIndex) -> None:
        pass

    def exitForIndex(self, tree: ast.ForIndex) -> None:
        pass

    def enterForEquation(self, tree: ast.ForEquation) -> None:
        pass

    def exitForEquation(self, tree: ast.ForEquation) -> None:
        pass

    def enterEquation(self, tree: ast.Equation) -> None:
        pass

    def exitEquation(self, tree: ast.Equation) -> None:
        pass

    def enterConnectClause(self, tree: ast.ConnectClause) -> None:
        pass

    def exitConnectClause(self, tree: ast.ConnectClause) -> None:
        pass

    def enterSymbol(self, tree: ast.Symbol) -> None:
        pass

    def exitSymbol(self, tree: ast.Symbol) -> None:
        pass

    def enterComponentClause(self, tree: ast.ComponentClause) -> None:
        pass

    def exitComponentClause(self, tree: ast.ComponentClause) -> None:
        pass

    def enterArray(self, tree: ast.Array) -> None:
        pass

    def exitArray(self, tree: ast.Array) -> None:
        pass

    def enterSlice(self, tree: ast.Slice) -> None:
        pass

    def exitSlice(self, tree: ast.Slice) -> None:
        pass

    def enterPrimary(self, tree: ast.Primary) -> None:
        pass

    def exitPrimary(self, tree: ast.Primary) -> None:
        pass

    def enterComponentRef(self, tree: ast.ComponentRef) -> None:
        pass

    def exitComponentRef(self, tree: ast.ComponentRef) -> None:
        pass


class TreeWalker(object):
    """
    Defines methods for tree walker. Inherit from this to make your own.
    """

    def walk(self, listener: TreeListener, tree: ast.Node) -> None:
        """
        Walks an AST tree recursively
        :param listener: 
        :param tree: 
        :return: None
        """
        name = tree.__class__.__name__
        if hasattr(listener, 'enterEvery'):
            getattr(listener, 'enterEvery')(tree)
        if hasattr(listener, 'enter' + name):
            getattr(listener, 'enter' + name)(tree)
        for child_name in tree.__dict__.keys():
            self.handle_walk(listener, tree.__dict__[child_name])
        if hasattr(listener, 'exitEvery'):
            getattr(listener, 'exitEvery')(tree)
        if hasattr(listener, 'exit' + name):
            getattr(listener, 'exit' + name)(tree)

    def handle_walk(self, listener: TreeListener, tree: Union[ast.Node, dict, list]) -> None:
        """
        Handles tree walking, has to account for dictionaries and lists
        :param listener: listener that reacts to walked events
        :param tree: the tree to walk
        :return: None
        """
        if isinstance(tree, ast.Node):
            self.walk(listener, tree)
        elif isinstance(tree, dict):
            for k in tree.keys():
                self.handle_walk(listener, tree[k])
        elif isinstance(tree, list):
            for i in range(len(tree)):
                self.handle_walk(listener, tree[i])
        else:
            pass


def flatten_class(root: ast.Collection, orig_class: ast.Class, instance_name: str,
                  class_modification: ast.ClassModification = None) -> ast.Class:
    """
    This function takes and flattens it so that all subclasses instances
    are replaced by the their equations and symbols with name mangling
    of the instance name passed.
    :param root: The root of the tree that contains all class definitions
    :param orig_class: The class we want to flatten
    :param instance_name: 
    :param class_modification: 
    :return: flat_class, the flattened class of type Class
    """

    # create the returned class
    flat_class = ast.Class(
        name=orig_class.name,
    )

    # append period to non empty instance_name
    if instance_name != '':
        instance_prefix = instance_name + CLASS_SEPARATOR
    else:
        instance_prefix = instance_name

    extended_orig_class = ast.Class(
        name=orig_class.name,
    )

    for extends in orig_class.extends:
        c = root.find_class(extends.component, orig_class.within)

        # recursively call flatten on the parent class
        # NOTE: We do not to pass the instance name along. The symbol renaming
        # is handled at the current level, not at the level of the base class.
        # That way we can properly apply class modifications to inherited
        # symbols.
        flat_parent_class = flatten_class(root, c, '')

        # set visibility
        for sym in flat_parent_class.symbols.values():
            if sym.visibility > extends.visibility:
                sym.visibility = extends.visibility

        # add parent class members symbols, equations and statements
        extended_orig_class.symbols.update(flat_parent_class.symbols)
        extended_orig_class.equations += flat_parent_class.equations
        extended_orig_class.statements += flat_parent_class.statements

        # carry out modifications
        extended_orig_class = modify_class(root, extended_orig_class, extends.class_modification)

    extended_orig_class.symbols.update(orig_class.symbols)
    extended_orig_class.equations += orig_class.equations
    extended_orig_class.statements += orig_class.statements

    if class_modification is not None:
        extended_orig_class = modify_class(root, extended_orig_class, class_modification)

    # for all symbols in the original class
    for sym_name, sym in extended_orig_class.symbols.items():
        flat_sym = flatten_symbol(sym, instance_prefix)
        try:
            c = root.find_class(flat_sym.type)
        except KeyError:
            # append original symbol to flat class
            flat_class.symbols[flat_sym.name] = flat_sym
        else:
            # recursively call flatten on the contained class
            flat_sub_class = flatten_class(root, c, flat_sym.name, flat_sym.class_modification)

            # carry class dimensions over to symbols
            for flat_class_symbol in flat_sub_class.symbols.values():
                if len(flat_class_symbol.dimensions) == 1 \
                        and isinstance(flat_class_symbol.dimensions[0], ast.Primary) \
                        and flat_class_symbol.dimensions[0].value == 1:
                    flat_class_symbol.dimensions = flat_sym.dimensions
                elif len(flat_sym.dimensions) == 1 and isinstance(flat_sym.dimensions[0], ast.Primary) \
                        and flat_sym.dimensions[0].value == 1:
                    flat_class_symbol.dimensions = flat_class_symbol.dimensions
                else:
                    flat_class_symbol.dimensions = flat_sym.dimensions + flat_class_symbol.dimensions

            # add sub_class members symbols and equations
            flat_class.symbols.update(flat_sub_class.symbols)
            flat_class.equations += flat_sub_class.equations
            flat_class.statements += flat_sub_class.statements

            # we keep connectors in the class hierarchy, as we may refer to them further
            # up using connect() clauses
            if c.type == 'connector':
                flat_class.symbols[flat_sym.name] = flat_sym

    # now resolve all references inside the symbol definitions
    for sym_name, sym in flat_class.symbols.items():
        flat_sym = flatten_component_refs(root, flat_class, sym, instance_prefix)
        flat_class.symbols[sym_name] = flat_sym

    # for all equations in original class
    flow_connections = OrderedDict()
    for equation in extended_orig_class.equations:
        flat_equation = flatten_component_refs(root, flat_class, equation, instance_prefix)
        if isinstance(equation, ast.ConnectClause):
            # expand connector
            connect_equations = []

            sym_left = root.find_symbol(flat_class, flat_equation.left)
            sym_right = root.find_symbol(flat_class, flat_equation.right)

            try:
                class_left = root.find_class(sym_left.type)
                # noinspection PyUnusedLocal
                class_right = root.find_class(sym_right.type)
            except KeyError:
                primary_types = ['Real']
                if sym_left.type.name not in primary_types or sym_right.type.name not in primary_types:
                    logger.warning("Connector class {} or {} not defined.  "
                                   "Assuming it to be an elementary type.".format(sym_left.type, sym_right.type))
                connect_equation = ast.Equation(left=flat_equation.left, right=flat_equation.right)
                connect_equations.append(connect_equation)
            else:
                # TODO: Add check about matching inputs and outputs

                flat_class_left = flatten_class(root, class_left, '')

                for connector_variable in flat_class_left.symbols.values():
                    left_name = flat_equation.left.name + CLASS_SEPARATOR + connector_variable.name
                    right_name = flat_equation.right.name + CLASS_SEPARATOR + connector_variable.name
                    left = ast.ComponentRef(name=left_name, indices=flat_equation.left.indices)
                    right = ast.ComponentRef(name=right_name, indices=flat_equation.right.indices)
                    if len(connector_variable.prefixes) == 0 or connector_variable.prefixes[0] in ['input', 'output']:
                        connect_equation = ast.Equation(left=left, right=right)
                        connect_equations.append(connect_equation)
                    elif connector_variable.prefixes == ['flow']:
                        left_repr = repr(left)
                        right_repr = repr(right)

                        left_connected_variables = flow_connections.get(left_repr, OrderedDict())
                        right_connected_variables = flow_connections.get(right_repr, OrderedDict())

                        left_connected_variables.update(right_connected_variables)
                        connected_variables = left_connected_variables
                        connected_variables[left_repr] = left
                        connected_variables[right_repr] = right

                        for connected_variable in connected_variables:
                            flow_connections[connected_variable] = connected_variables
                    else:
                        raise Exception(
                            "Unsupported connector variable prefixes {}".format(connector_variable.prefixes))

            flat_class.equations += connect_equations
        else:
            # flatten equation
            flat_class.equations += [flat_equation]

    flat_class.statements += [flatten_component_refs(root, flat_class, e, instance_prefix) for e in
                              extended_orig_class.statements]

    # add flow equations
    if len(flow_connections) > 0:
        # TODO Flatten first
        logger.warning(
            "Note: Connections between connectors with flow variables "
            "are not supported across levels of the class hierarchy")

    processed = []  # OrderedDict is not hashable, so we cannot use sets.
    for connected_variables in flow_connections.values():
        if connected_variables not in processed:
            operands = list(connected_variables.values())
            expr = ast.Expression(operator='+', operands=operands[-2:])
            for op in reversed(operands[:-2]):
                expr = ast.Expression(operator='+', operands=[op, expr])
            connect_equation = ast.Equation(left=expr, right=ast.Primary(value=0))
            flat_class.equations += [connect_equation]
            processed.append(connected_variables)

    # TODO: Also drag along any functions we need
    # function_set = set()
    # for eq in flat_class.equations + flat_class.statements:
    #     function_set |= pull_functions(eq, instance_prefix)

    # for f in function_set:
    #     if f not in flat_file.classes:
    #         flat_file.classes.update(flatten(root, f, instance_name).classes)
    return flat_class


def modify_class(root: ast.Collection, class_or_sym: Union[ast.Class, ast.Symbol], modification):
    """
    Apply a modification to a class or symbol.
    :param root: root tree for looking up symbols
    :param class_or_sym: class or symbol to modify
    :param modification: modification to apply
    :return: 
    """
    class_or_sym = copy.deepcopy(class_or_sym)
    for argument in modification.arguments:
        if isinstance(argument, ast.ElementModification):
            if argument.component.name in ast.Symbol.ATTRIBUTES:
                setattr(class_or_sym, argument.component.name, argument.modifications[0])
            else:
                s = root.find_symbol(class_or_sym, argument.component)
                for modification in argument.modifications:
                    if isinstance(modification, ast.ClassModification):
                        s.__dict__.update(modify_class(root, s, modification).__dict__)
                    else:
                        s.value = modification
        elif isinstance(argument, ast.ComponentClause):
            for new_sym in argument.symbol_list:
                orig_sym = class_or_sym.symbols[new_sym.name]
                orig_sym.__dict__.update(new_sym.__dict__)
        elif isinstance(argument, ast.ShortClassDefinition):
            for s in class_or_sym.symbols.values():
                if len(s.type.child) == 0 and s.type.name == argument.name:
                    s.type = argument.component
                    # TODO class modifications to short class definition
        else:
            raise Exception('Unsupported class modification argument {}'.format(argument))
    return class_or_sym


def flatten_symbol(s: ast.Symbol, instance_prefix: str) -> ast.Symbol:
    """
    Given a symbols and a prefix performs name mangling
    :param s: Symbol
    :param instance_prefix: Prefix for instance
    :return: flattened symbol
    """
    s_copy = copy.deepcopy(s)
    s_copy.name = instance_prefix + s.name
    if len(instance_prefix) > 0:
        # Strip 'input' and 'output' prefixes from nested symbols.
        strip_keywords = ['input', 'output']
        for strip_keyword in strip_keywords:
            try:
                s_copy.prefixes.remove(strip_keyword)
            except ValueError:
                pass
    return s_copy


class ComponentRefFlattener(TreeListener):
    """
    A listener that flattens references to components and performs name mangling,
    it also locates all symbols and determines which are states (
    one of the equations contains a derivative of the symbol)
    """

    def __init__(self, root: ast.Collection, container: ast.Class, instance_prefix: str):
        self.root = root
        self.container = container
        self.instance_prefix = instance_prefix
        self.depth = 0
        self.cutoff_depth = sys.maxsize
        super().__init__()

    def enterComponentRef(self, tree: ast.ComponentRef):
        self.depth += 1
        if self.depth > self.cutoff_depth:
            return

        # Compose flatted name
        new_name = self.instance_prefix + tree.name
        c = tree
        while len(c.child) > 0:
            c = c.child[0]
            new_name += CLASS_SEPARATOR + c.name

        # If the flattened name exists in the container, use it.
        # Otherwise, skip this reference.
        try:
            self.root.find_symbol(self.container, ast.ComponentRef(name=new_name))
        except KeyError:
            # The component was not found in the container.  We leave this
            # reference alone.
            self.cutoff_depth = self.depth
        else:
            tree.name = new_name
            c = tree
            while len(c.child) > 0:
                c = c.child[0]
                if len(c.indices) > 0:
                    tree.indices += c.indices
            tree.child = []

    def exitComponentRef(self, tree: ast.ComponentRef):
        self.depth -= 1
        if self.depth < self.cutoff_depth:
            self.cutoff_depth = sys.maxsize


def flatten_component_refs(
        root: ast.Collection, container: ast.Class,
        expression: ast.Union[ast.ConnectClause, ast.AssignmentStatement, ast.ForStatement, ast.Symbol],
        instance_prefix: str) -> ast.Union[ast.ConnectClause, ast.AssignmentStatement, ast.ForStatement, ast.Symbol]:
    """
    Flattens component refs in a tree
    :param root: root node
    :param container: class
    :param expression: original expression
    :param instance_prefix: prefix for instance
    :return: flattened expression
    """

    expression_copy = copy.deepcopy(expression)

    w = TreeWalker()
    w.walk(ComponentRefFlattener(root, container, instance_prefix), expression_copy)

    return expression_copy


class StateAnnotator(TreeListener):
    """
    This finds all variables that are differentiated and annotates them with the state prefix
    """

    def __init__(self, root: ast.Collection, node: ast.Node):
        self.root = root
        self.node = node
        super().__init__()

    def exitExpression(self, tree: ast.Expression):
        """
        When exiting an expression, check if it is a derivative, if it is
        put state prefix on symbol
        """
        if tree.operator == 'der':
            s = self.root.find_symbol(self.node, tree.operands[0])
            if 'state' not in s.prefixes:
                s.prefixes.append('state')


def annotate_states(root: ast.Collection, node: ast.Node) -> None:
    """
    TODO: document
    :param root: collection for performing symbol lookup etc.
    :param node: node of tree to walk
    :return: 
    """
    w = TreeWalker()
    w.walk(StateAnnotator(root, node), node)


class FunctionPuller(TreeListener):
    """
    Listener to extract functions
    """

    def __init__(self, instance_prefix: str, root, function_set):
        self.instance_prefix = instance_prefix
        self.root = root
        self.function_set = function_set
        super().__init__()

    def exitExpression(self, tree: ast.Expression):
        if isinstance(tree.operator, ast.ComponentRef) and \
                        tree.operator.name in self.root.classes:
            self.function_set.add(tree.operator.name)


# noinspection PyUnusedLocal
def pull_functions(root: ast.Collection, expression: ast.Expression, instance_prefix: str) -> set:
    """
    TODO: document
    :param root: collection for performing symbol lookup etc.
    :param expression: 
    :param instance_prefix: 
    :return: 
    """

    expression_copy = copy.deepcopy(expression)

    w = TreeWalker()
    function_set = set()
    w.walk(FunctionPuller(instance_prefix, root, function_set), expression_copy)
    return function_set


def flatten(root: ast.Collection, class_name: str) -> ast.File:
    """
    This function takes and flattens it so that all subclasses instances
    are replaced by the their equations and symbols with name mangling
    of the instance name passed.
    :param root: The root of the tree that contains all files with all class definitions
    :param class_name: The class we want to flatten
    :return: flat_file, a File containing the flattened class
    """

    # The within information is needed at the class level when extending
    for f in root.files:
        for c in f.classes.values():
            c.within = f.within

    # flatten class
    flat_class = flatten_class(root, root.find_class(class_name), '')

    # strip connector symbols
    for i, sym in list(flat_class.symbols.items()):
        try:
            # noinspection PyUnusedLocal
            c = root.find_class(sym.type)
        except KeyError:
            pass
        else:
            del flat_class.symbols[i]

    # annotate states
    annotate_states(root, flat_class)

    # flat file
    flat_file = ast.File()
    flat_file.classes[class_name] = flat_class

    return flat_file
