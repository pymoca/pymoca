#!/usr/bin/env python
"""
Tools for tree walking and visiting etc.
"""

from __future__ import print_function, absolute_import, division, unicode_literals

import copy  # TODO
import logging
import sys
from collections import OrderedDict
from typing import Union
import os

import numpy as np

from . import ast

CLASS_SEPARATOR = '.'

logger = logging.getLogger("pymoca")


# TODO Flatten function vs. conversion classes
class ModificationTargetNotFound(Exception):
    pass


# noinspection PyPep8Naming
class TreeListener:
    """
    Defines interface for tree listeners.
    """

    def __init__(self):
        self.context = {}

    def enterEvery(self, tree: ast.Node) -> None:
        self.context[type(tree).__name__] = tree

    def exitEvery(self, tree: ast.Node):
        self.context[type(tree).__name__] = None

    # -------------------------------------------------------------------------
    # enter ast listeners (sorted alphabetically)
    # -------------------------------------------------------------------------

    def enterArray(self, tree: ast.Array) -> None: pass

    def enterAssignmentStatement(self, tree: ast.AssignmentStatement) -> None: pass

    def enterClass(self, tree: ast.Class) -> None: pass

    def enterClassModification(self, tree: ast.ClassModification) -> None: pass

    def enterComponentClause(self, tree: ast.ComponentClause) -> None: pass

    def enterComponentRef(self, tree: ast.ComponentRef) -> None: pass

    def enterConnectClause(self, tree: ast.ConnectClause) -> None: pass

    def enterElementModification(self, tree: ast.ElementModification) -> None: pass

    def enterEquation(self, tree: ast.Equation) -> None: pass

    def enterExpression(self, tree: ast.Expression) -> None: pass

    def enterExtendsClause(self, tree: ast.ExtendsClause) -> None: pass

    def enterForEquation(self, tree: ast.ForEquation) -> None: pass

    def enterForIndex(self, tree: ast.ForIndex) -> None: pass

    def enterForStatement(self, tree: ast.ForStatement) -> None: pass

    def enterFunction(self, tree: ast.Function) -> None: pass

    def enterIfEquation(self, tree: ast.IfEquation) -> None: pass

    def enterIfExpression(self, tree: ast.IfExpression) -> None: pass

    def enterIfStatement(self, tree: ast.IfStatement) -> None: pass

    def enterImportAsClause(self, tree: ast.ImportAsClause) -> None: pass

    def enterImportFromClause(self, tree: ast.ImportFromClause) -> None: pass

    def enterPrimary(self, tree: ast.Primary) -> None: pass

    def enterSlice(self, tree: ast.Slice) -> None: pass

    def enterSymbol(self, tree: ast.Symbol) -> None: pass

    def enterTree(self, tree: ast.Tree) -> None: pass

    def enterWhenEquation(self, tree: ast.WhenEquation) -> None: pass

    def enterWhenStatement(self, tree: ast.WhenStatement) -> None: pass

    # -------------------------------------------------------------------------
    # exit ast listeners (sorted alphabetically)
    # -------------------------------------------------------------------------

    def exitArray(self, tree: ast.Array) -> None: pass

    def exitAssignmentStatement(self, tree: ast.AssignmentStatement) -> None: pass

    def exitClass(self, tree: ast.Class) -> None: pass

    def exitClassModification(self, tree: ast.ClassModification) -> None: pass

    def exitComponentClause(self, tree: ast.ComponentClause) -> None: pass

    def exitComponentRef(self, tree: ast.ComponentRef) -> None: pass

    def exitConnectClause(self, tree: ast.ConnectClause) -> None: pass

    def exitElementModification(self, tree: ast.ElementModification) -> None: pass

    def exitEquation(self, tree: ast.Equation) -> None: pass

    def exitExpression(self, tree: ast.Expression) -> None: pass

    def exitExtendsClause(self, tree: ast.ExtendsClause) -> None: pass

    def exitForEquation(self, tree: ast.ForEquation) -> None: pass

    def exitForIndex(self, tree: ast.ForIndex) -> None: pass

    def exitForStatement(self, tree: ast.ForStatement) -> None: pass

    def exitFunction(self, tree: ast.Function) -> None: pass

    def exitIfEquation(self, tree: ast.IfEquation) -> None: pass

    def exitIfExpression(self, tree: ast.IfExpression) -> None: pass

    def exitIfStatement(self, tree: ast.IfStatement) -> None: pass

    def exitImportAsClause(self, tree: ast.ImportAsClause) -> None: pass

    def exitImportFromClause(self, tree: ast.ImportFromClause) -> None: pass

    def exitPrimary(self, tree: ast.Primary) -> None: pass

    def exitSlice(self, tree: ast.Slice) -> None: pass

    def exitSymbol(self, tree: ast.Symbol) -> None: pass

    def exitTree(self, tree: ast.Tree) -> None: pass

    def exitWhenEquation(self, tree: ast.WhenEquation) -> None: pass

    def exitWhenStatement(self, tree: ast.WhenStatement) -> None: pass


class TreeWalker:
    """
    Defines methods for tree walker. Inherit from this to make your own.
    """

    def skip_child(self, tree: ast.Node, child_name: str) -> bool:
        """
        Skip certain childs in the tree walking. By default it prevents
        endless recursion by skipping references to e.g. parent nodes.
        :return: True if child needs to be skipped, False otherwise.
        """
        if isinstance(tree, ast.Class) and child_name == 'parent' or \
                isinstance(tree, ast.ClassModificationArgument) and child_name in ('scope', '__deepcopy__'):
            return True
        return False

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
            if self.skip_child(tree, child_name):
                continue
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


def flatten_extends(orig_class: Union[ast.Class, ast.InstanceClass], modification_environment=None,
                    parent=None) -> ast.InstanceClass:
    extended_orig_class = ast.InstanceClass(
        name=orig_class.name,
        type=orig_class.type,
        annotation=ast.ClassModification(),
        parent=parent
    )

    if isinstance(orig_class, ast.InstanceClass):
        extended_orig_class.modification_environment = orig_class.modification_environment

    for extends in orig_class.extends:
        c = orig_class.find_class(extends.component, check_builtin_classes=True)

        if c.type == "__builtin":
            if len(orig_class.extends) > 1:
                raise Exception(
                    "When extending a built-in class (Real, Integer, ...) you cannot extend other classes as well")
            extended_orig_class.type = c.type

        c = flatten_extends(c, extends.class_modification, parent=c.parent)

        extended_orig_class.classes.update(c.classes)
        extended_orig_class.symbols.update(c.symbols)
        extended_orig_class.equations += c.equations
        extended_orig_class.initial_equations += c.initial_equations
        extended_orig_class.statements += c.statements
        extended_orig_class.initial_statements += c.initial_statements
        if isinstance(c.annotation, ast.ClassModification):
            extended_orig_class.annotation.arguments += c.annotation.arguments
        extended_orig_class.functions.update(c.functions)

        # Note that all extends clauses are handled before any modifications
        # are applied.
        extended_orig_class.modification_environment.arguments.extend(c.modification_environment.arguments)

        # set visibility
        for sym in extended_orig_class.symbols.values():
            if sym.visibility > extends.visibility:
                sym.visibility = extends.visibility

    extended_orig_class.classes.update(orig_class.classes)
    extended_orig_class.symbols.update(orig_class.symbols)
    extended_orig_class.equations += orig_class.equations
    extended_orig_class.initial_equations += orig_class.initial_equations
    extended_orig_class.statements += orig_class.statements
    extended_orig_class.initial_statements += orig_class.initial_statements
    if isinstance(orig_class.annotation, ast.ClassModification):
        extended_orig_class.annotation.arguments += orig_class.annotation.arguments
    extended_orig_class.functions.update(orig_class.functions)

    if modification_environment is not None:
        extended_orig_class.modification_environment.arguments.extend(modification_environment.arguments)

    # If the current class is inheriting an elementary type, we shift modifications from the class to its __value symbol
    if extended_orig_class.type == "__builtin":
        if extended_orig_class.symbols['__value'].class_modification is not None:
            extended_orig_class.symbols['__value'].class_modification.arguments.extend(
                extended_orig_class.modification_environment.arguments)
        else:
            extended_orig_class.symbols['__value'].class_modification = extended_orig_class.modification_environment

        extended_orig_class.modification_environment = ast.ClassModification()

    return extended_orig_class


def extends_builtin(class_: ast.Class) -> bool:
    ret = False
    for extends in class_.extends:
        try:
            c = class_.find_class(extends.component)
            ret |= extends_builtin(c)
        except ast.FoundElementaryClassError:
            return True
    return ret


def build_instance_tree(orig_class: Union[ast.Class, ast.InstanceClass], modification_environment=None,
                        parent=None) -> ast.InstanceClass:
    extended_orig_class = flatten_extends(orig_class, modification_environment, parent)

    # Redeclarations take effect
    for class_mod_argument in extended_orig_class.modification_environment.arguments:
        if not class_mod_argument.redeclare:
            continue
        scope_class = class_mod_argument.scope if class_mod_argument.scope is not None else extended_orig_class
        argument = class_mod_argument.value
        if isinstance(argument, ast.ShortClassDefinition):
            extended_orig_class.classes[argument.name] = scope_class.find_class(argument.component)
        elif isinstance(argument, ast.ComponentClause):
            # Redeclaration of symbols
            # TODO: Do we need to handle scoping of redeclarations of symbols?
            for s in argument.symbol_list:
                extended_orig_class.symbols[s.name].type = s.type
        else:
            raise Exception("Unknown redeclaration type")

    extended_orig_class.modification_environment.arguments = [
        x for x in extended_orig_class.modification_environment.arguments if not x.redeclare]

    # Only ast.ElementModification type modifications left in the class's
    # modification environment. No more ComponentClause or
    # ShortClassDefinitions (which are both redeclares). There are still
    # possible redeclares in symbols though.

    # TODO: Remove redundancy in code below. Filtering of arguments and
    # shifting of modifications is fairly similar for both classes, elementary
    # symbols, and non-elementary symbols.

    # Merge/pass along modifications for classes
    for class_name, c in extended_orig_class.classes.items():
        sub_class_modification = ast.ClassModification()

        sub_class_arguments = [x for x in extended_orig_class.modification_environment.arguments
                               if isinstance(x.value, ast.ElementModification) and x.value.component.name == class_name]

        # Remove from current class's modification environment
        extended_orig_class.modification_environment.arguments = [
            x for x in extended_orig_class.modification_environment.arguments if x not in sub_class_arguments]

        for main_mod in sub_class_arguments:
            for elem_class_mod in main_mod.value.modifications:
                for arg in elem_class_mod.arguments:
                    sub_class_modification.arguments.append(arg)

        extended_orig_class.classes[class_name] = build_instance_tree(c, sub_class_modification, extended_orig_class)

    # Check that all symbol modifications to be applied on this class exist
    for arg in extended_orig_class.modification_environment.arguments:
        if not arg.value.component.name in extended_orig_class.symbols:
            raise ModificationTargetNotFound("Trying to modify symbol {}, which does not exist in class {}".format(
                arg.value.component.name,
                extended_orig_class.full_reference()
            ))

    # Merge/pass along modifications for symbols, including redeclares
    for sym_name, sym in extended_orig_class.symbols.items():
        class_name = sym.type

        try:
            if not isinstance(sym.type, ast.InstanceClass):
                c = extended_orig_class.find_class(sym.type)
            else:
                c = sym.type
        except ast.FoundElementaryClassError:
            # Symbol is elementary type. Check if we need to move any modifications to the symbol.
            sym_arguments = [x for x in extended_orig_class.modification_environment.arguments
                             if isinstance(x.value, ast.ElementModification) and x.value.component.name == sym_name]

            # Remove from current class's modification environment
            extended_orig_class.modification_environment.arguments = [
                x for x in extended_orig_class.modification_environment.arguments if x not in sym_arguments]

            sym_mod = ast.ClassModification()

            for arg in sym_arguments:
                if arg.value.component.indices != [[None]]:
                    raise Exception("Subscripting modifiers is not allowed.")
                for el_arg in arg.value.modifications:
                    # Behavior is different depending on whether the value is
                    # being set (which is an unnamed field not explicitly
                    # referred to), or a named attribute (e.g. nominal, min,
                    # etc).

                    if not isinstance(el_arg, ast.ClassModification):
                        # If the value is being set, we make a new class
                        # modification with attribute name "value" that we
                        # pick up later in modify_symbol()

                        # TODO: Figure out if it's easier to directly do this
                        # in the parser.
                        vmod_arg = ast.ClassModificationArgument()
                        vmod_arg.scope = arg.scope
                        vmod_arg.value = ast.ElementModification()
                        vmod_arg.value.component = ast.ComponentRef(name="value")
                        vmod_arg.value.modifications = [el_arg]
                        sym_mod.arguments.append(vmod_arg)
                    else:
                        sym_mod.arguments.extend(el_arg.arguments)

            if sym.class_modification:
                sym.class_modification.arguments.extend(sym_mod.arguments)
            else:
                sym.class_modification = sym_mod
        else:
            # Symbol is not elementary type. Check if we need to move any modifications along.
            sym_arguments = [x for x in extended_orig_class.modification_environment.arguments
                             if isinstance(x.value, ast.ElementModification) and x.value.component.name == sym_name]

            # Remove from current class's modification environment
            extended_orig_class.modification_environment.arguments = [
                x for x in extended_orig_class.modification_environment.arguments if x not in sym_arguments]

            # Fix component references to be one level deeper. E.g. applying a
            # modification "a.x = 3.0" on a symbol "a", will mean we pass
            # along a modification "x = 3.0" to the symbol's class instance.
            # We should only do this if we are not (in)directly inheriting from a builtin.
            inheriting_from_builtin = extends_builtin(c)
            sym_mod = ast.ClassModification()
            for arg in sym_arguments:
                if arg.value.component.indices != [[None]]:
                    raise Exception("Subscripting modifiers is not allowed.")

                if inheriting_from_builtin:
                    for el_arg in arg.value.modifications:
                        if not isinstance(el_arg, ast.ClassModification):
                            # If the value is being set, we make a new class
                            # modification with attribute name "value" that we
                            # pick up later in modify_symbol()

                            # TODO: Figure out if it's easier to directly do this
                            # in the parser.
                            vmod_arg = ast.ClassModificationArgument()
                            vmod_arg.scope = arg.scope
                            vmod_arg.value = ast.ElementModification()
                            vmod_arg.value.component = ast.ComponentRef(name="value")
                            vmod_arg.value.modifications = [el_arg]
                            sym_mod.arguments.append(vmod_arg)
                        else:
                            sym_mod.arguments.extend(el_arg.arguments)
                else:
                    arg.value.component = arg.value.component.child[0]
                    sym_mod.arguments.append(arg)

            if sym.class_modification:
                sym.class_modification.arguments.extend(sym_mod.arguments)
            else:
                sym.class_modification = sym_mod

            # Set the correct scope, e.g. for redeclaration modifications
            for arg in sym.class_modification.arguments:
                if arg.scope is None:
                    arg.scope = extended_orig_class

            try:
                sym.type = build_instance_tree(c, sym.class_modification, c.parent)
            except Exception as e:
                error_sym = str(orig_class.full_reference()) + "." + sym_name
                raise type(e)('Processing failed for symbol "{}"'.format(error_sym)) from e

            sym.class_modification = None

    return extended_orig_class


def flatten_symbols(class_: ast.InstanceClass, instance_name='') -> ast.Class:
    # Recursive symbol flattening

    flat_class = ast.Class(
        name=class_.name,
        type=class_.type,
        annotation=class_.annotation,
    )

    # append period to non empty instance_name
    if instance_name != '':
        instance_prefix = instance_name + CLASS_SEPARATOR
    else:
        instance_prefix = instance_name

    # for all symbols in the original class
    for sym_name, sym in class_.symbols.items():

        sym.name = instance_prefix + sym_name
        if instance_prefix:
            # Strip 'input' and 'output' prefixes from nested symbols.
            strip_keywords = ['input', 'output']
            for strip_keyword in strip_keywords:
                try:
                    sym.prefixes.remove(strip_keyword)
                except ValueError:
                    pass

        flat_sym = sym

        if isinstance(sym.type, ast.ComponentRef):
            # Elementary type
            flat_sym.dimensions = [flat_sym.dimensions]
            flat_class.symbols[flat_sym.name] = flat_sym
        elif sym.type.type == "__builtin":
            # Class inherited from elementary type (e.g. "type Voltage =
            # Real"). No flattening to be done, just copying over all
            # attributes and modifications to the class's "__value" symbol.
            flat_sym.dimensions = [flat_sym.dimensions]
            flat_class.symbols[flat_sym.name] = flat_sym

            if flat_sym.class_modification is not None:
                flat_sym.class_modification.arguments.extend(sym.type.symbols['__value'].class_modification.arguments)
            else:
                flat_sym.class_modification = sym.type.symbols['__value'].class_modification

            for att in flat_sym.ATTRIBUTES + ["type"]:
                setattr(flat_class.symbols[flat_sym.name], att, getattr(sym.type.symbols['__value'], att))

            continue
        else:
            # recursively call flatten on the contained class
            flat_sub_class = flatten_symbols(sym.type, flat_sym.name)

            # carry class dimensions over to symbols
            for flat_class_symbol in flat_sub_class.symbols.values():
                flat_class_symbol.dimensions = [flat_sym.dimensions] + flat_class_symbol.dimensions

            # add sub_class members symbols and equations
            flat_class.classes.update(flat_sub_class.classes)
            flat_class.symbols.update(flat_sub_class.symbols)
            flat_class.equations += flat_sub_class.equations
            flat_class.initial_equations += flat_sub_class.initial_equations
            flat_class.statements += flat_sub_class.statements
            flat_class.initial_statements += flat_sub_class.initial_statements
            flat_class.functions.update(flat_sub_class.functions)

            # we keep connectors in the class hierarchy, as we may refer to them further
            # up using connect() clauses
            if sym.type.type == 'connector':
                # We flatten sym.type here to avoid later deepcopy()
                # statements copying the entire instance tree due to
                # references to parent and/or root.
                flat_sym.__connector_type = flatten_class(sym.type)
                flat_class.symbols[flat_sym.name] = flat_sym

                # TODO: Do we need the symbol type after this?
                sym.type = sym.type.name

    # Apply any symbol modifications if the scope of said modification is equal to that of the current class
    apply_symbol_modifications(flat_class, class_)

    # now resolve all references inside the symbol definitions
    for sym_name, sym in flat_class.symbols.items():
        flat_sym = flatten_component_refs(flat_class, sym, instance_prefix)
        flat_class.symbols[sym_name] = flat_sym

    # A set of component refs to functions
    pulled_functions = OrderedDict()

    # for all equations in original class
    for equation in class_.equations:
        # Equation returned has function calls replaced with their full scope
        # equivalent, and it pulls out all references into the pulled_functions.
        fs_equation = fully_scope_function_calls(class_, equation, pulled_functions)

        flat_equation = flatten_component_refs(flat_class, fs_equation, instance_prefix)
        flat_class.equations.append(flat_equation)
        if isinstance(flat_equation, ast.ConnectClause):
            # following section 9.2 of the Modelica spec, we treat 'inner' and 'outer' connectors differently.
            if not hasattr(flat_equation, '__left_inner'):
                flat_equation.__left_inner = len(equation.left.child) > 0
            if not hasattr(flat_equation, '__right_inner'):
                flat_equation.__right_inner = len(equation.right.child) > 0

    # Create fully scoped equivalents
    fs_initial_equations = \
        [fully_scope_function_calls(class_, e, pulled_functions) for e in class_.initial_equations]
    fs_statements = \
        [fully_scope_function_calls(class_, e, pulled_functions) for e in class_.statements]
    fs_initial_statements = \
        [fully_scope_function_calls(class_, e, pulled_functions) for e in class_.initial_statements]

    flat_class.initial_equations += \
        [flatten_component_refs(flat_class, e, instance_prefix) for e in fs_initial_equations]
    flat_class.statements += \
        [flatten_component_refs(flat_class, e, instance_prefix) for e in fs_statements]
    flat_class.initial_statements += \
        [flatten_component_refs(flat_class, e, instance_prefix) for e in fs_initial_statements]

    for f, c in pulled_functions.items():
        pulled_functions[f] = flatten_class(c)
        c = pulled_functions[f]
        flat_class.functions.update(c.functions)
        c.functions = OrderedDict()

    flat_class.functions.update(pulled_functions)

    return flat_class


class ComponentRefFlattener(TreeListener):
    """
    A listener that flattens references to components and performs name mangling,
    it also locates all symbols and determines which are states (
    one of the equations contains a derivative of the symbol)
    """

    def __init__(self, container: ast.Class, instance_prefix: str):
        self.container = container
        self.instance_prefix = instance_prefix
        self.depth = 0
        self.cutoff_depth = sys.maxsize
        self.inside_modification = 0  # We do flatten component references in modifications
        super().__init__()

    def enterClassModificationArgument(self, tree: ast.ClassModificationArgument):
        if tree.scope is not None:
            self.inside_modification += 1

    def exitClassModificationArgument(self, tree: ast.ClassModificationArgument):
        if tree.scope is not None:
            self.inside_modification -= 1

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
        # We also do not want to modify any component references inside
        # modifications (that still need to be applied), as those have an
        # accompanying scope and will be handled by the modification applier.
        # Only when modifications have been applied, will they be picked up
        # below.
        if new_name in self.container.symbols and self.inside_modification == 0:
            tree.name = new_name
            c = tree
            while len(c.child) > 0:
                c = c.child[0]
                tree.indices += c.indices
            tree.child = []
        else:
            # The component was not found in the container.  We leave this
            # reference alone.
            self.cutoff_depth = self.depth

    def exitComponentRef(self, tree: ast.ComponentRef):
        self.depth -= 1
        if self.depth < self.cutoff_depth:
            self.cutoff_depth = sys.maxsize


def flatten_component_refs(
        container: ast.Class,
        expression: ast.Union[ast.ConnectClause, ast.AssignmentStatement, ast.ForStatement, ast.Symbol],
        instance_prefix: str) -> ast.Union[ast.ConnectClause, ast.AssignmentStatement, ast.ForStatement, ast.Symbol]:
    """
    Flattens component refs in a tree
    :param container: class
    :param expression: original expression
    :param instance_prefix: prefix for instance
    :return: flattened expression
    """

    expression_copy = copy.deepcopy(expression)

    w = TreeWalker()
    w.walk(ComponentRefFlattener(container, instance_prefix), expression_copy)

    return expression_copy


class FunctionExpander(TreeListener):
    """
    Listener to extract functions
    """

    def __init__(self, node: ast.Tree, function_set: set):
        self.node = node
        self.function_set = function_set
        super().__init__()

    def exitExpression(self, tree: ast.Expression):
        if isinstance(tree.operator, ast.ComponentRef):
            try:
                function_class = self.node.find_class(tree.operator)

                full_name = str(function_class.full_reference())

                tree.operator = full_name
                self.function_set[full_name] = function_class
            except (KeyError, ast.ClassNotFoundError) as e:
                # Assume built-in function
                pass


# noinspection PyUnusedLocal
def fully_scope_function_calls(node: ast.Tree, expression: ast.Expression, function_set: OrderedDict) -> ast.Expression:
    """
    Turns the function references in this expression into fully scoped
    references (e.g. relative to absolute). The component references of all
    referenced functions are put into the functions set.

    :param node: collection for performing symbol lookup etc.
    :param expression: original expression
    :param function_set: output of function component references
    :return:
    """
    expression_copy = copy.deepcopy(expression)

    w = TreeWalker()
    w.walk(FunctionExpander(node, function_set), expression_copy)
    return expression_copy


def modify_symbol(sym: ast.Symbol, scope: ast.InstanceClass) -> None:
    """
    Apply a modification to a symbol if the scope matches (or is None)
    :param sym: symbol to apply modifications for
    :param scope: scope of modification
    """

    # We assume that we do not screw up the order of applying modifications
    # when "moving up" with the scope.
    apply_args = [x for x in sym.class_modification.arguments
                  if x.scope is None or x.scope.full_reference().to_tuple() == scope.full_reference().to_tuple()]
    skip_args = [x for x in sym.class_modification.arguments
                 if x.scope is not None and x.scope.full_reference().to_tuple() != scope.full_reference().to_tuple()]

    for class_mod_argument in apply_args:
        argument = class_mod_argument.value

        assert isinstance(argument, ast.ElementModification), \
            "Found redeclaration modification which should already have been handled."

        # TODO: Strip all non-symbol stuff.
        if argument.component.name not in ast.Symbol.ATTRIBUTES:
            raise Exception("Trying to set unknown symbol property {}".format(argument.component.name))

        setattr(sym, argument.component.name, argument.modifications[0])

    sym.class_modification.arguments = skip_args


class SymbolModificationApplier(TreeListener):
    """
    This walker applies all modifications on elementary types (e.g. Real,
    Integer, etc.). It also checks if there are any lingering modifications
    that should not be present, e.g. redeclarations, or symbol modifications
    on non-elementary types.
    """

    def __init__(self, node: ast.Node, scope: ast.InstanceClass):
        self.node = node
        self.scope = scope
        super().__init__()

    def exitSymbol(self, tree: ast.Symbol):
        if not isinstance(tree.type, ast.ComponentRef):
            assert tree.class_modification is None, \
                "Found symbol modification on non-elementary type in instance tree."
        elif tree.class_modification is not None:
            if tree.class_modification.arguments:
                modify_symbol(tree, self.scope)

            if not tree.class_modification.arguments:
                tree.class_modification = None

    # Class modification arguments may exist within annotations.
    # def exitClassModificationArgument(self, tree: ast.ClassModificationArgument):
    #     assert isinstance(tree.value, ast.ElementModification), "Found unhandled redeclaration in instance tree."

    def exitInstanceClass(self, tree: ast.InstanceClass):
        assert tree.modification_environment is None or not tree.modification_environment.arguments, \
            "Found unhandled modification on instance class."


def apply_symbol_modifications(node: ast.Node, scope: ast.InstanceClass) -> None:
    w = TreeWalker()
    w.walk(SymbolModificationApplier(node, scope), node)


class ConstantReferenceApplier(TreeListener):
    """
    This walker applies all references to constants. For each referenced
    constant it makes a symbol in the passed in InstanceClass class_, with the
    flattened component reference to the constant as the symbol's name.
    """

    def __init__(self, class_: ast.InstanceClass):
        self.classes = []

        # We cannot directly mutate the dictionary while we are looping over
        # it, so instead we store symbol updates here.
        self.extra_symbols = []

        self.depth = 0

        super().__init__()

    def enterComponentRef(self, tree: ast.ComponentRef):
        # If it is not a nested comonent reference, we do not have to do
        # anyhing as the symbol we look for would already be in the current
        # class
        self.depth += 1

        if self.depth > 1:
            # Already inside a component reference. Do not perform lookups.
            return

        if tree.child:
            try:
                self.extra_symbols[-1][str(tree)] = self.classes[-1].find_constant_symbol(tree)
            except (KeyError, ast.ClassNotFoundError, ast.FoundElementaryClassError, ast.ConstantSymbolNotFoundError):
                pass

    def exitComponentRef(self, tree: ast.ComponentRef):
        self.depth -= 1

    def enterInstanceClass(self, tree: ast.InstanceClass):
        self.classes.append(tree)
        self.extra_symbols.append(OrderedDict())

    def exitInstanceClass(self, tree: ast.InstanceClass):
        c = self.classes.pop()
        syms = self.extra_symbols.pop()
        c.symbols.update(syms)

    def enterClass(self, tree: ast.InstanceClass):
        assert False, "All classes should have been replaced by instance classes."


def apply_constant_references(class_: ast.InstanceClass) -> None:
    w = TreeWalker()
    w.walk(ConstantReferenceApplier(class_), class_)


def flatten_class(orig_class: ast.Class) -> ast.Class:
    # First we build a tree of the to-be-flattened class, with all symbol
    # types expanded to classes as well. Modifications are shifted/passed
    # along to child classes.
    # Note that no element modifications are applied (e.g of values, nominals,
    # etc), and no symbol flattening is performed.
    instance_tree = build_instance_tree(orig_class, parent=orig_class.parent)

    # At this point:
    # 1. All redeclarations have been handled.
    # 2. InstanceClasses have no modifications anymore, nor do non-elementary
    #    symbols. All modifications have been shifted to Symbols that are of
    #    one of the elementary types (Real, Integer, ...). Scope of the
    #    modification is retained, such that flattening of symbols can be done
    #    correctly.

    # Pull references to constants
    apply_constant_references(instance_tree)

    # Finally we flatten all symbols and apply modifications.
    flat_class = flatten_symbols(instance_tree)

    return flat_class


def expand_connectors(node: ast.Class) -> None:
    # keep track of which flow variables have been connected to, and which ones haven't
    disconnected_flow_variables = OrderedDict()
    for sym in node.symbols.values():
        if 'flow' in sym.prefixes:
            disconnected_flow_variables[sym.name] = sym

    # add flow equations
    # for all equations in original class
    flow_connections = OrderedDict()
    orig_equations = node.equations[:]
    node.equations = []
    for equation in orig_equations:
        if isinstance(equation, ast.ConnectClause):
            # expand connector
            assert len(equation.left.child) == 0
            assert len(equation.right.child) == 0

            sym_left = node.symbols[equation.left.name]
            sym_right = node.symbols[equation.right.name]

            try:
                class_left = getattr(sym_left, '__connector_type', None)
                if class_left is None:
                    # We may be connecting classes which are not connectors, such as Reals.
                    class_left = node.find_class(sym_left.type)
                # noinspection PyUnusedLocal
                class_right = getattr(sym_right, '__connector_type', None)
                if class_right is None:
                    # We may be connecting classes which are not connectors, such as Reals.
                    class_right = node.find_class(sym_right.type)
            except ast.FoundElementaryClassError:
                primary_types = ['Real']
                # TODO
                if sym_left.type.name not in primary_types or sym_right.type.name not in primary_types:
                    logger.warning("Connector class {} or {} not defined.  "
                                   "Assuming it to be an elementary type.".format(sym_left.type, sym_right.type))
                connect_equation = ast.Equation(left=equation.left, right=equation.right)
                node.equations.append(connect_equation)
            else:
                # TODO: Add check about matching inputs and outputs

                flat_class_left = flatten_class(class_left)

                for connector_variable in flat_class_left.symbols.values():
                    left_name = equation.left.name + CLASS_SEPARATOR + connector_variable.name
                    right_name = equation.right.name + CLASS_SEPARATOR + connector_variable.name
                    left = ast.ComponentRef(name=left_name,
                                            indices=equation.left.indices
                                                    + connector_variable.type.indices)
                    right = ast.ComponentRef(name=right_name,
                                             indices=equation.right.indices
                                                     + connector_variable.type.indices)
                    if len(connector_variable.prefixes) == 0 or connector_variable.prefixes[0] in ['input', 'output']:
                        connect_equation = ast.Equation(left=left, right=right)
                        node.equations.append(connect_equation)
                    elif connector_variable.prefixes == ['flow']:
                        # TODO generic way to get a tuple representation of a component ref, including indices.
                        left_key = (left_name,
                                    tuple(i.value for index_array in left.indices
                                          for i in index_array if i is not None),
                                    equation.__left_inner)
                        right_key = (right_name,
                                     tuple(i.value for index_array in right.indices
                                           for i in index_array if i is not None),
                                     equation.__right_inner)

                        left_connected_variables = flow_connections.get(left_key, OrderedDict())
                        right_connected_variables = flow_connections.get(right_key, OrderedDict())

                        left_connected_variables.update(right_connected_variables)
                        connected_variables = left_connected_variables
                        connected_variables[left_key] = (left, equation.__left_inner)
                        connected_variables[right_key] = (right, equation.__right_inner)

                        for connected_variable in connected_variables:
                            flow_connections[connected_variable] = connected_variables

                        # TODO When dealing with an array of connectors, we can lose
                        # disconnected flow variables in this way.  We don't initialize
                        # all components of vectors to zero in 'flow_connections' as we
                        # do not always know the length of vectors a priori.
                        disconnected_flow_variables.pop(left_name, None)
                        disconnected_flow_variables.pop(right_name, None)
                    elif connector_variable.prefixes[0] in ['constant', 'parameter']:
                        # Skip constants and parameters in connectors.
                        pass
                    else:
                        raise Exception(
                            "Unsupported connector variable prefixes {}".format(connector_variable.prefixes))
        else:
            node.equations.append(equation)

    processed = []  # OrderedDict is not hashable, so we cannot use sets.
    for connected_variables in flow_connections.values():
        if connected_variables not in processed:
            operand_specs = list(connected_variables.values())
            if np.all([not op_spec[1] for op_spec in operand_specs]):
                # All outer variables. Don't include unnecessary minus expressions.
                operands = [op_spec[0] for op_spec in operand_specs]
            else:
                operands = [op_spec[0] if op_spec[1] else ast.Expression(operator='-', operands=[op_spec[0]]) for
                            op_spec in operand_specs]
            expr = operands[-1]
            for op in reversed(operands[:-1]):
                expr = ast.Expression(operator='+', operands=[op, expr])
            connect_equation = ast.Equation(left=expr, right=ast.Primary(value=0))
            node.equations.append(connect_equation)
            processed.append(connected_variables)

    # disconnected flow variables default to 0
    for sym in disconnected_flow_variables.values():
        connect_equation = ast.Equation(left=sym, right=ast.Primary(value=0))
        node.equations.append(connect_equation)

    # strip connector symbols
    for i, sym in list(node.symbols.items()):
        if hasattr(sym, '__connector_type'):
            del node.symbols[i]


def add_state_value_equations(node: ast.Node) -> None:
    # we do this here, instead of in flatten_class, because symbol values
    # inside flattened classes may be modified later by modify_class().
    non_state_prefixes = {'constant', 'parameter'}
    for sym in node.symbols.values():
        if not (isinstance(sym.value, ast.Primary) and sym.value.value is None):
            if len(non_state_prefixes & set(sym.prefixes)) == 0:
                node.equations.append(ast.Equation(left=sym, right=sym.value))
                sym.value = ast.Primary(value=None)


def add_variable_value_statements(node: ast.Node) -> None:
    # we do this here, instead of in flatten_class, because symbol values
    # inside flattened classes may be modified later by modify_class().
    for sym in node.symbols.values():
        if not (isinstance(sym.value, ast.Primary) and sym.value.value is None):
            node.statements.append(ast.AssignmentStatement(left=[sym], right=sym.value))
            sym.value = ast.Primary(value=None)


class StateAnnotator(TreeListener):
    """
    Finds all variables that are differentiated and annotates them with the state prefix
    """

    def __init__(self, node: ast.Node):
        self.node = node
        self.in_der = 0
        super().__init__()

    def enterExpression(self, tree: ast.Expression):
        """
        When entering an expression, check if it is a derivative, if it is
        put state prefix on contained symbols
        """
        if tree.operator == 'der':
            self.in_der += 1

    def exitExpression(self, tree: ast.Expression):
        if tree.operator == 'der':
            self.in_der -= 1

    def exitComponentRef(self, tree: ast.Expression):
        if self.in_der > 0:
            assert len(tree.child) == 0

            try:
                s = self.node.symbols[tree.name]
            except KeyError:
                # Ignore index variables, parameters, and so forth.
                pass
            else:
                if 'state' not in s.prefixes:
                    s.prefixes.append('state')


def annotate_states(node: ast.Node) -> None:
    """
    Finds all derivative expressions and annotates all differentiated
    symbols as states by adding state the prefix list
    :param node: node of tree to walk
    :return:
    """
    w = TreeWalker()
    w.walk(StateAnnotator(node), node)


def flatten(root: ast.Tree, class_name: ast.ComponentRef) -> ast.Class:
    """
    This function takes a Tree and flattens it so that all subclasses instances
    are replaced by the their equations and symbols with name mangling
    of the instance name passed.
    :param root: The Tree to flatten
    :param class_name: The class that we want to create a flat model for
    :return: flat_class, a Class containing the flattened class
    """
    orig_class = root.find_class(class_name, copy=False)

    flat_class = flatten_class(orig_class)

    # expand connectors
    expand_connectors(flat_class)

    # add equations for state symbol values
    add_state_value_equations(flat_class)
    for func in flat_class.functions.values():
        add_variable_value_statements(func)

    # annotate states
    annotate_states(flat_class)

    # Put class in root
    root = ast.Tree()
    root.classes[orig_class.name] = flat_class

    # pull functions to the top level,
    # putting them prior to the model class so that they are visited
    # first by the tree walker.
    functions_and_classes = flat_class.functions
    flat_class.functions = OrderedDict()
    functions_and_classes.update(root.classes)
    root.classes = functions_and_classes

    return root
