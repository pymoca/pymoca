#!/usr/bin/env python
"""
Tools for tree walking and visiting etc.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy  # TODO
import logging
import math
import sys
from collections import OrderedDict
from typing import Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from . import ast

CLASS_SEPARATOR = "."

logger = logging.getLogger("pymoca")


# TODO Flatten function vs. conversion classes
class ModificationTargetNotFound(Exception):
    pass


class ModelicaError(Exception):
    def __init__(self, msg):
        self.msg = msg
        super().__init__(self)

    def __str__(self) -> str:
        return str(self.msg)

    def __repr__(self) -> str:
        return type(self).__name__ + "(" + str(self) + ")"


class ModelicaSemanticError(ModelicaError):
    pass


class NameLookupError(ModelicaError):
    pass


class InstantiationError(ModelicaError):
    pass


class UnimplementedError(Exception):
    pass


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

    def enterArray(self, tree: ast.Array) -> None:
        pass

    def enterAssignmentStatement(self, tree: ast.AssignmentStatement) -> None:
        pass

    def enterClass(self, tree: ast.Class) -> None:
        pass

    def enterClassModification(self, tree: ast.ClassModification) -> None:
        pass

    def enterComponentClause(self, tree: ast.ComponentClause) -> None:
        pass

    def enterComponentRef(self, tree: ast.ComponentRef) -> None:
        pass

    def enterConnectClause(self, tree: ast.ConnectClause) -> None:
        pass

    def enterElementModification(self, tree: ast.ElementModification) -> None:
        pass

    def enterEquation(self, tree: ast.Equation) -> None:
        pass

    def enterExpression(self, tree: ast.Expression) -> None:
        pass

    def enterExtendsClause(self, tree: ast.ExtendsClause) -> None:
        pass

    def enterForEquation(self, tree: ast.ForEquation) -> None:
        pass

    def enterForIndex(self, tree: ast.ForIndex) -> None:
        pass

    def enterForStatement(self, tree: ast.ForStatement) -> None:
        pass

    def enterFunction(self, tree: ast.Function) -> None:
        pass

    def enterIfEquation(self, tree: ast.IfEquation) -> None:
        pass

    def enterIfExpression(self, tree: ast.IfExpression) -> None:
        pass

    def enterIfStatement(self, tree: ast.IfStatement) -> None:
        pass

    def enterImportClause(self, tree: ast.ImportClause) -> None:
        pass

    def enterPrimary(self, tree: ast.Primary) -> None:
        pass

    def enterSlice(self, tree: ast.Slice) -> None:
        pass

    def enterSymbol(self, tree: ast.Symbol) -> None:
        pass

    def enterTree(self, tree: ast.Tree) -> None:
        pass

    def enterWhenEquation(self, tree: ast.WhenEquation) -> None:
        pass

    def enterWhenStatement(self, tree: ast.WhenStatement) -> None:
        pass

    # -------------------------------------------------------------------------
    # exit ast listeners (sorted alphabetically)
    # -------------------------------------------------------------------------

    def exitArray(self, tree: ast.Array) -> None:
        pass

    def exitAssignmentStatement(self, tree: ast.AssignmentStatement) -> None:
        pass

    def exitClass(self, tree: ast.Class) -> None:
        pass

    def exitClassModification(self, tree: ast.ClassModification) -> None:
        pass

    def exitComponentClause(self, tree: ast.ComponentClause) -> None:
        pass

    def exitComponentRef(self, tree: ast.ComponentRef) -> None:
        pass

    def exitConnectClause(self, tree: ast.ConnectClause) -> None:
        pass

    def exitElementModification(self, tree: ast.ElementModification) -> None:
        pass

    def exitEquation(self, tree: ast.Equation) -> None:
        pass

    def exitExpression(self, tree: ast.Expression) -> None:
        pass

    def exitExtendsClause(self, tree: ast.ExtendsClause) -> None:
        pass

    def exitForEquation(self, tree: ast.ForEquation) -> None:
        pass

    def exitForIndex(self, tree: ast.ForIndex) -> None:
        pass

    def exitForStatement(self, tree: ast.ForStatement) -> None:
        pass

    def exitFunction(self, tree: ast.Function) -> None:
        pass

    def exitIfEquation(self, tree: ast.IfEquation) -> None:
        pass

    def exitIfExpression(self, tree: ast.IfExpression) -> None:
        pass

    def exitIfStatement(self, tree: ast.IfStatement) -> None:
        pass

    def exitImportClause(self, tree: ast.ImportClause) -> None:
        pass

    def exitPrimary(self, tree: ast.Primary) -> None:
        pass

    def exitSlice(self, tree: ast.Slice) -> None:
        pass

    def exitSymbol(self, tree: ast.Symbol) -> None:
        pass

    def exitTree(self, tree: ast.Tree) -> None:
        pass

    def exitWhenEquation(self, tree: ast.WhenEquation) -> None:
        pass

    def exitWhenStatement(self, tree: ast.WhenStatement) -> None:
        pass


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
        if (
            isinstance(tree, (ast.Class, ast.Symbol))
            and child_name == "parent"
            or isinstance(tree, ast.ClassModificationArgument)
            and child_name in ("scope", "__deepcopy__")
        ):
            return True
        return False

    def order_keys(self, keys: Iterable[str]):
        return keys

    def walk(self, listener: TreeListener, tree: ast.Node) -> None:
        """
        Walks an AST tree recursively
        :param listener:
        :param tree:
        :return: None
        """
        name = tree.__class__.__name__
        if hasattr(listener, "enterEvery"):
            listener.enterEvery(tree)
        if hasattr(listener, "enter" + name):
            getattr(listener, "enter" + name)(tree)
        for child_name in self.order_keys(tree.__dict__.keys()):
            if self.skip_child(tree, child_name):
                continue
            self.handle_walk(listener, tree.__dict__[child_name])
        if hasattr(listener, "exitEvery"):
            listener.exitEvery(tree)
        if hasattr(listener, "exit" + name):
            getattr(listener, "exit" + name)(tree)

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


def find_name(
    name: Union[str, ast.ComponentRef],
    scope: ast.Class,
) -> Optional[Union[ast.Class, ast.Symbol]]:
    """Modelica name lookup on a tree of ast.Class and ast.InstanceClass starting at scope class

    :param name: name to look up (can be a Class or Symbol name)
    :param scope: scope in which to start name lookup

    Implements lookup rules per Modelica Language Specification version 3.5 chapter 5,
    see also chapter 13. This is more succinctly outlined in the "Modelica by Example"
    book https://mbe.modelica.university/components/packages/lookup/
    """

    return _find_name(name, scope)


def _find_name(
    name: Union[str, ast.ComponentRef],
    scope: ast.Class,
    search_imports: bool = True,
    search_parent: bool = True,
    search_inherited: bool = True,
    current_extends: Optional[Set[Union[ast.ExtendsClause, ast.InstanceExtends]]] = None,
) -> Optional[Union[ast.Class, ast.Symbol]]:
    """Internal start point for name lookup with extra parameters to control the lookup"""
    # Look for ast.Class or ast.Symbol per the MLS v3.5:
    # 1. Simple Name Lookup (spec 5.3.1)
    #     0.1 Predefined types (`Real`, `Integer`, `Boolean`, `String`) (spec 4.8)
    #     0.2 The scope itself (the given scope is the name)
    #     1. Iteration variables
    #     2. Classes
    #     3. Components (Symbols in Pymoca)
    #     4. Classes and Components from Extends Clauses
    #     5. Qualified Import names, see 4 (but not from Extends Clauses) (spec 13.2.1)
    #     6. Public Unqualified Imports (error if multiple are found) (spec 13.2.1)
    #     7. Repeat 1-6 for each lexically enclosing instance scope, stopping at
    #        `encapsulated`
    #     unless predefined type, function, operator. If name matches a variable (a.k.a.
    #     component, symbol) in an enclosing class, it must be a `constant`.
    # 2. Composite Name Lookup (e.g. `A.B.C`) (spec 5.3.2)
    #     1. `A` is looked up using Simple Name Lookup
    #     2. If `A` is a Component:
    #         1. `B.C` is looked up from named component elements of `A`
    #         2. If not found and if `A.B.C` is used as a function call and `A` is a
    #            scalar or can be
    #         evaluated as a scalar from an array and `B` and `C` are classes, it is a
    #         non-operator function call.
    #     3. If `A` is a Class:
    #         1. `A` is temporarily flattened without modifiers
    #         2. `B.C` is looked up among named elements of temp flattened class,
    #         but if `A` is not a package, lookup is restricted to `encapsulated` elements
    #         only and "the class we look inside shall not be partial in a simulation
    #         model".
    # 3. Global Name Lookup (e.g. `.A.B.C`) (spec 5.3.3)
    #     1. `A` is looked up in global scope. (`A` must be a class or a global constant.)
    #     2. If `A` is a class, follow procedure 2.3.
    # 4. Imported Name Lookup (e.g. `A.B.C`, `D = A.B.C`, `A.B.*`, or `A.B.{C,D}`) (spec
    #    13.2.1)
    #     1. `A` is looked up in global scope
    #     2. `B.C` (and `B.D`) or `B.*` is looked up. `A.B` must be a package.
    # TODO: Global name lookup

    left_name, rest_of_name = _parse_str_or_ref(name)

    # Lookup simple name first (the `A` part)
    found = _find_simple_name(
        left_name,
        scope,
        search_imports=search_imports,
        search_parent=search_parent,
        current_extends=current_extends,
    )

    # Lookup rest of name (e.g. `B.C`) to complete composite name lookup
    if found is not None and rest_of_name:
        found = _find_rest_of_name(found, rest_of_name)

    # Maintaining backward compatibility by including InstanceTree (not strictly correct)
    # TODO: (0.11) Remove InstanceTree to make spec compliant and fix test/models
    if not found and isinstance(scope, (ast.InstanceClass, InstanceTree)):
        # Not found in instance tree, look in class tree
        found = _find_name(
            name=name,
            scope=scope.ast_ref,
            search_imports=search_imports,
            search_parent=search_parent,
            current_extends=current_extends,
        )

    return found


def _parse_str_or_ref(name: Union[str, ast.ComponentRef]) -> Tuple[str, str]:
    """Return (left_name, rest_of_name) given composite name as a str or ComponentRef"""
    assert isinstance(name, (str, ast.ComponentRef))
    if isinstance(name, str):
        left_name, _, rest_of_name = name.partition(".")
    else:
        name_parts = name.to_tuple()
        left_name = name_parts[0]
        rest_of_name = ".".join(name_parts[1:])
    return left_name, rest_of_name


def _find_simple_name(
    name: str,
    scope: ast.Class,
    search_imports: bool = True,
    search_parent: bool = True,
    search_inherited: bool = True,
    current_extends: Optional[Set[Union[ast.ExtendsClause, ast.InstanceExtends]]] = None,
) -> Optional[Union[ast.Class, ast.Symbol]]:
    """Lookup name per Modelica spec 3.5 section 5.3.1 Simple Name Lookup"""

    # 0.1 Predefined types (`Real`, `Integer`, `Boolean`, `String`) (spec 4.8)
    # 0.2 The scope itself (the given scope is the name)
    # 1. Iteration variables
    # 2. Classes
    # 3. Components (Symbols in Pymoca)
    # 4. Classes and Components from Extends Clauses
    # 5. Qualified Import names, see 4 (but not from Extends Clauses) (spec 13.2.1)
    # 6. Public Unqualified Imports (error if multiple are found) (spec 13.2.1)
    # 7. Repeat 1-6 for each lexically enclosing instance scope, stopping at `encapsulated`
    # unless predefined type, function, operator. If name matches a variable (a.k.a. component,
    # symbol) in an enclosing class, it must be a `constant`.

    # Step 0.1: Predefined types
    # TODO: Implement fix for #333 to ensure we don't silently override user-defined BUILTINS
    if name in InstanceTree.BUILTIN_TYPES:
        return scope.root.classes[name]

    # Step 0.2: The scope itself
    if scope.name == name:
        return scope

    # Steps 1 - 7
    current_scope = scope
    while True:
        if (
            (
                found := _find_local(
                    name,
                    current_scope,
                )
            )
            or (
                found := _find_inherited(
                    name,
                    current_scope,
                    current_extends=current_extends,
                )
            )
            or search_imports
            and (
                found := _find_imported(
                    name,
                    current_scope,
                    current_extends=current_extends,
                )
            )
            or not search_parent
            or not current_scope.parent
            or current_scope.encapsulated
        ):
            break
        current_scope = current_scope.parent
    # If name matches a variable (a.k.a. component a.k.a. symbol) in an enclosing class,
    # it must be a `constant`.
    if (
        isinstance(found, ast.Symbol)
        and current_scope != scope
        and "constant" not in found.prefixes
        and found.name not in InstanceTree.BUILTIN_TYPES
    ):
        raise NameLookupError("Non-constant Symbol found in enclosing class")

    # If not found and we stopped at an encapsulated class,
    # then search predefined functions and operators in global scope
    # TODO: Add predefined functions and operators to global scope before this
    if found is None and current_scope.encapsulated:
        found = _find_local(name, scope.root)
        if not isinstance(found, ast.Class) or found.type not in ("function", "operator"):
            found = None

    return found


def _find_rest_of_name(
    first: Union[ast.Class, ast.Symbol], rest_of_name: str
) -> Optional[Union[ast.Class, ast.Symbol]]:
    """Lookup the `B.C` part of Composite Name Lookup (`A.B.C`) (spec 5.3.2)"""

    # 1. `A` is looked up using Simple Name Lookup and passed as `first` argument
    # 2. If `A` is a Component:
    #     1. `B.C` is looked up from named component elements of `A`
    #     2. if not found and if `A.B.C` is used as a function call and `A` is a scalar or can be
    #     evaluated as a scalar from an array and `B` and `C` are classes,
    #     it is a non-operator function call.
    # 3. If `A` is a Class:
    #     1. `A` is temporarily flattened without modifiers
    #     2. `B.C` is looked up among named elements of temp flattened class,
    #     but if `A` is not a package, lookup is restricted to `encapsulated` elements only
    #     and "the class we look inside shall not be partial in a simulation model".
    if isinstance(first, ast.Symbol):
        # Find the symbol type
        if isinstance(first.type, ast.Class):
            type_class = first.type
        else:
            type_class = _find_name(first.type, first.parent)
            if type_class is None:
                full_ref = str(first.parent.full_reference()) + "." + first.name
                raise NameLookupError(f"Lookup failed for type of symbol {full_ref}")
        found = _find_composite_name_in_symbols(rest_of_name, type_class)
        if not found:
            # Can only find in classes if the below rules apply:
            # 2b. if not found and if `A.B.C` is used as a function call
            # and `A` is a scalar or can be evaluated as a scalar from an array
            # and `B` and `C` are classes,
            # it is a non-operator function call.
            found = _find_composite_name_in_classes(rest_of_name, type_class)
            if isinstance(found, ast.Class):
                if found.type != "function":
                    found = None
                else:
                    # TODO: Fix for `test_function_lookup_via_array_element` + other possibilities
                    if first.dimensions[0][0].value is not None:
                        raise NameLookupError(
                            f"Array {first.name} must have subscripts to lookup function {found.name}"
                        )

    elif isinstance(first, ast.Class):
        found = _flatten_first_and_find_rest(first, rest_of_name)
    else:
        raise NameLookupError(f'Found unexpected node "{first!r}" during name lookup')

    return found


def _find_composite_name_in_symbols(name: str, scope: ast.Class) -> Optional[ast.Symbol]:
    """Search for composite name (e.g. A.B.C) in local symbols, recursively"""
    first_name, _, next_names = name.partition(".")
    # See spec 5.3.2 bullet 2 (emphasis mine): "If the first identifier denotes
    # a component, the rest of the name (e.g., B or B.C) is looked up among the
    # declared named *component* elements of the component".
    # This can include inherited and imported components.
    # Look up the type (Class) within the current scope if necessary
    found = _find_name(first_name, scope, search_parent=False)
    if isinstance(found, ast.Symbol):
        if next_names:
            if isinstance(found.type, ast.ComponentRef):
                type_name = str(found.type)
                found_type_class = _find_name(type_name, scope)
                if found_type_class is None or isinstance(found_type_class, ast.Symbol):
                    scope_full_reference = str(scope.full_reference())
                    raise NameLookupError(
                        f'Symbol type "{type_name}" not found in scope "{scope_full_reference}"'
                    )
            else:
                # type is already an InstanceClass
                found_type_class = found.type
            # Look in symbols of the type
            found = _find_composite_name_in_symbols(next_names, found_type_class)
    else:
        found = None
    return found


def _find_composite_name_in_classes(name: str, scope: ast.Class) -> Optional[ast.Class]:
    """Search for composite name (e.g. A.B.C) in local classes, recursively"""
    first_name, _, next_names = name.partition(".")
    found = None
    if first_name in scope.classes:
        found = scope.classes[first_name]
    if found and next_names:
        found = _find_composite_name_in_classes(next_names, found)
    return found


def _flatten_first_and_find_rest(
    first: ast.Class, rest_of_name: str
) -> Optional[Union[ast.Class, ast.Symbol]]:
    """Lookup the `B.C` part of Composite Name Lookup (`A.B.C`) where`A` is a Class"""

    # 3. If `A` is a Class:
    #     1. `A` is temporarily flattened without modifiers
    #     2. `B.C` is looked up among named elements of temp flattened class,
    #     but if `A` is not a package, lookup is restricted to `encapsulated` elements only
    #     and "the class we look inside shall not be partial in a simulation model".

    #     Checking "the class we look inside shall not be partial in a simulation model"
    #     is left to the caller.

    # Spec Section 5.3.2, 4th bullet (`A` is a class):
    # "If the identifier denotes a class, that class is temporarily
    # flattened (as if instantiating a component without modifiers of this
    # class, see section 7.2.2) and using the enclosing classes of the
    # denoted class. The rest of the name (e.g., B or B.C) is looked up
    # among the declared named elements of the temporary flattened class. If
    # the class does not satisfy the requirements for a package, the lookup
    # is restricted to encapsulated elements only. The class we look inside
    # shall not be partial in a simulation model."
    # Why do we have to temporarily flatten the class? Flattening requires name lookup
    # and with this name lookup requires flattening. Yikes! Can it be simplified?

    # TODO: Per spec v3.5 section 5.3.2 bullet 4, class is temporarily flattened
    # For now, we use recursive name lookup in contained elements
    found = _find_name(rest_of_name, first, search_parent=False)

    # Check that found meets non-package lookup requirements in spec section 5.3.2
    # The found.name test is so we only check going left to right in composite name
    # and not the other direction as we pop the recursive call stack.
    if (
        found is not None
        and found.name == _first_name(rest_of_name)
        and first.type != "package"
        and not (isinstance(found, ast.Class) and found.encapsulated)
    ):
        raise NameLookupError(f"{first.name} is not a package so {found.name} must be encapsulated")

    return found


def _first_name(name: str) -> str:
    return name.split(".")[0]


def _find_local(
    name: str,
    scope: ast.Class,
) -> Optional[Union[ast.Class, ast.Symbol]]:
    """Name lookup for predefined classes and contained elements"""

    # 1. Iteration variables
    # 2. Classes
    # 3. Components (Symbols in Pymoca)

    # 1. Iteration variables
    # TODO: Refactor when handling iteration variables (it will move up one level)
    if found := _find_iteration_variable(name, scope):
        return found

    # 2. Classes
    if name in scope.classes:
        return scope.classes[name]

    # 3. Components (Symbols in Pymoca)
    if name in scope.symbols:
        return scope.symbols[name]

    return None


def _find_iteration_variable(name: str, scope: ast.Class) -> Optional[ast.Symbol]:
    """Currently a pass"""
    # TODO: Implement find name in iteration variables
    return None


def _find_inherited(
    name: str,
    scope: ast.Class,
    current_extends: Optional[Set[Union[ast.ExtendsClause, ast.InstanceExtends]]] = None,
) -> Optional[Union[ast.Class, ast.Symbol]]:
    """Find simple name in inherited classes"""
    for extends in scope.extends:
        # Avoid infinite recursion by keeping track of where we have been with current_extends
        # A common case is when multiple classes in the same hierarchy extend the same class
        # such as Icons in the Modelica Standard Library
        if current_extends:
            if extends in current_extends:
                continue
        else:
            current_extends = set()
        current_extends.add(extends)

        if isinstance(extends, ast.InstanceExtends):
            return _find_name(
                name,
                extends,
                current_extends=current_extends,
            )

        extends_scope = _find_name(
            extends.component,
            scope,
            current_extends=current_extends,
        )
        if extends_scope is not None:
            if isinstance(extends_scope, ast.Symbol):
                continue
            found = _find_name(
                name,
                extends_scope,
                search_parent=False,
                current_extends=current_extends,
                search_imports=False,
            )
            current_extends.remove(extends)
            if found is not None:
                return found
        else:
            current_extends.remove(extends)
    return None


def _find_imported(
    name: str,
    scope: ast.Class,
    current_extends: Optional[Set[Union[ast.ExtendsClause, ast.InstanceExtends]]] = None,
) -> Optional[Union[ast.Class, ast.Symbol]]:
    """Find simple name in imports per MLS v3.5 section 13.2.1"""
    # TODO: Rewrite this to work with parser rewrite of import_clause handler.
    # TODO: Can we do a scope.imports[name] = found Class or Symbol to speed up future calls?
    # Search qualified imports (most common case)
    if name in scope.imports:
        import_: Union[ast.ImportClause, ast.ComponentRef] = scope.imports[name]
        if isinstance(import_, ast.ImportClause):
            # TODO: Handle import of multiple classes (now only does `A.B.C` for `A.B.{C,D,E}`)
            import_ = import_.components[0]
        found = _find_name(
            import_,
            scope.root,
            search_parent=False,
        )
        _check_import_rules(found, scope)
        return found
    # Unqualified imports
    if "*" in scope.imports:
        c = None
        for package_ref in scope.imports["*"].components:
            imported_comp_ref = package_ref.concatenate(ast.ComponentRef(name=name))
            # Search within the package
            # Avoid infinite recursion with search_imports = False
            c = _find_name(
                imported_comp_ref,
                scope.root,
                search_imports=False,
                search_parent=False,
                current_extends=current_extends,
            )
            # TODO: Should _check_import_rules be inside `if c is not None` check? (fix in rewrite)
            _check_import_rules(c, scope)
            if c is not None:
                # Store result for next lookup
                scope.imports[name] = imported_comp_ref
                return c
    return None


def _check_import_rules(
    element: Optional[Union[ast.Class, ast.Symbol]],
    scope: ast.Class,
) -> None:
    """Check import rules per the Modelica spec"""
    if element is None:
        return
    # TODO: Is `not element.parent` a sufficient check for the error message? (fix in rewrite)
    if not element.parent:
        raise NameLookupError(f"Import {element.name} must be contained in a package")
    if element.parent.type != "package":
        full_name = element.name
        if element.parent.name:
            full_name = str(element.parent.full_reference()) + "." + full_name
            parent = element.parent.name
            message = f"{parent} must be a package in import {full_name}"
        else:
            message = f"{full_name} is not in a package so can't be imported"
        raise NameLookupError(message)
    # TODO: Remove ast.Symbol test when visibility is added to ast.Class (see grammar)
    if isinstance(element, ast.Symbol) and element.visibility != ast.Visibility.PUBLIC:
        raise NameLookupError(f"Import {element.name} must not be protected")
    # We test on parent and name instead of just "is" because we may have a copy of a Class
    if element.parent is scope.parent and element.name == scope.name:
        full_name = str(element.parent.full_reference()) + "." + element.name
        raise NameLookupError(f"Import {full_name} is recursive")


class InstanceTree(ast.Tree):
    """The root class of an instance tree

    :param ast_ref: The root of the Abstract Syntax Tree (AST) produced by the parser

    The InstanceTree contains a reference to an `ast.Tree` produced by the parser and a
    method to instantiate a class ready for flattening. Built-in types, functions, and
    operators are added to the root of the InstanceTree when it is created.
    """

    BUILTIN_TYPES = {
        "Real": ast.Symbol(
            name="Real",
            type=ast.ComponentRef(name="Real"),
            start=ast.Primary(value=0.0),
            min=ast.Primary(value=-math.inf),
            max=ast.Primary(value=math.inf),
            nominal=ast.Primary(value=None),
            fixed=ast.Primary(value=False),  # True for parameters and constants
            unit=ast.Primary(value=""),
            quantity=ast.Primary(value=""),
            displayUnit=ast.Primary(value=""),
            # TODO: unbounded from spec is missing in Symbol
            # TODO: stateSelect from spec is missing in Symbol
        ),
        "Integer": ast.Symbol(
            name="Integer",
            type=ast.ComponentRef(name="Integer"),
            start=ast.Primary(value=0.0),
            min=ast.Primary(value=-sys.maxsize),
            max=ast.Primary(value=sys.maxsize),
            fixed=ast.Primary(value=False),  # True for parameters and constants
            quantity=ast.Primary(value=""),
        ),
        "Boolean": ast.Symbol(
            name="Boolean",
            type=ast.ComponentRef(name="Boolean"),
            start=ast.Primary(value=0.0),
            fixed=ast.Primary(value=False),  # True for parameters and constants
            quantity=ast.Primary(value=""),
        ),
        "String": ast.Symbol(
            name="String",
            type=ast.ComponentRef(name="String"),
            start=ast.Primary(value=0.0),
            fixed=ast.Primary(value=False),  # True for parameters and constants
            quantity=ast.Primary(value=""),
        ),
    }

    def __init__(self, ast_ref: ast.Tree, **kwargs):
        # The Class AST
        self.ast_ref = ast_ref

        super().__init__(**kwargs)
        self._create_builtins()

    def _create_builtins(self):
        """Add builtins to root of tree"""
        for name, symbol in self.BUILTIN_TYPES.items():
            type_class = ast.Class(name=name, type=name, parent=self)
            symbol.parent = type_class
            type_class.symbols[name] = symbol
            self.classes[name] = type_class
        # TODO: Add built-in functions (and operators?)

    def instantiate(self, class_name: str) -> ast.InstanceClass:
        """Create an instance tree used in flattening

        :param class_name: The name of the class to instantiate
        :return: Instantiated class tree ready for flattening

        Name lookup on the returned tree may still return an `ast.Class` or an
        `ast.InstanceClass` with its `fully_instantiated` attribute set to `False`. If
        so, the class will need to be fully instantiated for flattening.
        """
        class_ = find_name(class_name, self.ast_ref)
        if class_ is None:
            raise NameLookupError(f"{class_name} not found in given tree")
        if isinstance(class_, ast.Symbol):
            raise InstantiationError(f"Found Symbol for {class_name} but need Class to instantiate")
        instance = self._instantiate_class(class_, ast.ClassModification(), self)
        return instance

    def _instantiate_class(
        self,
        orig_class: Union[ast.Class, ast.InstanceClass],
        modification_environment: ast.ClassModification,
        parent: Union[ast.Class, ast.InstanceClass],
    ) -> ast.InstanceClass:
        """Instantiate a class

        :param orig_class: The class to be instantiated
        :param modification_environment: The modification environment of the class
            instance
        :param parent: The parent class of the class instance
        :return: The instantiated class

        Implements the instantiation rules per Modelica Language Specification version
        3.5 section 5.6.1.
        """

        # Outline of spec 3.5 section 5.6.1 *Instantiation*:
        # Definitions
        #   - Element: Class, Component (Symbol in Pymoca), or Extends Clause
        # 1. For element itself:
        #   1. Create an instance of the class to be instantiated ("partially instantiated element")
        #   2. Modifiers are merged for the element itself (but contained references are resolved during flattening)
        #   3. Redeclare of element itself is done
        # 2. For each element (Class or Component) in the local contents of the current element:
        #   1. Apply step 1 to the element
        #   2. Equations, algorithms, and annotations are copied into the component instance without merging
        #      (but references in equations are resolved later during flattening)
        # 3. For each element in the extends clauses of the current element:
        #   1. Apply steps 1 and 2 to the element, replacing the extends clause with the extends instance
        # 4. Lookup classes of extends and ensure it is identical to lookup result from step 3
        # 5. Check that all children of the current element (including extends) with same name are identical
        #    (error if not) and only keep one if so (to preserve function argument order)
        # 6. Components are recursively instantiated

        # TODO: Can we shortcut the partial instantiation of a partial instance with a copy/modify?
        # TODO: Investigate reuse of instantiated elements or parts of elements (mentioned in spec)

        # 1.1. Partially instantiate the element itself and 1.2 merge modifiers
        new_class = self._instantiate_partially(
            orig_class,
            modification_environment,
            parent,
        )

        # 1.3. Redeclare of element itself is done
        self._apply_class_redeclares(new_class, modification_environment)

        # 2.1 Partially instantiate local classes and symbols
        if isinstance(orig_class, ast.InstanceClass) and not orig_class.fully_instantiated:
            from_class = new_class.ast_ref
        else:
            from_class = orig_class
        for name, class_ in from_class.classes.items():
            instance = self._instantiate_partially(
                class_,
                modification_environment,
                new_class,
            )
            new_class.classes[name] = instance

        for name, symbol in from_class.symbols.items():
            instance = self._instantiate_partially(
                symbol,
                modification_environment,
                new_class,
            )
            new_class.symbols[name] = instance

        # 2.2 Copy local contents into the element itself
        self._copy_class_contents(new_class, copy_extends=True)

        # 3. Instantiate extends and 4. Check extends class lookup
        new_class.extends = self._instantiate_extends(
            new_class.extends, modification_environment, new_class
        )

        # TODO: Step 5: Check and cull elements with same name in _instantiate_class

        # 6. Recursively instantiate symbols
        for symbol in new_class.symbols.values():
            self._instantiate_symbol(symbol, new_class)

        new_class.fully_instantiated = True

        return new_class

    def _instantiate_extends(
        self,
        extends_list: List[ast.ExtendsClause],
        modification_environment: ast.ClassModification,
        parent: ast.InstanceClass,
    ) -> List[ast.InstanceClass]:
        """Instantiate extends clauses"""

        # Make sure we do not modify the passed-in list directly
        extends_list_orig = extends_list
        extends_list = extends_list.copy()

        for index, extends in enumerate(extends_list):
            extends_instance = self._instantiate_extends_single(
                extends, modification_environment, parent
            )
            extends_list[index] = extends_instance

        # Check we do not extend from any symbols/classes inherited
        extends_names = {
            _parse_str_or_ref(e.component)[0]: str(e.component) for e in extends_list_orig
        }

        for extends_name, extends_component_ref in extends_names.items():
            if extends_name in self.BUILTIN_TYPES:
                # Built-in classes contain a symbol with the same name (which
                # would cause an error). Note that this is an implementation
                # detail where we differ a little from the spec at the moment.
                continue
            for other_class in extends_list:
                other_names = {
                    *other_class.ast_ref.symbols.keys(),
                    *other_class.ast_ref.classes.keys(),
                }
                if extends_name in other_names:
                    raise ModelicaSemanticError(
                        f"Cannot extend '{parent.full_reference()}' with '{extends_component_ref}'; "
                        f"'{extends_name}' also exists in names inherited from '{other_class.ast_ref.name}'"
                    )

        return extends_list

    def _instantiate_extends_single(
        self,
        extends: ast.ExtendsClause,
        modification_environment: ast.ClassModification,
        parent: ast.InstanceClass,
    ) -> ast.InstanceClass:
        """Instantiate a single extends clause"""

        extends_class = find_name(extends.component, parent)

        if extends_class is None:
            raise ModelicaSemanticError(
                f"Extends name {extends.component} not found in scope {parent.full_reference()}"
            )
        if isinstance(extends_class, ast.Symbol):
            raise ModelicaSemanticError(
                f"Cannot extend a Symbol: {extends.component} in {parent.full_reference()}"
            )
        if str(extends_class.full_reference()) == str(parent.full_reference()):
            raise ModelicaSemanticError(
                f"Cannot extend class '{extends_class.full_reference()}' with itself"
            )
        if self._is_transitively_replaceable(extends_class):
            comp = extends_class.name
            full_name = extends_class.parent.full_reference()
            raise ModelicaSemanticError(
                f"In {full_name} extends {comp}, {comp} and parents cannot be replaceable"
            )
        # TODO: Check with spec about what to do with extends.class_modification in case of redeclare
        extend_mod = self._append_modifications(
            extends.class_modification,
            modification_environment,
        )
        extends_class = self._instantiate_class(extends_class, extend_mod, extends_class.parent)
        extends_instance = ast.InstanceExtends(
            visibility=extends.visibility,
            **extends_class.__dict__,
        )
        # TODO: Is there another way to handle `extends.class_modification`? This seems like a hack.
        # _instantiate_partially removes from extend_mod, but not original modification_environment
        # Reflect changes in extend_mod back into the passed-in modification env
        modification_environment.arguments = [
            arg for arg in modification_environment.arguments if arg in extend_mod.arguments
        ]

        if extends_instance.type in InstanceTree.BUILTIN_TYPES:
            if len(parent.extends) > 1:
                raise ModelicaSemanticError(
                    "When extending a built-in class (Real, Integer, ...) "
                    "you cannot extend other classes as well"
                )

        # TODO: Step 4 Check class lookup before and after extends

        return extends_instance

    def _is_transitively_replaceable(self, class_: ast.Class) -> bool:
        while True:
            if class_.replaceable:
                return True
            class_ = class_.parent
            if isinstance(class_, ast.Tree):
                break
        return False

    def _instantiate_symbol(
        self,
        symbol: ast.InstanceSymbol,
        parent: ast.InstanceClass,
    ) -> None:
        """Instantiate given symbol"""

        assert isinstance(symbol, ast.InstanceSymbol)
        assert isinstance(parent, (ast.InstanceClass, ast.InstanceExtends))

        if symbol.name in InstanceTree.BUILTIN_TYPES:
            symbol.fully_instantiated = True
            return

        if not isinstance(symbol.type, ast.InstanceClass):
            symbol_type = find_name(symbol.type, parent)
            if symbol_type is None:
                raise NameLookupError(f"{symbol.type} not found in {parent.full_reference()}")
            if isinstance(symbol_type, ast.InstanceElement):
                extend_args = [
                    arg
                    for arg in symbol_type.modification_environment.arguments
                    if not isinstance(arg.value, ast.ShortClassDefinition)
                ]
                symbol.modification_environment.arguments = (
                    extend_args + symbol.modification_environment.arguments
                )
            symbol.type = self._instantiate_class(
                symbol_type,
                symbol.modification_environment,
                symbol_type.parent,
            )

        self._copy_symbol_contents(symbol)

        symbol.fully_instantiated = True

    def _instantiate_partially(
        self,
        element: Union[
            ast.Class,
            ast.Symbol,
            ast.InstanceClass,
            ast.InstanceSymbol,
        ],
        modification_environment: ast.ClassModification,
        parent: Union[ast.Class, ast.InstanceClass, ast.InstanceExtends],
    ) -> Union[ast.InstanceClass, ast.InstanceSymbol]:
        """Partially instantiate a class or symbol, apply modifiers, and set visibility"""

        #  Create an instance of the class to be instantiated ("partially instantiated element")
        if isinstance(element, ast.InstanceElement):
            ast_ref = element.ast_ref
            # Merge element modifications into environment
            modification_environment.arguments = (
                element.modification_environment.arguments + modification_environment.arguments
            )
        else:
            ast_ref = element

        if isinstance(element, ast.Class):
            instance = ast.InstanceClass(
                name=element.name,
                ast_ref=ast_ref,
                parent=parent,
            )
            instance.annotation = ast.ClassModification()
            instance.replaceable = ast_ref.replaceable

            # TODO: Try connecting into class tree instead of doing _instantiate_parents_partially
            # Mirror class tree for name lookup in the InstanceTree
            # TODO: Is there a better way to maintain the path to root for all classes?
            if not isinstance(parent, (InstanceTree, ast.InstanceClass)):
                self._instantiate_parents_partially(instance)
                self.extend(instance.root)

        else:
            # TODO: Try using Symbol (and Symbol.class_modification) instead of InstanceSymbol
            instance = ast.InstanceSymbol(
                name=element.name,
                ast_ref=ast_ref,
                parent=parent,
            )

            # Merge visibility
            if hasattr(parent, "visibility"):
                if instance.visibility > parent.visibility:
                    instance.visibility = parent.visibility

        # Modifiers are merged for the element itself
        # TODO: Factor out merging/applying modifiers to separate function
        # Modifiers are added in reverse priority so the last one in the list overrides previous ones
        # TODO: Would on-the-fly culling modifiers of same attribute be more efficient?

        # Apply modifications for this instance
        instance.modification_environment.arguments = [
            arg
            for arg in modification_environment.arguments
            if isinstance(arg.value, ast.ComponentClause)
            and arg.value.type.name == element.name
            or isinstance(arg.value, ast.ElementModification)
            and arg.value.component.name == element.name
            or isinstance(arg.value, ast.ShortClassDefinition)
            and arg.value.name == element.name
            or isinstance(element, ast.Symbol)
            and element.name in InstanceTree.BUILTIN_TYPES
        ]

        # Remove applied arguments from merged modification_environment
        if instance.modification_environment.arguments:
            modification_environment.arguments = [
                arg
                for arg in modification_environment.arguments
                if arg not in instance.modification_environment.arguments
            ]

        # Shift modifiers down
        if isinstance(element, ast.Class):
            apply_modification = ast.ClassModification()
            for arg in instance.modification_environment.arguments:
                if isinstance(arg.value, ast.ElementModification):
                    for elem_class_mod in arg.value.modifications:
                        for sub_arg in elem_class_mod.arguments:
                            apply_modification.arguments.append(sub_arg)
                elif isinstance(arg.value, ast.ShortClassDefinition):
                    apply_modification.arguments.append(arg)

            instance.modification_environment = apply_modification
        elif element.name not in InstanceTree.BUILTIN_TYPES:
            # Symbol
            if ast_ref.class_modification:
                sym_mod = self._append_modifications(ast_ref.class_modification)
            else:
                sym_mod = ast.ClassModification()
            for arg in instance.modification_environment.arguments:
                if isinstance(arg.value, ast.ElementModification):
                    if arg.value.component.indices != [[None]]:
                        raise ModelicaSemanticError("Subscripting modifiers is not allowed.")
                    if len(arg.value.component.child):
                        # Move component reference down a level and apply modification to symbol
                        # Don't stomp on original that may be used elsewhere
                        arg = copy.copy(arg)
                        arg.value = copy.copy(arg.value)
                        arg.value.component = arg.value.component.child[0]
                        sym_mod.arguments.append(arg)
                    else:
                        for sub_arg in arg.value.modifications:
                            if isinstance(sub_arg, ast.ClassModification):
                                sym_mod.arguments += sub_arg.arguments
                            else:
                                # Value modification - apply to symbol
                                # Don't stomp on original that may be used elsewhere
                                sub_arg = copy.copy(arg)
                                sub_arg.value = copy.copy(arg.value)
                                sub_arg.value.component = ast.ComponentRef(name="value")
                                sym_mod.arguments.append(sub_arg)
                else:
                    raise UnimplementedError(f"{arg.value.__class__} symbol modification")

            instance.modification_environment = sym_mod

        # TODO: Fix modification scope. It is often *not* the instance parent as done here!
        # Modification scope should be the parent of the class, extends, or symbol declaration

        # Set modification argument scope now that we have an instance
        for index, arg in enumerate(instance.modification_environment.arguments):
            if arg.scope is None:
                # Make a copy so we don't change original AST or same arg used elsewhere
                new_arg = copy.copy(arg)
                new_arg.scope = instance.parent
                instance.modification_environment.arguments[index] = new_arg

        return instance

    def _instantiate_parents_partially(
        self,
        class_: ast.InstanceClass,
    ) -> None:
        """Patially instantiate parents up to root and connect to given instance tree

        This ensures names can be found in the instance tree.
        """
        instance_class = class_
        parent_class = instance_class.ast_ref.parent
        if parent_class is None:
            raise ValueError(f"Parent of {instance_class.ast_ref} unexpectedly None")
        if isinstance(parent_class, ast.Tree):
            instance_class.parent = InstanceTree(parent_class)
            instance_class.parent.classes[instance_class.name] = instance_class
            return
        parent_instance = self._instantiate_partially(
            parent_class, ast.ClassModification(), parent_class.parent
        )
        instance_class.parent = parent_instance
        parent_instance.classes[instance_class.name] = instance_class

    def _append_modifications(self, *mods: ast.ClassModification) -> ast.ClassModification:
        """Append modifications in order given"""
        combined_modification = ast.ClassModification()
        for mod in mods:
            combined_modification.arguments += mod.arguments
        return combined_modification

    def _apply_class_redeclares(
        self,
        element: ast.InstanceClass,
        modification_environment: ast.ClassModification,
    ) -> bool:
        """Apply redeclare if any and remove from environment"""

        redeclare = None
        for arg in element.modification_environment.arguments:
            if not arg.redeclare:
                continue
            if isinstance(arg.value, ast.ShortClassDefinition) and arg.value.name == element.name:
                redeclare = arg
                break
        if redeclare:
            # TODO: Remove isinstance check when Symbol.replaceable is added
            if isinstance(element, ast.Class) and not element.replaceable:
                raise ModelicaSemanticError(
                    f"Redeclaring {element.full_reference()} that is not replaceable"
                )
            scope_class = redeclare.scope
            assert scope_class, "Redeclare scope should have been set by now"
            redeclare_name = redeclare.value.component
            redeclare_class = find_name(redeclare_name, scope_class)
            if redeclare_class is None:
                raise NameLookupError(
                    f"Redeclare class {redeclare_name} not found"
                    f" in scope {scope_class.full_reference()}"
                )
            if isinstance(redeclare_class, ast.Symbol):
                raise ModelicaSemanticError(
                    f"Redeclaring class {element.name}"
                    f" with a component ({redeclare_name})"
                    f" in scope {scope_class.full_reference()}"
                )
            element.ast_ref = (
                redeclare_class.ast_ref
                if isinstance(redeclare_class, ast.InstanceClass)
                else redeclare_class
            )
            element.modification_environment.arguments.remove(redeclare)
            modification_environment.arguments = (
                redeclare.value.class_modification.arguments + modification_environment.arguments
            )

        return True if redeclare else False

    def _copy_class_contents(
        self,
        to_class: Union[ast.InstanceClass, ast.InstanceExtends],
        copy_extends=True,
    ) -> None:
        """Shallow copy of references from original to new class"""
        from_class = to_class.ast_ref
        to_class.imports.update(from_class.imports)
        if copy_extends:
            to_class.extends += from_class.extends
        to_class.equations += from_class.equations
        to_class.initial_equations += from_class.initial_equations
        to_class.statements += from_class.statements
        to_class.initial_statements += from_class.initial_statements
        if isinstance(from_class.annotation, ast.ClassModification):
            to_class.annotation.arguments += from_class.annotation.arguments
        to_class.functions.update(from_class.functions)
        to_class.comment = from_class.comment

    def _copy_symbol_contents(self, to_symbol: ast.InstanceSymbol) -> None:
        """Shallow copy of references from original to new symbol"""
        from_symbol = to_symbol.ast_ref
        for attr_name in (
            name
            for name in from_symbol.__dict__
            if name
            not in (
                "name",
                "type",
                "visibility",
                "ATTRIBUTES",
                "class_modification",
                "parent",
            )
        ):
            setattr(to_symbol, attr_name, getattr(from_symbol, attr_name))


def flatten_extends(
    orig_class: Union[ast.Class, ast.InstanceClass], modification_environment=None, parent=None
) -> ast.InstanceClass:
    extended_orig_class = ast.InstanceClass(
        name=orig_class.name,
        type=orig_class.type,
        comment=orig_class.comment,
        annotation=ast.ClassModification(),
        parent=parent,
    )

    if isinstance(orig_class, ast.InstanceClass):
        extended_orig_class.modification_environment = orig_class.modification_environment

    for extends in orig_class.extends:
        c = orig_class.find_class(extends.component, check_builtin_classes=True)

        if str(c.full_reference()) == str(orig_class.full_reference()):
            raise Exception("Cannot extend class '{}' with itself".format(c.full_reference()))

        if c.type == "__builtin":
            if len(orig_class.extends) > 1:
                raise Exception(
                    "When extending a built-in class (Real, Integer, ...) you cannot extend other classes as well"
                )
            extended_orig_class.type = c.type

        c = flatten_extends(c, extends.class_modification, parent=c.parent)

        # Imports are not inherited (spec 3.5 sections 5.3.1 and 7.1)
        # extended_orig_class.imports.update(c.imports)
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
        extended_orig_class.modification_environment.arguments.extend(
            c.modification_environment.arguments
        )

        # set visibility and parent
        for sym in extended_orig_class.symbols.values():
            if sym.visibility > extends.visibility:
                sym.visibility = extends.visibility
            sym.parent = extended_orig_class

    extended_orig_class.imports.update(orig_class.imports)
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
        extended_orig_class.modification_environment.arguments.extend(
            modification_environment.arguments
        )

    # If the current class is inheriting an elementary type, we shift modifications from the class to its __value symbol
    if extended_orig_class.type == "__builtin":
        if extended_orig_class.symbols["__value"].class_modification is not None:
            extended_orig_class.symbols["__value"].class_modification.arguments.extend(
                extended_orig_class.modification_environment.arguments
            )
        else:
            extended_orig_class.symbols["__value"].class_modification = (
                extended_orig_class.modification_environment
            )

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


def build_instance_tree(
    orig_class: Union[ast.Class, ast.InstanceClass], modification_environment=None, parent=None
) -> ast.InstanceClass:
    extended_orig_class = flatten_extends(orig_class, modification_environment, parent)

    # Redeclarations take effect
    for class_mod_argument in extended_orig_class.modification_environment.arguments:
        if not class_mod_argument.redeclare:
            continue
        scope_class = (
            class_mod_argument.scope
            if class_mod_argument.scope is not None
            else extended_orig_class
        )
        argument = class_mod_argument.value
        if isinstance(argument, ast.ShortClassDefinition):
            old_class = extended_orig_class.classes[argument.name]
            extended_orig_class.classes[argument.name] = scope_class.find_class(argument.component)

            # Fix references to symbol types that were already in the instance tree
            for sym in extended_orig_class.symbols.values():
                if isinstance(sym.type, ast.InstanceClass) and sym.type.name is old_class.name:
                    c = extended_orig_class.classes[argument.name]
                    sym.type = build_instance_tree(c, sym.class_modification, c.parent)
        elif isinstance(argument, ast.ComponentClause):
            # Redeclaration of symbols
            # TODO: Do we need to handle scoping of redeclarations of symbols?
            for s in argument.symbol_list:
                extended_orig_class.symbols[s.name].type = s.type
        else:
            raise Exception("Unknown redeclaration type")

    extended_orig_class.modification_environment.arguments = [
        x for x in extended_orig_class.modification_environment.arguments if not x.redeclare
    ]

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

        sub_class_arguments = [
            x
            for x in extended_orig_class.modification_environment.arguments
            if isinstance(x.value, ast.ElementModification) and x.value.component.name == class_name
        ]

        # Remove from current class's modification environment
        extended_orig_class.modification_environment.arguments = [
            x
            for x in extended_orig_class.modification_environment.arguments
            if x not in sub_class_arguments
        ]

        for main_mod in sub_class_arguments:
            for elem_class_mod in main_mod.value.modifications:
                for arg in elem_class_mod.arguments:
                    sub_class_modification.arguments.append(arg)

        extended_orig_class.classes[class_name] = build_instance_tree(
            c, sub_class_modification, extended_orig_class
        )

    # Check that all symbol modifications to be applied on this class exist
    for arg in extended_orig_class.modification_environment.arguments:
        if (
            arg.value.component.name not in extended_orig_class.symbols
            and arg.value.component.name not in ast.Symbol.ATTRIBUTES
        ):
            raise ModificationTargetNotFound(
                'Trying to modify symbol "{}", which does not exist in class {}'.format(
                    arg.value.component.name, extended_orig_class.full_reference()
                )
            )

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
            sym_arguments = [
                x
                for x in extended_orig_class.modification_environment.arguments
                if isinstance(x.value, ast.ElementModification)
                and x.value.component.name == sym_name
                or sym_name == "__value"
                and x.value.component.name == "value"
            ]

            # Remove from current class's modification environment
            extended_orig_class.modification_environment.arguments = [
                x
                for x in extended_orig_class.modification_environment.arguments
                if x not in sym_arguments
            ]

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
            sym_arguments = [
                x
                for x in extended_orig_class.modification_environment.arguments
                if isinstance(x.value, ast.ElementModification)
                and x.value.component.name == sym_name
            ]

            # Remove from current class's modification environment
            extended_orig_class.modification_environment.arguments = [
                x
                for x in extended_orig_class.modification_environment.arguments
                if x not in sym_arguments
            ]

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
                raise type(e)('Processing failed for symbol "{}":\n{}'.format(error_sym, e)) from e

            sym.class_modification = None

    return extended_orig_class


def flatten_symbols(class_: ast.InstanceClass, instance_name="") -> ast.Class:
    # Recursive symbol flattening

    flat_class = ast.Class(
        name=class_.name,
        type=class_.type,
        comment=class_.comment,
        annotation=class_.annotation,
    )

    # append period to non empty instance_name
    if instance_name != "":
        instance_prefix = instance_name + CLASS_SEPARATOR
    else:
        instance_prefix = instance_name

    # for all symbols in the original class
    for sym_name, sym in class_.symbols.items():
        sym.name = instance_prefix + sym_name
        if instance_prefix:
            # Strip 'input' and 'output' prefixes from nested symbols.
            strip_keywords = ["input", "output"]
            for strip_keyword in strip_keywords:
                try:
                    sym.prefixes.remove(strip_keyword)
                except ValueError:
                    pass

        flat_sym = sym

        if isinstance(sym.type, ast.ComponentRef):
            # Elementary type
            flat_sym.dimensions = flat_sym.dimensions
            flat_class.symbols[flat_sym.name] = flat_sym
        elif sym.type.type == "__builtin" or (
            sym.type.type == "type"
            and "__value" in sym.type.symbols
            and sym.type.symbols["__value"].type.name in ast.Class.BUILTIN
        ):
            # Class inherited from elementary type (e.g. "type Voltage =
            # Real"). No flattening to be done, just copying over all
            # attributes and modifications to the class's "__value" symbol.
            flat_sym.dimensions = flat_sym.dimensions
            flat_class.symbols[flat_sym.name] = flat_sym

            if flat_sym.class_modification is not None:
                flat_sym.class_modification.arguments.extend(
                    sym.type.symbols["__value"].class_modification.arguments
                )
            else:
                flat_sym.class_modification = sym.type.symbols["__value"].class_modification

            for att in flat_sym.ATTRIBUTES + ["type"]:
                setattr(
                    flat_class.symbols[flat_sym.name],
                    att,
                    getattr(sym.type.symbols["__value"], att),
                )

            continue
        else:
            # recursively call flatten on the contained class
            flat_sub_class = flatten_symbols(sym.type, flat_sym.name)

            # carry class dimensions over to symbols
            for flat_class_symbol in flat_sub_class.symbols.values():
                flat_class_symbol.dimensions = flat_sym.dimensions + flat_class_symbol.dimensions

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
            if sym.type.type == "connector":
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
            if not hasattr(flat_equation, "__left_inner"):
                flat_equation.__left_inner = len(equation.left.child) > 0
            if not hasattr(flat_equation, "__right_inner"):
                flat_equation.__right_inner = len(equation.right.child) > 0

    # Create fully scoped equivalents
    fs_initial_equations = [
        fully_scope_function_calls(class_, e, pulled_functions) for e in class_.initial_equations
    ]
    fs_statements = [
        fully_scope_function_calls(class_, e, pulled_functions) for e in class_.statements
    ]
    fs_initial_statements = [
        fully_scope_function_calls(class_, e, pulled_functions) for e in class_.initial_statements
    ]

    flat_class.initial_equations += [
        flatten_component_refs(flat_class, e, instance_prefix) for e in fs_initial_equations
    ]
    flat_class.statements += [
        flatten_component_refs(flat_class, e, instance_prefix) for e in fs_statements
    ]
    flat_class.initial_statements += [
        flatten_component_refs(flat_class, e, instance_prefix) for e in fs_initial_statements
    ]

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
    instance_prefix: str,
) -> ast.Union[ast.ConnectClause, ast.AssignmentStatement, ast.ForStatement, ast.Symbol]:
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

    def __init__(self, node: ast.Tree, function_set: OrderedDict):
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
            except (KeyError, ast.ClassNotFoundError):
                # Assume built-in function
                pass


def fully_scope_function_calls(
    node: ast.Tree, expression: ast.Expression, function_set: OrderedDict
) -> ast.Expression:
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
    apply_args = [
        x
        for x in sym.class_modification.arguments
        if x.scope is None
        or x.scope.full_reference().to_tuple() == scope.full_reference().to_tuple()
    ]
    skip_args = [
        x
        for x in sym.class_modification.arguments
        if x.scope is not None
        and x.scope.full_reference().to_tuple() != scope.full_reference().to_tuple()
    ]

    for class_mod_argument in apply_args:
        argument = class_mod_argument.value

        assert isinstance(
            argument, ast.ElementModification
        ), "Found redeclaration modification which should already have been handled."

        # TODO: Strip all non-symbol stuff.
        if argument.component.name not in ast.Symbol.ATTRIBUTES:
            raise Exception(
                "Trying to set unknown symbol property {}".format(argument.component.name)
            )

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
            assert (
                tree.class_modification is None
            ), "Found symbol modification on non-elementary type in instance tree."
        elif tree.class_modification is not None:
            if tree.class_modification.arguments:
                modify_symbol(tree, self.scope)

            if not tree.class_modification.arguments:
                tree.class_modification = None

    # Class modification arguments may exist within annotations.
    # def exitClassModificationArgument(self, tree: ast.ClassModificationArgument):
    #     assert isinstance(tree.value, ast.ElementModification), "Found unhandled redeclaration in instance tree."

    def exitInstanceClass(self, tree: ast.InstanceClass):
        assert (
            tree.modification_environment is None or not tree.modification_environment.arguments
        ), "Found unhandled modification on instance class."


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
            except (
                KeyError,
                ast.ClassNotFoundError,
                ast.FoundElementaryClassError,
                ast.ConstantSymbolNotFoundError,
            ):
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
        raise AssertionError("All classes should have been replaced by instance classes.")


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
        if "flow" in sym.prefixes:
            disconnected_flow_variables[sym.name] = sym

    # add flow equations
    # for all equations in original class
    flow_connections = OrderedDict()
    orig_equations = node.equations[:]
    node.equations = []
    for equation in orig_equations:
        if isinstance(equation, ast.ConnectClause):
            # expand connector
            if len(equation.left.child) != 0:
                raise Exception(
                    "Could not resolve {} in connect clause ({}*, {}*)".format(
                        equation.left, equation.left, equation.right
                    )
                )
            if len(equation.right.child) != 0:
                raise Exception(
                    "Could not resolve {} in connect clause ({}*, {}*)".format(
                        equation.right, equation.left, equation.right
                    )
                )

            sym_left = node.symbols[equation.left.name]
            sym_right = node.symbols[equation.right.name]

            try:
                class_left = getattr(sym_left, "__connector_type", None)
                if class_left is None:
                    # We may be connecting classes which are not connectors, such as Reals.
                    class_left = node.find_class(sym_left.type)
                class_right = getattr(sym_right, "__connector_type", None)
                if class_right is None:
                    # We may be connecting classes which are not connectors, such as Reals.
                    class_right = node.find_class(sym_right.type)
            except ast.FoundElementaryClassError:
                primary_types = ["Real"]
                # TODO
                if (
                    sym_left.type.name not in primary_types
                    or sym_right.type.name not in primary_types
                ):
                    logger.warning(
                        "Connector class {} or {} not defined.  "
                        "Assuming it to be an elementary type.".format(
                            sym_left.type, sym_right.type
                        )
                    )
                connect_equation = ast.Equation(left=equation.left, right=equation.right)
                node.equations.append(connect_equation)
            else:
                # TODO: Add check about matching inputs and outputs

                flat_class_left = flatten_class(class_left)

                for connector_variable in flat_class_left.symbols.values():
                    left_name = equation.left.name + CLASS_SEPARATOR + connector_variable.name
                    right_name = equation.right.name + CLASS_SEPARATOR + connector_variable.name
                    left = ast.ComponentRef(
                        name=left_name,
                        indices=equation.left.indices + connector_variable.type.indices,
                    )
                    right = ast.ComponentRef(
                        name=right_name,
                        indices=equation.right.indices + connector_variable.type.indices,
                    )
                    if len(connector_variable.prefixes) == 0 or connector_variable.prefixes[0] in [
                        "input",
                        "output",
                    ]:
                        connect_equation = ast.Equation(left=left, right=right)
                        node.equations.append(connect_equation)
                    elif connector_variable.prefixes == ["flow"]:
                        # TODO generic way to get a tuple representation of a component ref, including indices.
                        left_key = (
                            left_name,
                            tuple(
                                i.value
                                for index_array in left.indices
                                for i in index_array
                                if i is not None
                            ),
                            equation.__left_inner,
                        )
                        right_key = (
                            right_name,
                            tuple(
                                i.value
                                for index_array in right.indices
                                for i in index_array
                                if i is not None
                            ),
                            equation.__right_inner,
                        )

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
                    elif connector_variable.prefixes[0] in ["constant", "parameter"]:
                        # Skip constants and parameters in connectors.
                        pass
                    else:
                        raise Exception(
                            "Unsupported connector variable prefixes {}".format(
                                connector_variable.prefixes
                            )
                        )
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
                operands = [
                    (
                        op_spec[0]
                        if op_spec[1]
                        else ast.Expression(operator="-", operands=[op_spec[0]])
                    )
                    for op_spec in operand_specs
                ]
            expr = operands[-1]
            for op in reversed(operands[:-1]):
                expr = ast.Expression(operator="+", operands=[op, expr])
            connect_equation = ast.Equation(left=expr, right=ast.Primary(value=0))
            node.equations.append(connect_equation)
            processed.append(connected_variables)

    # disconnected flow variables default to 0
    for sym in disconnected_flow_variables.values():
        connect_equation = ast.Equation(left=sym, right=ast.Primary(value=0))
        node.equations.append(connect_equation)

    # strip connector symbols
    for i, sym in list(node.symbols.items()):
        if hasattr(sym, "__connector_type"):
            del node.symbols[i]


def add_state_value_equations(node: ast.Node) -> None:
    # we do this here, instead of in flatten_class, because symbol values
    # inside flattened classes may be modified later by modify_class().
    non_state_prefixes = {"constant", "parameter"}
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
        if tree.operator == "der":
            self.in_der += 1

    def exitExpression(self, tree: ast.Expression):
        if tree.operator == "der":
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
                if "state" not in s.prefixes:
                    s.prefixes.append("state")


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
    flat_name = str(orig_class.full_reference())
    flat_class.name = flat_name
    root.classes[flat_name] = flat_class

    # pull functions to the top level,
    # putting them prior to the model class so that they are visited
    # first by the tree walker.
    functions_and_classes = flat_class.functions
    flat_class.functions = OrderedDict()
    functions_and_classes.update(root.classes)
    root.classes = functions_and_classes

    return root
