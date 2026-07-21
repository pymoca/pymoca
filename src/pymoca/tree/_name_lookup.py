#!/usr/bin/env python
"""
Modelica name lookup — MLS Chapter 5

Entry: find_name(scope, name) → _find_name() → _find_simple_name() / _find_rest_of_name()
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import cast

from . import LookupOptions, NameLookupError, RecursionGuard
from .instance import (
    InstanceClass,
    InstanceSymbol,
    InstanceTree,
    InstantiationState,
)
from .. import ast

# Predefined type names visible past an encapsulated boundary (MLS 5.3.1)
_PREDEFINED_NAMES = (
    frozenset(ast.Tree.BUILTIN_TYPES)
    | frozenset(ast.Tree.BUILTIN_ENUM_TYPES)
    | ast.Tree.BUILTIN_OPAQUE_TYPES
)


def find_name(
    scope: ast.Class,
    name: str | ast.ComponentRef,
) -> ast.Class | ast.Symbol | None:
    """Modelica name lookup on a tree of ast.Class and InstanceClass starting at scope class

    :param scope: scope in which to start name lookup
    :param name: name to look up (can be a Class or Symbol name)

    Implements lookup rules per Modelica Language Specification version 3.5 chapter 5,
    see also chapter 13. This is more succinctly outlined in the "Modelica by Example"
    book https://mbe.modelica.university/components/packages/lookup/
    """
    if not isinstance(scope, ast.Class):
        raise TypeError(f"scope must be a Class or Tree, not {type(scope)}")
    if not isinstance(name, (str, ast.ComponentRef)):
        raise TypeError(f"name must be a string or ComponentRef, not {type(name)}")
    return _find_name(scope, name, RecursionGuard(), LookupOptions())


def _find_name(
    scope: ast.Class,
    name: str | ast.ComponentRef,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> ast.Class | ast.Symbol | None:
    """Internal start point for name lookup with extra parameters to control the lookup"""
    # Look for ast.Class or ast.Symbol per the MLS v3.5:
    # 1. Simple Name Lookup (spec 5.3.1)
    #     1. Iteration variables (handled during flattening, never reach lookup)
    #     2. Classes
    #     3. Components (Symbols in Pymoca)
    #     4. Classes and Components from Extends Clauses
    #     5. Qualified Import names, see 4 (but not from Extends Clauses) (spec 13.2.1)
    #     6. Public Unqualified Imports (error if multiple are found) (spec 13.2.1)
    #     7. a) Repeat 1-6 for each lexically enclosing instance scope,
    #        b) stopping at `encapsulated`
    #        c) unless predefined type, function, operator then look in root.
    #        d) If name matches a variable (a.k.a. component, symbol) in an enclosing class, it
    #           must be a `constant`.
    # 2. Composite Name Lookup (e.g. `A.B.C`) (spec 5.3.2)
    #     1. `A` is looked up using Simple Name Lookup
    #     2. If `A` is a Component:
    #         1. `B.C` is looked up from named component elements of `A`
    #         2. If not found and if `A.B.C` is used as a function call and `A` is a
    #            scalar or can be
    #         evaluated as a scalar from an array and `B` and `C` are classes, it is a
    #         non-operator function call.
    #     3. If `A` is a Class:
    #         1. `A` is temporarily flattened without modifiers of this class
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
    left_name, rest_of_name = _parse_str_or_ref(name)

    # Global name lookup (MLS 5.3.3): a leading dot (empty-name head element) means
    # the first identifier is looked up in the global scope, so restart the lookup
    # from the root with the enclosing-scope search disabled.
    if left_name == "":
        if not rest_of_name:
            raise NameLookupError(f'Invalid global name "{name}" in scope {scope.full_name}')
        return _find_name(scope.root, rest_of_name, guard, replace(opts, search_parent=False))

    # Lookup simple name first (the `A` part)
    found = _find_simple_name(scope, left_name, guard, opts)

    # Lookup rest of name (e.g. `B.C`) to complete composite name lookup
    # Per MLS 5.6.1, search_inherited=False only restricts the first (simple) name
    # lookup in extends clauses. Once the first name is found, the rest of the
    # composite name should use normal lookup including inherited elements.
    if found is not None and rest_of_name:
        found = _find_rest_of_name(found, rest_of_name, guard, replace(opts, search_inherited=True))

    # Whole-search fallback to the lexical class tree (ast_ref) for instance scopes
    # whose parent chain does not reach the root - classes temporarily flattened for
    # composite name lookup (MLS 5.3.2 bullet 4) - so enclosing scopes and root
    # builtins stay reachable through the lexical chain.
    if (
        not found
        and isinstance(scope, InstanceClass)
        and (
            not guard.current_extends
            # Also fall back to the class tree for uninstantiated InstanceClasses even
            # when inside an extends traversal. An InstanceClass at state=0 has empty
            # .classes, so without this fallback types defined directly inside an
            # uninstantiated package (e.g. FixedPhase inside Types_ic) are invisible.
            # Falling back to ast_ref is safe: ast.Class is not an InstanceClass, so
            # this recursive call cannot re-trigger the fallback.
            or scope.instantiation_state < InstantiationState.PARTIAL
        )
    ):
        # Not found in instance tree, look in class tree
        ast_ref = scope.ast_ref
        assert isinstance(ast_ref, ast.Class), "InstanceClass/InstanceTree.ast_ref must be a Class"
        found = _find_name(ast_ref, name, guard, opts)

    return found


def _parse_str_or_ref(name: str | ast.ComponentRef) -> tuple[str, str]:
    """Return (left_name, rest_of_name) given composite name as a str or ComponentRef"""
    assert isinstance(name, (str, ast.ComponentRef))
    if isinstance(name, str):
        left_name, _, rest_of_name = name.partition(".")
    else:
        name_parts = name.to_tuple()
        left_name = name_parts[0]
        rest_of_name = ".".join(name_parts[1:])
    return left_name, rest_of_name


def _instantiate_class_if_needed_for_lookup(
    class_: ast.Class | InstanceClass,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> ast.Class | InstanceClass:
    """Instantiate class as needed for name lookup"""
    # If not an instance, then all names are already available for lookup
    if isinstance(class_, ast.Tree) or not isinstance(class_, InstanceClass):
        return class_
    # Same if already at least partially instantiated
    if class_.instantiation_state >= InstantiationState.PARTIAL:
        return class_
    # Only instantiate if we are not already instantiating this class
    # If a name is not available yet in class_ in process of being instantiated,
    # then name lookup will fall back to looking for it in the AST
    if class_ in guard.current_instances:
        return class_
    # Late import to avoid circular dependency
    from ._instantiation import _instantiate_class

    instance = _instantiate_class(
        class_,
        ast.ClassModification(),
        class_.parent_instance,  # type: ignore[arg-type]
        guard=guard,
        opts=opts,
        target_state=InstantiationState.PARTIAL,
    )
    return instance
    # TODO: Full instantiation in place for flattening
    # return _instantiate_class(
    #     class_,
    #     ast.ClassModification(),
    #     class_.parent_instance,
    # )


def _find_simple_name(
    scope: ast.Class,
    name: str,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> ast.Class | ast.Symbol | None:
    """Lookup name per Modelica spec 3.5 section 5.3.1 Simple Name Lookup"""

    # Step numbers below refer to part 1 of the outline in _find_name.

    current_scope = scope

    # Search through enclosing scopes until we find something or hit a boundary
    while True:

        if opts.instantiate_in_place:
            current_scope = _instantiate_class_if_needed_for_lookup(current_scope, guard, opts)

        # Steps 2-3: Try local lookup first (classes, symbols)
        if found := _find_local(current_scope, name):
            break

        # Step 4: Look in inherited classes if enabled
        if opts.search_inherited:
            if found := _find_inherited(current_scope, name, guard, opts):
                break

        # Steps 5-6: Look in imports if enabled
        if opts.search_imports:
            if found := _find_imported(current_scope, name, guard, opts):
                break

        # Step 7b: Continue unless we should stop
        if not opts.search_parent or not current_scope.parent or current_scope.encapsulated:
            break

        # Step 7a: Move up to parent scope and continue searching
        current_scope = current_scope.parent

    # Step 7c: If not found and we stopped at an encapsulated class, then search
    # predefined types, functions, and operators in global scope (MLS 5.3.1).
    # Other root-level names are not visible past an encapsulated boundary; use a
    # global name (MLS 5.3.3) or an import (MLS 13.2.1) instead.
    # TODO: Add predefined functions and operators to global scope before this
    if found is None and current_scope.encapsulated and name in _PREDEFINED_NAMES:
        found = _find_local(scope.root, name)

    # Step 7d: If name matches a variable (a.k.a. component, symbol) in an enclosing class,
    # it must be a `constant`.
    if isinstance(found, ast.Symbol) and current_scope != scope:
        # For InstanceSymbol before full instantiation, prefixes may not have
        # been copied from ast_ref yet; check the authoritative source.
        prefixes = (
            found.ast_ref.prefixes  # type: ignore[union-attr]
            if isinstance(found, InstanceSymbol)
            else found.prefixes
        )
        if "constant" not in prefixes:
            raise NameLookupError("Non-constant Symbol found in enclosing class")

    return found


def _satisfies_package_requirements(cls: ast.Class) -> bool:
    """True if *cls* may be looked up like a package (MLS 5.3.3, 13.1).

    A class satisfies the requirements for a package when it contains only
    classes and constants -- i.e. every component is a constant. Enumeration
    literals are constants, so enumeration types qualify.
    """
    return all("constant" in sym.prefixes for sym in cls.symbols.values())


def _find_rest_of_name(
    first: ast.Class | ast.Symbol,
    rest_of_name: str,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> ast.Class | ast.Symbol | None:
    """Lookup the `B.C` part of Composite Name Lookup (`A.B.C`) (spec 5.3.2)

    `first` is the already-looked-up `A`."""

    # See part 2 of the outline in _find_name.

    if isinstance(first, ast.Symbol):
        # Find the symbol type
        if isinstance(first.type, ast.Class):
            type_class = first.type
        else:
            assert first.parent is not None, "Symbol must have a parent class"
            type_class = _find_name(first.parent, first.type, guard, opts)
            if type_class is None or isinstance(type_class, ast.Symbol):
                raise NameLookupError(f"Lookup failed for type of symbol {first.full_name}")
        type_class_c = cast(ast.Class, type_class)
        found = _find_composite_name_in_symbols(type_class_c, rest_of_name, guard, opts)
        if not found:
            # Can only find in classes if the below rules apply:
            # 2b. if not found and if `A.B.C` is used as a function call
            # and `A` is a scalar or can be evaluated as a scalar from an array
            # and `B` and `C` are classes,
            # it is a non-operator function call.
            found, _ = _find_composite_name_in_classes(type_class_c, rest_of_name, guard, opts)
            if isinstance(found, ast.Class):
                if found.type != "function":
                    found = None
                else:
                    # TODO: Fix for `test_function_lookup_via_array_element` + other possibilities
                    if first.dimensions[0][0].value is not None:  # type: ignore[union-attr]
                        raise NameLookupError(
                            f"Array {first.name} must have subscripts to lookup function {found.name}"
                        )

    elif isinstance(first, ast.Class):
        # Don't call _flatten_first_and_find_rest when first is a package that is currently
        # being instantiated — doing so would cause infinite recursion through extends-class
        # name lookup that re-enters this same scope (MLS 5.3.2 bullet 4).
        _currently_instantiating = first.type == "package" and (
            first in guard.current_instances
            or any(inst.ast_ref is first for inst in guard.current_instances)
        )
        if not opts.instantiate_in_place and not _currently_instantiating:
            found = _flatten_first_and_find_rest(
                first, rest_of_name, guard, replace(opts, instantiate_in_place=True)
            )
        else:
            found = _find_name(first, rest_of_name, guard, opts)
        # Check that found meets non-package lookup requirements in spec section 5.3.2
        # The found.name test is so we only check going left to right in composite name
        # and not the other direction as we pop the recursive call stack. A class that
        # satisfies the requirements for a package (only classes and constants) is looked
        # up like a package (MLS 5.3.3); this also covers enumeration literal access.
        if (
            opts.check_encapsulated
            and found is not None
            and found.name == _first_name(rest_of_name)
            and first.type != "package"
            and not _satisfies_package_requirements(first)
            and not (isinstance(found, ast.Class) and found.encapsulated)
        ):
            raise NameLookupError(
                f"{first.name} is not a package so {found.name} must be encapsulated"
            )

    else:
        raise NameLookupError(f'Found unexpected node "{first!r}" during name lookup')

    return found


def _find_composite_name_in_symbols(
    scope: ast.Class,
    name: str,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> ast.Symbol | None:
    """Search for composite name (e.g. A.B.C) in local symbols, recursively"""
    if opts.instantiate_in_place:
        scope = _instantiate_class_if_needed_for_lookup(scope, guard, opts)

    first_name, _, next_names = name.partition(".")

    # See spec 5.3.2 bullet 2 (emphasis mine): "If the first identifier denotes
    # a component, the rest of the name (e.g., B or B.C) is looked up among the
    # declared named *component* elements of the component".
    # This can include inherited and imported components.
    # Look up the type (Class) within the current scope if necessary
    found = _find_name(scope, first_name, guard, replace(opts, search_parent=False))
    if isinstance(found, ast.Symbol):
        if next_names:
            if isinstance(found.type, ast.ComponentRef):
                type_name = str(found.type)
                found_type_class = _find_name(scope, type_name, guard, opts)
                if found_type_class is None or isinstance(found_type_class, ast.Symbol):
                    raise NameLookupError(
                        f'Symbol type "{type_name}" not found in scope "{scope.full_name}"'
                    )
            else:
                # type is already an InstanceClass
                found_type_class = found.type
            # Look in symbols of the type
            found = _find_composite_name_in_symbols(found_type_class, next_names, guard, opts)
    else:
        found = None
    return found


def _find_composite_name_in_classes(
    scope: ast.Class,
    name: str,
    guard: RecursionGuard,
    opts: LookupOptions,
    return_last_found=False,
) -> tuple[ast.Class | None, str]:
    """Search for composite name (e.g. A.B.C) in local classes, recursively"""
    if opts.instantiate_in_place:
        scope = _instantiate_class_if_needed_for_lookup(scope, guard, opts)

    first_name, _, next_names = name.partition(".")
    found = _find_local_class(scope, first_name)
    if found is None:
        next_names = name
    if found and next_names:
        next_found, next_names = _find_composite_name_in_classes(found, next_names, guard, opts)
        if next_found:
            found = next_found
        elif not return_last_found:
            found = None
    return found, next_names


def _find_composite_name(
    scope: ast.Class,
    name: str,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> ast.Symbol | ast.Class | None:
    """Composite name lookup using partial instantiation only"""
    opts_in_place = replace(opts, instantiate_in_place=True)
    # Recurse through children classes
    found, next_names = _find_composite_name_in_classes(
        scope, name, guard, opts_in_place, return_last_found=True
    )
    # Recurse through children symbols, if any
    if found and next_names:
        found = _find_composite_name_in_symbols(found, next_names, guard, opts_in_place)

    return found


def _flatten_first_and_find_rest(
    first: ast.Class,
    rest_of_name: str,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> ast.Class | ast.Symbol | None:
    """Lookup the `B.C` part of Composite Name Lookup (`A.B.C`) where `A` is a Class

    Checking "the class we look inside shall not be partial in a simulation model"
    is left to the caller."""
    # Implements part 2.3 of the outline in _find_name.

    # Late imports to avoid circular dependency
    from ._instantiation import _get_lexical_parent_instance, _instantiate_class
    from ._flattening import _create_partial_flat_instance, _flatten_instance

    # Per spec v3.5 section 5.3.2 bullet 4, class is temporarily flattened
    if (
        not isinstance(first, InstanceClass)
        or first.instantiation_state < InstantiationState.PARTIAL
    ):
        parent_instance = getattr(first, "parent_instance", None)
        if parent_instance is None:
            assert first.parent is not None, "Class must have a parent for name lookup"
            parent_instance = _get_lexical_parent_instance(
                first,
                first.parent,
                guard=guard,
                opts=opts,
            )
        first_instance = _instantiate_class(
            first,
            ast.ClassModification(),
            parent_instance,
            guard=guard,
            opts=opts,
            target_state=InstantiationState.PARTIAL,
        )
    else:
        first_instance = first
    flat_class = _create_partial_flat_instance(first_instance)
    _flatten_instance(
        first_instance,
        flat_class,
        guard=guard,
        opts=replace(opts, instantiate_in_place=True),
    )
    # Need non-flat name for name lookup
    assert flat_class.name is not None, "flat_class must have a name"
    flat_class.name = flat_class.name.split(".")[-1]
    first = flat_class

    if rest_of_name in flat_class.symbols:
        # Flattening allows us to bypass standard name lookup if all symbols
        # Copy before mutating — flat_class uses shallow copies so the symbol
        # object may be shared with the original instance tree.
        found = copy.copy(flat_class.symbols[rest_of_name])
        # Restore non-flat name of instance (last name if compound name)
        found.name = found.name.split(".")[-1]
    else:
        # Classes with embedded references to the same class cause infinite recursion if we do
        # instantiation or temporary flattening recursively, so it is only done on the first
        # in a composite name.
        # The spec is a ambiguous about what to do when looking up the rest.
        found = _find_name(
            first_instance,
            rest_of_name,
            guard,
            replace(opts, search_parent=False),
        )

    return found


def _first_name(name: str) -> str:
    return name.split(".")[0]


def _find_local(
    scope: ast.Class,
    name: str,
) -> ast.Class | ast.Symbol | None:
    """Name lookup for predefined classes and contained elements"""

    # 2. Classes
    if found := _find_local_class(scope, name):
        return found

    # 3. Components (Symbols in Pymoca)
    if name in scope.symbols:
        return scope.symbols[name]
    if isinstance(scope, (InstanceClass, InstanceTree)):
        ast_ref = scope.ast_ref
        assert isinstance(ast_ref, ast.Class), "InstanceClass/InstanceTree.ast_ref must be a Class"
        if name in ast_ref.symbols:
            return ast_ref.symbols[name]

    return None


def _find_local_class(scope: ast.Class, name: str) -> ast.Class | None:
    """Find a locally declared class, including declared-but-uninstantiated ones.

    An instance scope may lack instances for some declared children (an unparsed
    MODELICAPATH stub skipped during partial instantiation, a scope below PARTIAL,
    or a top-level class never copied into the InstanceTree root). Falling through
    to the scope's ast_ref keeps such names visible at the local tier, so they
    shadow inherited, imported, and enclosing-scope names per MLS 5.3.1.
    """
    if name in scope.classes:
        return scope.classes[name]
    if isinstance(scope, (InstanceClass, InstanceTree)):
        ast_ref = scope.ast_ref
        assert isinstance(ast_ref, ast.Class), "InstanceClass/InstanceTree.ast_ref must be a Class"
        if name in ast_ref.classes:
            return ast_ref.classes[name]
    return None


_SENTINEL = object()


def _find_inherited(
    scope: ast.Class,
    name: str,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> ast.Class | ast.Symbol | None:
    """Find simple name in inherited classes"""
    # Normally .extends is populated at PARTIAL instantiation. But an enclosing
    # InstanceClass reached as a parent scope can be FULL-but-empty-extends while its
    # ast_ref still carries the original extends clauses. Fall back to those so
    # transitively-inherited nested types remain visible during name lookup.
    extends_list = scope.extends
    if (
        not extends_list
        and isinstance(scope, InstanceClass)
        and isinstance(scope.ast_ref, ast.Class)
    ):
        extends_list = scope.ast_ref.extends
    if not extends_list:
        return None

    # Cross-call memoization: the extends chain is populated at PARTIAL instantiation
    # and doesn't grow after that, so (name, id(scope)) results are stable.
    # ast.Class scopes are immutable AST nodes and are also safe to cache.
    # This prevents O(N_lookups × N_extends_depth) work when the same scope appears
    # in the enclosing chain of many different symbol-type lookups.
    cacheable = (
        not isinstance(scope, InstanceClass)
        or scope.instantiation_state >= InstantiationState.PARTIAL
    )
    cache_key = (name, id(scope))
    if cacheable:
        cached = guard._find_inherited_cache.get(cache_key, _SENTINEL)
        if cached is not _SENTINEL:
            return cached

    # Initialize the per-lookup deduplication set on the first call in a chain.
    # Keyed by (name, id(extends_scope)) so the same class is never searched twice for the
    # same name via different diamond-inheritance paths, reducing O(N^D) to O(N*D).
    # The set is a mutable reference so all recursive calls in the same chain share it.
    if opts._searched_extends is None:
        opts = replace(opts, _searched_extends=set())
    searched = opts._searched_extends
    assert searched is not None

    result = None
    for extends in extends_list:
        # Avoid infinite recursion by keeping track of where we have been with current_extends
        # A common case is when multiple classes in the same hierarchy extend the same class
        # such as Icons in the Modelica Standard Library
        if extends in guard.current_extends:
            continue
        guard.current_extends.add(extends)

        if isinstance(extends, InstanceClass):
            key = (name, id(extends))
            if key in searched:
                guard.current_extends.discard(extends)
                continue
            searched.add(key)
            found = _find_name(
                extends, name, guard, replace(opts, search_imports=False, search_parent=False)
            )
            guard.current_extends.discard(extends)
            if found is not None:
                result = found
                break
            continue

        # Resolve the extends class name using lexical lookup only. Inherited lookup
        # is not needed for base class name resolution and would cause exponential
        # recursion through nested extends chains. search_parent=True is forced because
        # base class names always resolve lexically up to enclosing packages — callers
        # may pass search_parent=False for the target-name search itself, but that
        # restriction must not bleed into base-class-name resolution.
        if extends.component is None:
            guard.current_extends.discard(extends)
            continue
        extends_scope = _find_name(
            scope,
            extends.component,
            guard,
            replace(opts, search_inherited=False, search_parent=True, _searched_extends=None),
        )
        if extends_scope is not None:
            if isinstance(extends_scope, ast.Symbol):
                guard.current_extends.discard(extends)
                continue
            key = (name, id(extends_scope))
            if key in searched:
                guard.current_extends.discard(extends)
                continue
            searched.add(key)
            found = _find_name(
                extends_scope,
                name,
                guard,
                replace(opts, search_imports=False, search_parent=False),
            )
            guard.current_extends.discard(extends)
            if found is not None:
                result = found
                break
        else:
            guard.current_extends.discard(extends)

    if cacheable:
        guard._find_inherited_cache[cache_key] = result
    return result


def _find_imported(
    scope: ast.Class,
    name: str,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> ast.Class | ast.Symbol | None:
    """Find simple name in imports per MLS v3.5 section 13.2.1"""
    # TODO: Rewrite this to work with parser rewrite of import_clause handler.

    def _lookup_via_import(
        import_ref: ast.ComponentRef, search_imports: bool = True
    ) -> tuple[ast.Class | ast.Symbol | None, bool]:
        """Resolve a single import ref to a class or symbol"""
        # Detect self-referential imports to prevent infinite recursion during instantiation
        common_parent, children = _get_common_parent(scope, str(import_ref))
        if common_parent is not None:
            if not children:
                # Import target IS common_parent (e.g. importing an enclosing package).
                # _find_composite_name("", ...) returns None, so return the package directly.
                return common_parent, True
            found = _find_composite_name(common_parent, children, guard, opts)
            _check_import_rules(scope, found)
            return found, True
        fallback_opts = replace(opts, search_parent=False, search_imports=search_imports)
        found = _find_name(scope.root, import_ref, guard, fallback_opts)
        # TODO: Should _check_import_rules be inside `if found is not None` check? (fix in rewrite)
        _check_import_rules(scope, found)
        return found, False

    # Search qualified imports (most common case)
    if name in scope.imports:
        import_: ast.ImportClause | ast.ComponentRef = scope.imports[name]
        if isinstance(import_, ast.ImportClause):
            # TODO: Handle import of multiple classes (now only does `A.B.C` for `A.B.{C,D,E}`)
            import_ = import_.components[0]
        if scope.full_name == str(import_):
            raise NameLookupError(f"Import {import_} in scope {scope.full_name} is recursive")
        found, _ = _lookup_via_import(import_)
        return found
    # Unqualified imports
    if "*" in scope.imports:
        wildcard_import = scope.imports["*"]
        assert isinstance(wildcard_import, ast.ImportClause)
        for package_ref in wildcard_import.components:
            imported_comp_ref = package_ref.concatenate(ast.ComponentRef(name=name))
            found, stop_search = _lookup_via_import(imported_comp_ref, search_imports=False)
            if stop_search:
                return found
            if found is not None:
                # Cache only on InstanceClass; raw ast.Class is shared across flatten calls.
                if isinstance(scope, InstanceClass):
                    scope.imports[name] = imported_comp_ref
                return found
    return None


def _get_common_parent(class_: ast.Class, name: str) -> tuple[ast.Class | None, str]:
    class_parts = class_.full_name.split(".")
    name_parts = name.split(".")
    common_count = 0
    for a, b in zip(class_parts, name_parts):
        if a != b:
            break
        common_count += 1
    if common_count == 0:
        return (None, "")
    parent = class_
    for _ in range(len(class_parts) - common_count):
        assert parent.parent is not None, "Class tree does not have enough parent levels"
        parent = parent.parent
    # If the common ancestor is an instance-tree stub (e.g. a branch created with
    # update_parent_instance=False whose state was synced via _instantiate_class but
    # whose classes dict was never fully populated), substitute the authoritative
    # instance stored in parent_instance.classes so that sub-package lookups succeed.
    if isinstance(parent, InstanceClass) and parent.name is not None:
        pi = parent.parent_instance
        if pi is not None:
            real = pi.classes.get(parent.name)
            if real is not None and real is not parent:
                parent = real
    child_names = name_parts[common_count:]
    return parent, ".".join(child_names)


def _check_import_rules(
    scope: ast.Class,
    element: ast.Class | ast.Symbol | None,
) -> None:
    """Check import rules per the Modelica spec"""
    if element is None:
        return
    # TODO: Is `not element.parent` a sufficient check for the error message? (fix in rewrite)
    if not element.parent:
        raise NameLookupError(
            f"Import {element.name} in scope {scope.full_name} must be contained in a package"
        )
    if element.parent.type != "package":
        if element.parent.name:
            message = (
                f"{element.parent.name} must be a package in "
                f"import {element.full_name} in scope {scope.full_name}"
            )
        else:
            message = (
                f"{element.full_name} is not in a package "
                f"so can't be imported in scope {scope.full_name}"
            )
        raise NameLookupError(message)
    if element.visibility != ast.Visibility.PUBLIC:
        raise NameLookupError(
            f"Import {element.name} must not be protected in scope {scope.full_name}"
        )
    if scope.full_name == element.full_name:
        raise NameLookupError(f"Import {element.full_name} is recursive in scope {scope.full_name}")
