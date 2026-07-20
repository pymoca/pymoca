#!/usr/bin/env python
"""
Modelica instantiation — MLS 5.6.1

Entry: instantiate(class_tree, class_name) → _instantiate_class()
"""

from __future__ import annotations

import copy
from typing import cast

from . import (
    InstantiationError,
    LookupOptions,
    ModelicaSemanticError,
    RecursionGuard,
)
from ._name_lookup import _find_name, find_name
from .instance import (
    InstanceClass,
    InstanceElement,
    InstanceSymbol,
    InstanceTree,
    InstantiationState,
)
from .. import ast


def instantiate(class_tree: ast.Tree, class_name: str) -> InstanceClass:
    """Instantiate a class from the class tree

    :param class_tree: The AST tree containing class definitions
    :param class_name: The name of the class to instantiate
    :return: Instantiated class in an instance tree ready for flattening

    The returned class will be fully instantiated, but name lookup
    in the instance tree may return `ast.Class` or
    `InstanceClass` with `instantiation_state` less than FULL.
    If so, these cases will need to be fully instantiated for
    flattening by calling this function on the class.
    """
    instance_tree = InstanceTree(class_tree)
    class_ = find_name(instance_tree, class_name)
    if class_ is None:
        raise InstantiationError(f"{class_name} not found in given tree")
    if isinstance(class_, ast.Symbol):
        raise InstantiationError(f"Found Symbol for {class_name} but need Class to instantiate")
    # Spec v 3.5 section 5.6.1.3 says the instance tree root is parent in the top-level call
    # That section also says the instance should be stored in the parent instance
    # so _instantiate_class does this
    instance = _instantiate_class(class_, ast.ClassModification(), instance_tree)
    return instance


def _instantiate_class(
    orig_class: ast.Class | InstanceClass,
    modification_environment: ast.ClassModification,
    parent_instance: InstanceTree | InstanceClass,
    *,
    guard: RecursionGuard | None = None,
    opts: LookupOptions | None = None,
    update_parent_instance: bool = True,
    target_state: InstantiationState = InstantiationState.FULL,
) -> InstanceClass:
    """Instantiate a class

    :param orig_class: The class to be instantiated
    :param modification_environment: The modification environment of the class
        instance
    :param parent_instance: The parent class this class is contained in
    :param guard: Cycle detection state shared across a single operation
    :param opts: Name lookup options (instantiate_in_place, search flags, etc.)
    :param update_parent_instance: If True, the parent instance will be updated with the instantiated class
    :return: The instantiated class

    Implements the instantiation rules per Modelica Language Specification version
    3.5 section 5.6.1.
    """

    # Outline of spec 3.5 section 5.6.1 *Instantiation*:
    # Definitions
    #   - Element: Class, Component (Symbol in Pymoca), or Extends Clause
    # 1. For element itself:
    #   1. Create an instance of the element to be instantiated ("partially instantiated element")
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

    if guard is None:
        guard = RecursionGuard()
    if opts is None:
        opts = LookupOptions(instantiate_in_place=False)

    lexical_parent = _get_lexical_parent_instance(
        orig_class,
        parent_instance,
        guard=guard,
        opts=opts,
    )

    # 1.1. Partially instantiate the element itself and 1.2 merge modifiers
    if (
        not isinstance(orig_class, InstanceClass)
        or orig_class.name not in parent_instance.classes
        or modification_environment.arguments
    ):
        new_class = cast(
            InstanceClass,
            _create_partial_instance(
                orig_class,
                modification_environment,
                parent_instance,
                lexical_parent,
                update_parent_instance=update_parent_instance,
            ),
        )
    else:
        # Class already at least partially instantiated and no modifications
        assert orig_class.name is not None
        new_class = parent_instance.classes[orig_class.name]
        if new_class.instantiation_state >= target_state:
            # Sync state onto orig_class so _instantiate_class_if_needed_for_lookup
            # won't re-trigger for stale InstanceClass objects that share the same
            # parent_instance but were not updated in-place (e.g. lexical-parent stubs).
            if orig_class is not new_class:
                orig_class.instantiation_state = new_class.instantiation_state
            return new_class

    guard.current_instances.add(new_class)

    # 1.3. Redeclare of element itself is done
    redeclared = _apply_redeclares(
        new_class,
        modification_environment,
        parent_instance,
        guard=guard,
        opts=opts,
    )
    if redeclared:
        return new_class

    # 2.1 Partially instantiate local classes, symbols, and extends
    # Use the parsed ast_ref as the symbol/extends source when orig_class is an
    # InstanceClass that hasn't been fully built yet (state < FULL), *or* when it is
    # a state-synced stub: the early-return path above propagates instantiation_state
    # onto the passed-in orig_class without copying symbols or extends. Sourcing from
    # such a stub would leave new_class without symbols, making
    # _check_modification_targets wrongly reject modifications that target inherited
    # symbols (e.g. "m" in PartialThermalPortInductionMachines).
    _orig_is_stub = (
        isinstance(orig_class, InstanceClass)
        and orig_class.instantiation_state >= InstantiationState.FULL
        and not orig_class.symbols
        and not orig_class.extends
    )
    if (
        isinstance(orig_class, InstanceClass)
        and orig_class.instantiation_state < InstantiationState.FULL
        or _orig_is_stub
        or orig_class.name in InstanceTree.BUILTIN_TYPES
    ):
        from_class = new_class.ast_ref
    else:
        from_class = orig_class
    assert isinstance(from_class, ast.Class), "from_class must be a Class"

    # Maintain lexical instance tree for the new class
    new_class_ast_ref = new_class.ast_ref
    assert isinstance(new_class_ast_ref, ast.Class)
    assert new_class_ast_ref.name is not None
    if new_class_ast_ref.name in lexical_parent.classes:
        lexical_parent = lexical_parent.classes[new_class_ast_ref.name]
    else:
        lexical_parent = cast(
            InstanceClass,
            _create_partial_instance(
                from_class,
                ast.ClassModification(),
                lexical_parent,
                lexical_parent,
            ),
        )
        assert isinstance(lexical_parent, (InstanceClass, InstanceTree))

    if new_class.instantiation_state < InstantiationState.PARTIAL:
        # When orig_class is already partial-instantiated, re-source its child
        # classes from the existing instances rather than the AST so any pending
        # modifications already attached to those instances (e.g. redeclares
        # waiting to apply) propagate into the fresh new_class.
        reuse_orig = (
            isinstance(orig_class, InstanceClass)
            and orig_class is not new_class
            and orig_class.instantiation_state >= InstantiationState.PARTIAL
        )
        classes_source = (
            list(orig_class.classes.values()) if reuse_orig else list(from_class.classes.values())
        )
        for class_ in classes_source:
            _create_partial_instance(
                class_,
                modification_environment,
                new_class,
                lexical_parent,
            )

        for symbol in from_class.symbols.values():
            if not isinstance(symbol, InstanceSymbol):
                symbol = _update_class_modification_scopes(symbol, new_class)  # type: ignore[assignment]
            _create_partial_instance(
                symbol,  # type: ignore[arg-type]
                modification_environment,
                new_class,
                lexical_parent,
                class_modification=symbol.class_modification,
            )

        new_class.extends = _instantiate_extends_list(  # type: ignore[attr-defined]
            from_class.extends,  # type: ignore[arg-type]
            modification_environment,
            new_class,
            guard=guard,
            opts=opts,
            target_state=InstantiationState.PARTIAL,
        )

    new_class.instantiation_state = InstantiationState.PARTIAL
    guard.current_instances.remove(new_class)

    # 2.2 Copy local contents into the element itself
    # Equations/statements are always copied (even for partial instantiation used
    # by extends) so that flattening can collect them without falling back to ast_ref.
    _copy_equations_contents(new_class)

    if target_state <= InstantiationState.PARTIAL:
        return new_class

    _copy_class_contents(new_class, copy_extends=False)

    # Step 3 (extends) was already done above during partial instantiation

    # TODO: Step 4. Check extends class lookup
    # TODO: Step 5: Check and cull elements with same name in _instantiate_class
    # See `parse_test.test_instantiation_function_input_order`

    # 6. Recursively instantiate symbols (local and inherited)
    # Class modifications like `extends A(B(x=4))` shift x=4 into B's modification_environment
    # during partial instantiation but are not yet propagated to child symbols.
    class_mod_env = new_class.modification_environment
    for symbol in cast(dict[str, InstanceSymbol], new_class.symbols).values():
        if class_mod_env.arguments:
            _apply_modifications(symbol, class_mod_env)
        _instantiate_symbol(
            symbol,
            new_class,
            guard=guard,
            opts=opts,
        )

    # Recurse down through all levels of extends and instantiate symbols
    _instantiate_extends_symbols(
        cast(list[InstanceClass], new_class.extends),
        guard=guard,
        opts=opts,
    )

    # Validate modification targets: check that all remaining ElementModification
    # args in modification_environment target existing symbols or known attributes.
    # Unconsumed args at this point indicate typos (e.g., b(x=3) when symbol is X).
    _check_modification_targets(new_class, modification_environment)

    new_class.instantiation_state = InstantiationState.FULL

    return new_class


def _instantiate_extends_symbols(
    extends: list[InstanceClass] | list[ast.ExtendsClause | InstanceClass],
    *,
    guard: RecursionGuard,
    opts: LookupOptions,
):
    for extend_node in extends:
        if not isinstance(extend_node, InstanceClass):
            continue
        for symbol in cast(dict[str, InstanceSymbol], extend_node.symbols).values():
            _instantiate_symbol(
                symbol,
                extend_node,
                guard=guard,
                opts=opts,
            )
        if extend_node.extends:
            _instantiate_extends_symbols(
                extend_node.extends,  # type: ignore[arg-type]
                guard=guard,
                opts=opts,
            )


def _update_class_modification_scopes(
    element: ast.Symbol | ast.ExtendsClause,
    current_scope: InstanceClass,
) -> ast.Symbol | ast.ExtendsClause:
    """Update the scopes of modification arguments to the given scope if not already done"""

    def resolve_scope(old_scope, new_scope):
        scope = new_scope
        if new_scope.ast_ref.full_name != old_scope.full_name:
            # Must be a short class definition which does not create a new scope
            # So we go up one level to the parent instance of the parent instance
            scope = scope.parent
        return scope

    def update_nested_args(
        args: list[ast.ClassModificationArgument],
        scope: ast.Class | InstanceClass,
    ) -> list[ast.ClassModificationArgument]:
        """Recursively update scopes of nested ClassModification args.

        Inner args inside ``Temperature(min=T_min, max=T_max)`` keep the raw
        AST class scope set at parse time.  When the outer arg's scope is
        updated to an InstanceClass, we must update the inner args too so
        that _resolve_name can skip the parent_instance walk (which may fail
        for cached type instantiations with broken parent chains).

        Returns the original list unchanged if no updates were needed.
        """
        if not isinstance(scope, InstanceClass):
            return args
        updated = []
        any_changed = False
        for arg in args:
            if isinstance(arg.scope, InstanceClass):
                updated.append(arg)
                continue
            arg_scope = resolve_scope(arg.scope, scope)
            new_arg = copy.copy(arg)
            new_arg.scope = arg_scope
            # Recurse into nested ClassModifications (e.g. T(min=a, max=b) inside outer mod)
            if isinstance(arg.value, ast.ElementModification):
                inner_scope = arg_scope if isinstance(arg_scope, InstanceClass) else scope
                new_mods = []
                sub_changed = False
                for sub in arg.value.modifications:
                    if isinstance(sub, ast.ClassModification) and sub.arguments:
                        new_sub_args = update_nested_args(sub.arguments, inner_scope)
                        if new_sub_args is not sub.arguments:
                            new_sub = copy.copy(sub)
                            new_sub.arguments = new_sub_args
                            new_mods.append(new_sub)
                            sub_changed = True
                            continue
                    new_mods.append(sub)
                if sub_changed:
                    new_value = copy.copy(arg.value)
                    new_value.modifications = new_mods
                    new_arg.value = new_value
            updated.append(new_arg)
            any_changed = True
        return updated if any_changed else args

    scoped_mod_args = []
    for arg in element.class_modification.arguments if element.class_modification else []:
        if isinstance(arg.scope, InstanceClass):
            # Already done
            scoped_mod_args.append(arg)
            continue
        scope = resolve_scope(arg.scope, current_scope)
        # Make a copy so we don't change original AST or same arg used elsewhere
        new_arg = copy.copy(arg)
        new_arg.scope = scope
        # Also update scopes of any nested ClassModification args so that
        # _resolve_name (called during flattening) can locate the scope InstanceClass
        # without relying on the parent_instance chain (which may be stale for
        # type-cache shallow clones).
        if isinstance(arg.value, ast.ElementModification) and isinstance(scope, InstanceClass):
            inner_scope = scope
            new_mods = []
            changed = False
            for sub in arg.value.modifications:
                if isinstance(sub, ast.ClassModification) and sub.arguments:
                    new_sub_args = update_nested_args(sub.arguments, inner_scope)
                    if new_sub_args is not sub.arguments:
                        new_sub = copy.copy(sub)
                        new_sub.arguments = new_sub_args
                        new_mods.append(new_sub)
                        changed = True
                        continue
                new_mods.append(sub)
            if changed:
                new_value = copy.copy(arg.value)
                new_value.modifications = new_mods
                new_arg.value = new_value
        scoped_mod_args.append(new_arg)
    if scoped_mod_args or isinstance(element, ast.ExtendsClause):
        element = copy.copy(element)
        element.class_modification = ast.ClassModification(arguments=scoped_mod_args)
    if isinstance(element, ast.ExtendsClause):
        element.scope = resolve_scope(element.scope, current_scope)
    return element


def _check_extends_rules(
    extends_list: list[InstanceClass],
    extends_refs: list[tuple[str, ...]],
    class_: InstanceClass,
) -> None:
    """Check the extends rules over the full extends list"""

    extends_builtin = set()
    extends_other = set()
    for i, ref_tuple in enumerate(extends_refs):
        composite_name = ".".join(ref_tuple)
        if len(ref_tuple) == 1 and ref_tuple[0] in InstanceTree.BUILTIN_TYPES:
            extends_builtin.add(ref_tuple[0])
            # Built-in classes contain a symbol with the same name. This causes an error
            # in the check below, so skip them. Built-in names are checked after this
            # loop completes.
            continue
        extends_other.add(composite_name)
        # Only the first path component is resolved from the class's merged scope;
        # subsequent components are resolved relative to the preceding component,
        # so inherited symbols cannot shadow them (MLS §5.3.2).
        first_ident = ref_tuple[0]
        for j, other_class in enumerate(extends_list):
            if j == i:
                # Don't check the first identifier against the class it names — only
                # against names inherited from the OTHER extends clauses (MLS §7.1.4).
                continue
            other_ast_ref = other_class.ast_ref
            assert isinstance(other_ast_ref, ast.Class)
            other_names = {
                *other_ast_ref.symbols.keys(),
                *other_ast_ref.classes.keys(),
            }
            if first_ident in other_names:
                raise ModelicaSemanticError(
                    f"Cannot extend '{class_.full_name}' with '{composite_name}'; "
                    f"'{first_ident}' also exists in names inherited from '{other_ast_ref.name}'"
                )
    if len(extends_builtin) > 1 or len(extends_builtin) and len(extends_other):
        raise ModelicaSemanticError(
            "When extending a built-in class (Real, Integer, ...) you cannot extend other classes as well"
        )


def _instantiate_extends_partially(
    extends: ast.ExtendsClause | ast.Class | InstanceClass,
    modification_environment: ast.ClassModification,
    parent_instance: InstanceClass,
    *,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> InstanceClass:
    """Instantiate an extends clause partially"""

    class_modification = ast.ClassModification()
    if isinstance(extends, ast.ExtendsClause):
        extends_updated = _update_class_modification_scopes(extends, parent_instance)
        assert isinstance(extends_updated, ast.ExtendsClause)
        extends = extends_updated
        assert extends.component is not None
        extends_class = _find_extends_class(
            extends.component,
            extends.scope,  # type: ignore[arg-type]
            guard=guard,
            opts=opts,
            class_extends=extends.is_class_extends,
        )
        ext_class_mod = extends.class_modification
        class_modification.arguments = (
            ext_class_mod.arguments if ext_class_mod else []
        ) + parent_instance.modification_environment.arguments
    else:
        extends_class = extends
        class_modification.arguments = copy.copy(parent_instance.modification_environment.arguments)

    # When the extends class is an InstanceElement (e.g., from name lookup of A.D),
    # its modification_environment contains mods from the base class definition
    # (inner/middle level). These must come BEFORE the extends clause mods (outer)
    # in class_modification to maintain correct inner-to-outer ordering.
    # We skip the normal element merge in _create_partial_instance to prevent these
    # mods from being placed in the shared modification_environment separately,
    # which would cause them to be re-merged in the wrong order during the full pass.
    skip_element_merge = False
    if (
        isinstance(extends_class, InstanceElement)
        and extends_class.modification_environment.arguments
    ):
        class_modification.arguments = (
            extends_class.modification_environment.arguments + class_modification.arguments
        )
        skip_element_merge = True

    lexical_parent = _get_lexical_parent_instance(
        extends_class,
        parent_instance,
        guard=guard,
        opts=opts,
    )
    extends_class = cast(
        InstanceClass,
        _create_partial_instance(
            extends_class,
            modification_environment,
            parent_instance,
            lexical_parent,
            update_parent_instance=False,
            class_modification=class_modification,
            skip_element_merge=skip_element_merge,
        ),
    )

    # Unnamed node (per spec)
    extends_class.name = ""

    return extends_class


def _instantiate_extends_list(
    extends_list: list[ast.ExtendsClause | ast.Class | InstanceClass],
    modification_environment: ast.ClassModification,
    parent_instance: InstanceClass,
    *,
    guard: RecursionGuard,
    opts: LookupOptions,
    target_state: InstantiationState = InstantiationState.FULL,
) -> list[InstanceClass]:
    """Instantiate extends in given list either partially for name lookup or fully"""

    # Nothing to do, just return instead of wasting time below
    if not extends_list:
        return []

    # We make 2 passes on the extends list due to potential name lookup dependencies
    extends_partially_instantiated = []
    for extends in extends_list:
        extends_instance = _instantiate_extends_partially(
            extends,
            modification_environment,
            parent_instance,
            guard=guard,
            opts=opts,
        )
        extends_partially_instantiated.append(extends_instance)

    extends_refs: list[tuple[str, ...]] = []
    for extends in extends_list:
        if isinstance(extends, ast.ExtendsClause):
            assert extends.component is not None, "extends clause must have a component"
            extends_refs.append(extends.component.to_tuple())
        else:
            assert extends.name is not None, "extends target must have a name"
            extends_refs.append((extends.name,))
    _check_extends_rules(extends_partially_instantiated, extends_refs, parent_instance)

    extends_list_instantiated = []
    for extends in extends_partially_instantiated:
        extends_instance = _instantiate_class(
            extends,
            modification_environment,
            parent_instance,
            guard=guard,
            opts=opts,
            update_parent_instance=False,
            target_state=target_state,
        )
        extends_list_instantiated.append(extends_instance)

    return extends_list_instantiated


def _ast_class_lookup(dotted_name: str, scope: ast.Class) -> ast.Class | None:
    """Resolve a dotted class name by walking up the pure-AST class hierarchy.

    Used only for `class extends X` base-class resolution, where the instance tree
    may not yet be populated for the relevant parent classes.
    """
    parts = dotted_name.split(".")
    current = scope
    while current is not None:
        if isinstance(current, InstanceClass):
            current = getattr(current, "ast_ref", None)
            if current is None:
                break
        classes = getattr(current, "classes", {})
        if parts[0] in classes:
            result = classes[parts[0]]
            ok = True
            for p in parts[1:]:
                sub = getattr(result, "classes", {})
                if p not in sub:
                    ok = False
                    break
                result = sub[p]
            if ok and isinstance(result, ast.Class):
                return result
        current = getattr(current, "parent", None)
    return None


def _find_class_extends_target(
    name: str,
    scope: ast.Class,
    visited: set | None = None,
) -> ast.Class | None:
    """Find `name` in scope's inherited AST chain for `class extends X` lookup.

    For `class extends X`, `name` must resolve to the inherited version (from scope's
    base classes), not scope's own redeclaration.  Searches scope.extends at the
    pure-AST level to find the original definition before any redeclarations.
    """
    if visited is None:
        visited = set()
    if id(scope) in visited:
        return None
    visited.add(id(scope))

    for extends in getattr(scope, "extends", []):
        if not isinstance(extends, ast.ExtendsClause):
            continue
        base = _ast_class_lookup(str(extends.component), scope)
        if base is None:
            continue
        if name in getattr(base, "classes", {}):
            return base.classes[name]
        found = _find_class_extends_target(name, base, visited)
        if found is not None:
            return found
    return None


def _find_extends_class(
    extends_name: str | ast.ComponentRef,
    scope: InstanceTree | InstanceClass,
    *,
    guard: RecursionGuard,
    opts: LookupOptions,
    class_extends: bool = False,
) -> ast.Class | InstanceClass:
    """Find the extends class and do checks"""

    if class_extends:
        # `class extends X` (MLS §7.3.1): X must resolve to the inherited version,
        # not the local redeclaration.  Use a pure-AST search on the enclosing class
        # to find the pre-redeclaration definition, avoiding the instance tree which
        # may have uninstantiated stubs or the wrong scope at this point.
        ast_scope = getattr(scope, "ast_ref", scope) if isinstance(scope, InstanceClass) else scope
        extends_class = _find_class_extends_target(str(extends_name), ast_scope)
    else:
        extends_class = _find_name(
            scope,
            extends_name,
            guard,
            LookupOptions(
                instantiate_in_place=opts.instantiate_in_place,
                search_inherited=False,
            ),
        )

    if extends_class is None:
        raise ModelicaSemanticError(
            f"Extends name {extends_name} not found in scope {scope.full_name}"
        )
    if isinstance(extends_class, ast.Symbol):
        raise ModelicaSemanticError(
            f"Cannot extend a component: {extends_name} in {scope.full_name}"
        )
    if extends_class.full_name == scope.full_name:
        raise ModelicaSemanticError(f"Cannot extend class '{extends_class.full_name}' with itself")
    if not class_extends and _is_transitively_replaceable(extends_class):
        # MLS 7.3: A redeclare without `replaceable` drops the flag. During
        # partial instantiation the redeclare hasn't been applied yet, so the
        # InstanceClass still carries the original replaceable=True.
        #
        # When the extends class was found through a composite name (e.g.,
        # `extends P2.A` where P2 = P(redeclare model A = B)), the class
        # comes from a different scope than the extends clause.  In that case
        # a pending redeclare that drops replaceability makes the extends OK.
        #
        # When the extends class is a direct child of the scope (e.g.,
        # `extends C` inside D where C is a local replaceable class), the
        # replaceability check is unconditional per MLS 7.1.4.
        has_clearing_redeclare = False
        extends_parent_ref = getattr(extends_class.parent, "ast_ref", extends_class.parent)
        scope_ref = getattr(scope, "ast_ref", scope)
        extends_parent_name = getattr(extends_parent_ref, "full_name", None)
        scope_name = getattr(scope_ref, "full_name", None)
        if isinstance(extends_class, InstanceClass) and extends_parent_name != scope_name:
            for arg in extends_class.modification_environment.arguments:
                if not arg.redeclare:
                    continue
                rdecl = arg.value
                matches = (
                    isinstance(rdecl, ast.ShortClassDefinition)
                    and rdecl.name == extends_class.name
                    or isinstance(rdecl, ast.ComponentClause)
                    and rdecl.symbol_list[0].name == extends_class.name
                )
                if matches and not rdecl.replaceable:  # type: ignore[union-attr]
                    has_clearing_redeclare = True
                    break
        if not has_clearing_redeclare:
            comp = extends_class.name
            assert extends_class.parent is not None
            full_name = extends_class.parent.full_name
            raise ModelicaSemanticError(
                f"In {full_name} extends {comp}, {comp} and parents cannot be replaceable"
            )

    return extends_class


def _get_lexical_parent_instance(
    class_: ast.Class | InstanceClass,
    lookup_scope: ast.Class | InstanceClass | InstanceTree,
    *,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> InstanceClass | InstanceTree:
    """Find the lexical parent instance of the given class"""

    if isinstance(class_, InstanceClass):
        return class_.parent  # type: ignore[return-value]
    if not isinstance(lookup_scope, InstanceClass):
        # Scope is an ast.Class, so the class parent is the lexical parent
        found = class_.parent
    else:
        # Lookup class in the instance tree
        assert class_.name is not None
        found = _find_name(
            lookup_scope,
            class_.name,
            guard,
            opts,
        )
    if (
        found is not None
        and isinstance(found.parent, InstanceClass)
        and found.full_name == class_.full_name
    ):
        return found.parent
    # Instance parent not found, so create a branch with parent instances of the class
    ancestors = class_.ancestors()
    root = lookup_scope.root
    assert isinstance(root, ast.Tree)
    parent: InstanceClass | InstanceTree
    if isinstance(root, InstanceTree):
        parent = root
    else:
        parent = InstanceTree(root)
    modifications = ast.ClassModification()
    for cls in ancestors[1:]:
        # Guard against stomping on a name in root
        update_parent = cls.name not in parent.classes
        parent = cast(
            InstanceClass,
            _create_partial_instance(cls, modifications, parent, parent, update_parent),
        )
        modifications = parent.modification_environment  # type: ignore[attr-defined]
    return parent


def _check_modification_targets(
    instance: InstanceClass,
    modification_environment: ast.ClassModification,
) -> None:
    """Check for unconsumed modification args targeting non-existent symbols.

    After all symbols (local + inherited) are instantiated, any remaining
    ElementModification args in modification_environment that don't match
    a symbol or known attribute indicate a typo in the model.
    """
    # Replaceable classes can be redeclared to a concrete type with additional
    # symbols; modifications targeting those symbols are valid (MLS §7.3).
    if instance.replaceable:
        return

    all_names = set(instance.symbols.keys()) | set(instance.classes.keys())

    # Also collect names from extends chain
    def _collect_extends_names(extends_list, names):
        for ext in extends_list:
            names.update(ext.symbols.keys())
            names.update(ext.classes.keys())
            _collect_extends_names(ext.extends, names)

    _collect_extends_names(instance.extends, all_names)

    for arg in modification_environment.arguments:
        if not isinstance(arg.value, ast.ElementModification):
            continue
        component = arg.value.component
        target = component.name
        # Skip dotted paths (e.g. x.y=3) — the first segment was already validated
        # when the parent modification was shifted down; validating sub-paths would
        # require recursing into the target's type.
        if component.child:
            continue
        if target not in all_names and target not in ast.Symbol.ATTRIBUTES:
            raise ModelicaSemanticError(
                f'Trying to modify symbol "{target}", which does not exist '
                f"in class {instance.full_name}"
            )


def _is_transitively_replaceable(class_: ast.Class) -> bool:
    while True:
        if class_.replaceable:
            return True
        if class_.parent is None:
            break
        class_ = class_.parent
        if isinstance(class_, ast.Tree):
            break
    return False


def _instantiate_symbol(
    symbol: InstanceSymbol,
    parent_instance: InstanceClass,
    *,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> None:
    """Instantiate given symbol"""

    assert isinstance(symbol, InstanceSymbol)
    assert isinstance(parent_instance, InstanceClass)

    if symbol.instantiation_state >= InstantiationState.FULL:
        return

    if symbol.name in InstanceTree.BUILTIN_TYPES:
        symbol.instantiation_state = InstantiationState.FULL
        return
    # Enum constants whose type is the enum class they live in would cause
    # infinite recursion; skip type instantiation, leaving type as ComponentRef.
    if "constant" in symbol.prefixes and ast.is_enumeration(parent_instance):
        symbol.instantiation_state = InstantiationState.FULL
        return

    modification_environment = symbol.modification_environment
    _apply_redeclares(
        symbol,
        modification_environment,
        parent_instance,
        guard=guard,
        opts=opts,
    )

    if not isinstance(symbol.type, InstanceClass):
        symbol_type = _find_name(
            parent_instance,
            symbol.type,
            guard,
            LookupOptions(instantiate_in_place=opts.instantiate_in_place),
        )
        if symbol_type is None:
            raise ModelicaSemanticError(
                f"Type {symbol.type} of symbol {symbol.name} "
                f"not found in {parent_instance.full_name}"
            )
        if isinstance(symbol_type, ast.Symbol):
            raise ModelicaSemanticError(
                f"Type {symbol.type} of symbol {symbol.name} "
                f"resolved to a Symbol, not a Class in {parent_instance.full_name}"
            )
    else:
        symbol_type = symbol.type

    # Cache type instantiation for unmodified types: the same logical type class
    # (e.g., Temperature) is fully instantiated once per flatten() operation and
    # shallow-cloned for each subsequent symbol so each symbol gets its own
    # parent_instance slot without repeating the full instantiation work.
    # Key by the object itself (not id()) so the dict holds a strong reference and
    # prevents CPython from reusing the address for a different object mid-instantiation.
    type_cache_key = None
    if not modification_environment.arguments:
        type_cache_key = symbol_type
        cached_type = guard._symbol_type_cache.get(type_cache_key)
        if cached_type is not None:
            symbol.type = _clone_type_instance(cached_type, symbol)
            _copy_symbol_contents(symbol)
            symbol.instantiation_state = InstantiationState.FULL
            return

    new_type = _instantiate_class(
        symbol_type,
        modification_environment,
        parent_instance,
        guard=guard,
        opts=opts,
        update_parent_instance=False,
    )
    symbol.type = new_type
    _copy_symbol_contents(symbol)

    # Per MLS 5.6.2 (p67): instance identifiers use component names along
    # the instance path; reparent type class under its symbol so the parent
    # chain reflects component names, not class names
    new_type.parent_instance = symbol  # type: ignore[attr-defined]

    symbol.instantiation_state = InstantiationState.FULL

    if type_cache_key is not None:
        guard._symbol_type_cache[type_cache_key] = symbol.type


def _clone_type_instance(
    type_instance: InstanceClass, parent_instance: InstanceClass | InstanceSymbol
) -> InstanceClass:
    """Clone a cached type instance so parent chains and scopes lead to the clone.

    A shallow copy would share the nested symbols, extends instances, and
    modification-argument scopes among all clones, so member references would
    resolve through whichever instance the cached original was instantiated for.
    """
    clone_by_id: dict[int, InstanceClass] = {}
    cloned = _clone_instance_subtree(type_instance, parent_instance, clone_by_id)
    _rebind_modification_scopes(cloned, clone_by_id)
    return cloned


def _clone_instance_subtree(
    instance: InstanceClass,
    parent_instance: InstanceClass | InstanceSymbol,
    clone_by_id: dict[int, InstanceClass],
) -> InstanceClass:
    """Recursively clone an instance's symbols and extends, recording originals."""
    cloned = copy.copy(instance)
    clone_by_id[id(instance)] = cloned
    cloned.parent_instance = parent_instance  # type: ignore[assignment]
    cloned.modification_environment = _clone_modification(instance.modification_environment)
    cloned.symbols = copy.copy(instance.symbols)
    for name, member in instance.symbols.items():
        member_clone = copy.copy(member)
        member_clone.parent_instance = cloned
        member_clone.modification_environment = _clone_modification(member.modification_environment)
        if isinstance(member_clone.type, InstanceClass):
            member_clone.type = _clone_instance_subtree(
                member_clone.type, member_clone, clone_by_id
            )
        cloned.symbols[name] = member_clone
    cloned.extends = [
        _clone_instance_subtree(ext, cloned, clone_by_id) if isinstance(ext, InstanceClass) else ext
        for ext in instance.extends
    ]
    return cloned


def _clone_modification(mod: ast.ClassModification) -> ast.ClassModification:
    """Copy a ClassModification with per-argument copies so scopes can be rebound."""
    new_mod = copy.copy(mod)
    new_mod.arguments = [copy.copy(arg) for arg in mod.arguments]
    return new_mod


def _rebind_modification_scopes(
    cloned: InstanceClass, clone_by_id: dict[int, InstanceClass]
) -> None:
    """Point modification-argument scopes inside a cloned subtree at the clones."""
    for arg in cloned.modification_environment.arguments:
        replacement = clone_by_id.get(id(arg.scope))
        if replacement is not None:
            arg.scope = replacement
    for member in cloned.symbols.values():
        for arg in member.modification_environment.arguments:
            replacement = clone_by_id.get(id(arg.scope))
            if replacement is not None:
                arg.scope = replacement
        if isinstance(member.type, InstanceClass):
            _rebind_modification_scopes(member.type, clone_by_id)
    for ext in cloned.extends:
        if isinstance(ext, InstanceClass):
            _rebind_modification_scopes(ext, clone_by_id)


def _create_partial_instance(
    element: ast.Class | ast.Symbol | InstanceClass | InstanceSymbol,
    modification_environment: ast.ClassModification,
    parent_instance: InstanceTree | InstanceClass,
    parent: InstanceTree | InstanceClass,
    update_parent_instance: bool = True,
    class_modification: ast.ClassModification | None = None,
    skip_element_merge: bool = False,
) -> InstanceClass | InstanceSymbol:
    """Create a class or symbol, apply modifiers, and set visibility"""

    #  Create an instance of the class to be instantiated ("partially instantiated element")
    if isinstance(element, InstanceElement):
        ast_ref = element.ast_ref
        # Merge element modifications into environment
        if not skip_element_merge:
            modification_environment.arguments = (
                element.modification_environment.arguments + modification_environment.arguments
            )
    else:
        ast_ref = element

    # Create the instance and copy attributes needed in name lookup or modification
    assert element.name is not None, "element must have a name"
    if isinstance(element, ast.Class):
        assert isinstance(ast_ref, ast.Class)
        instance = InstanceClass(
            name=element.name,
            ast_ref=ast_ref,
            parent_instance=parent_instance,
            parent=parent,
            annotation=ast.ClassModification(),
            replaceable=element.replaceable,
            encapsulated=element.encapsulated,
            partial=element.partial,
            final=element.final,
            enumeration=element.enumeration,
        )
        # Imports available immediately so enclosing-scope lookup resolves imported types.
        instance.imports.update(ast_ref.imports)
        if update_parent_instance:
            parent_instance.classes[element.name] = instance
    else:
        instance = InstanceSymbol(
            name=element.name,
            ast_ref=ast_ref,
            parent_instance=parent_instance,
            parent=parent,
            replaceable=element.replaceable,
            final=element.final,
            inner=element.inner,
            outer=element.outer,
            prefixes=list(ast_ref.prefixes),  # type: ignore[union-attr]
        )
        if update_parent_instance:
            parent_instance.symbols[element.name] = instance

    # Merge visibility
    instance.visibility = min(ast_ref.visibility, parent.visibility)  # type: ignore[union-attr]

    # Modifiers are merged for the element itself
    _apply_modifications(instance, modification_environment, class_modification)

    return instance


def _apply_modifications(
    instance: InstanceClass | InstanceSymbol,
    modification_environment: ast.ClassModification,
    class_modification: ast.ClassModification | None = None,
) -> None:
    """
    Apply modifications to an instance

    The modifications are given in `modification_environment` and an optional `class_modification`.

    Modifications are applied in reverse priority, meaning the last modification in the list
    overrides previous ones. This function handles merging modifiers, shifting them down to
    child components, and applying value modifications.

    Args:
        instance: The instance to which modifications are applied (previously created from element arg)
        modification_environment: The modification environment containing the modifiers
        class_modification: `class_modification`s to add, if any
    Raises:
        ModelicaSemanticError: If subscripting modifiers, which is not allowed
        NotImplementedError: If an unsupported modification type is encountered
    """
    # TODO: Would on-the-fly culling modifiers of same attribute be more efficient?

    # Find args from given modification_environment that apply to this instance
    apply_mod_args = [
        arg
        for arg in modification_environment.arguments
        if isinstance(arg.value, ast.ComponentClause)
        and instance.name in (arg.value.type.name, arg.value.symbol_list[0].name)
        or isinstance(arg.value, ast.ElementModification)
        and instance.name == arg.value.component.name
        or isinstance(arg.value, ast.ShortClassDefinition)
        and instance.name == arg.value.name
        or isinstance(instance, ast.Symbol)
        and instance.name in InstanceTree.BUILTIN_TYPES
        # A bare "(value=X)" on a symbol whose type is an alias one level removed
        # from a builtin (e.g. `type Accel = Real(...); constant Accel g = 9.8;`)
        # must reach the alias's own unnamed extends-to-builtin instance, the same
        # way it reaches a builtin-typed symbol's synthetic same-named member above.
        or isinstance(instance, InstanceClass)
        and instance.name in InstanceTree.BUILTIN_TYPES
        and isinstance(arg.value, ast.ElementModification)
        and arg.value.component.name == "value"
    ]

    # Remove from given modification_environment and add to instance
    if apply_mod_args:
        modification_environment.arguments = [
            arg for arg in modification_environment.arguments if arg not in apply_mod_args
        ]
        instance.modification_environment.arguments += apply_mod_args

    # All given class_modification arguments are applied
    mod = ast.ClassModification()
    if class_modification is not None:
        mod.arguments += class_modification.arguments

    # Shift modifiers down
    for arg in instance.modification_environment.arguments:
        if isinstance(arg.value, ast.ElementModification):
            if arg.value.component.indices != [[None]]:
                raise ModelicaSemanticError("Subscripting modifiers is not allowed.")
            if arg.value.component.child:
                # Move component reference down a level and apply modification
                # Don't stomp on original that may be used elsewhere
                arg = copy.copy(arg)
                arg.value = copy.copy(arg.value)
                assert isinstance(arg.value, ast.ElementModification)
                arg.value.component = arg.value.component.child[0]
                mod.arguments.append(arg)
            else:
                if instance.ast_ref.name in InstanceTree.BUILTIN_TYPES:  # type: ignore[union-attr]
                    mod.arguments.append(arg)
                else:
                    for sub_arg in arg.value.modifications:
                        if isinstance(sub_arg, ast.ClassModification):
                            mod.arguments += sub_arg.arguments
                        else:
                            # Value modification
                            # Only include the value itself in modifications, not any
                            # ClassModification siblings from the combined k(attr=v)=val form.
                            new_arg = copy.copy(arg)
                            new_arg.value = copy.copy(arg.value)
                            new_arg.value.component = ast.ComponentRef(name="value")
                            new_arg.value.modifications = [sub_arg]
                            mod.arguments.append(new_arg)
        elif isinstance(arg.value, (ast.ShortClassDefinition, ast.ComponentClause)):
            # Redeclares are handled separately
            mod.arguments.append(arg)
        else:
            raise NotImplementedError(f"{arg.value.__class__} modification")

    instance.modification_environment = mod


def _apply_redeclares(
    element: InstanceClass | InstanceSymbol,
    modification_environment: ast.ClassModification,
    parent_instance: InstanceClass | InstanceTree,
    *,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> bool:
    """Apply redeclare if any and remove from environment

    Modifies the element with the redeclare added as an unnamed extends node.

    :return: True if redeclared."""

    # TODO: Reduce the number of if statements and isinstance checks (separate class and symbol)?
    redeclares = []
    for arg in element.modification_environment.arguments:
        if not arg.redeclare:
            continue
        if (
            isinstance(arg.value, ast.ShortClassDefinition)
            and arg.value.name == element.name
            or isinstance(arg.value, ast.ComponentClause)
            and arg.value.symbol_list[0].name == element.name
        ):
            redeclares.append(arg)
    if not redeclares:
        return False

    if not element.replaceable:
        raise ModelicaSemanticError(f"Redeclaring {element.full_name} that is not replaceable")
    if element.final:
        raise ModelicaSemanticError(f"Redeclaring {element.full_name} that is final is not allowed")
    if isinstance(element, ast.Symbol) and "constant" in element.prefixes:
        raise ModelicaSemanticError(
            f"Redeclaring {element.full_name} that is constant is not allowed"
        )

    # TODO: Disallow redeclaring protected as public or public as protected
    # TODO: Check type sub-typing rules (section 6.4) w.r.t. array dimensions

    # Remove from passed modification_environment
    element.modification_environment.arguments = [
        arg for arg in element.modification_environment.arguments if arg not in redeclares
    ]
    # "Last wins" — the last redeclare in the list is the outermost.
    # The modification merging flow in _instantiate_extends_partially ensures
    # that inner mods come before outer mods in the list.
    apply_redeclare = redeclares[-1]
    redeclare = apply_redeclare.value
    scope_class = apply_redeclare.scope
    assert scope_class, "Redeclare scope should have been set by now"
    if isinstance(redeclare, ast.ShortClassDefinition):
        redeclare_name = redeclare.component
    else:  # ast.ComponentClause
        redeclare_name = redeclare.type

    redeclare_class = _find_name(
        scope_class,
        redeclare_name,
        guard,
        # Type class lookup: instantiate_in_place=False avoids triggering cascading
        # partial instantiation of the entire package hierarchy.  The ast_ref fallback
        # in _find_name covers uninstantiated InstanceClass scopes.
        LookupOptions(instantiate_in_place=False),
    )

    if redeclare_class is None:
        raise ModelicaSemanticError(
            f"Redeclare class {redeclare_name} not found in scope {scope_class.full_name}"
        )
    if isinstance(redeclare_class, ast.Symbol):
        if_symbol_msg = " type" if isinstance(element, ast.Symbol) else ""
        raise ModelicaSemanticError(
            f"Redeclaring {element.name}{if_symbol_msg} with component {redeclare_name}"
            f" in scope {scope_class.full_name}"
        )
    merged_prefixes: list[str] | None = None
    if isinstance(element, InstanceSymbol) and isinstance(redeclare, ast.ComponentClause):
        merged_prefixes = _check_and_preserve_prefixes(element, redeclare)
    _check_type_compatibility(element, redeclare_class)

    if isinstance(redeclare, ast.ShortClassDefinition):
        apply_args = redeclare.class_modification.arguments if redeclare.class_modification else []
    else:  # ast.ComponentClause
        sym_class_mod = redeclare.symbol_list[0].class_modification
        apply_args = sym_class_mod.arguments if sym_class_mod else []

    # Preserve modifications from the original class's extends clauses
    # MLS §7.3.2: when the original declaration is a short-class-definition with no
    # constrainedby clause, "the modifiers for subsequent redeclarations … are the
    # modifiers on the short-class-definition".  For `replaceable model Load=Resistor(R=R)`
    # redeclared as `model Load=Resistor`, the (R=R) from the original short-class-
    # definition must survive.  We recover those modifiers from element.ast_ref.extends
    # (the extends clause that a short-class-definition expands to) and prepend them so
    # they have the lowest priority: instance mods and explicit redeclare args (apply_args)
    # both follow and therefore override them (last wins in modification merging).
    # TODO: Implement the constrainedby case
    orig_ext_mods = []
    if isinstance(element, InstanceClass) and getattr(
        element.ast_ref, "is_short_class_definition", False
    ):
        assert isinstance(element.ast_ref, ast.Class)
        for orig_ext in element.ast_ref.extends:
            if isinstance(orig_ext, ast.ExtendsClause):
                updated = _update_class_modification_scopes(orig_ext, element)
                updated_class_mod = updated.class_modification  # type: ignore[union-attr]
                if updated_class_mod:
                    orig_ext_mods.extend(updated_class_mod.arguments)

    modification_environment.arguments = (
        orig_ext_mods + modification_environment.arguments + apply_args
    )

    # Instantiate as an extends and add to extends list
    is_class = isinstance(element, InstanceClass)
    parent_instance = element if is_class else element.parent_instance  # type: ignore[assignment]
    assert parent_instance is not None
    extends_list = [redeclare_class]
    extends_list_instantiated = _instantiate_extends_list(
        extends_list,  # type: ignore[arg-type]
        modification_environment,
        cast(InstanceClass, parent_instance),
        guard=guard,
        opts=opts,
    )
    redeclare_class = extends_list_instantiated[0]
    redeclare_class.replaceable = redeclare.replaceable
    if isinstance(element, InstanceClass):
        redeclared = element
    # Next cases are InstanceSymbol
    elif isinstance(element.type, InstanceClass):
        redeclared = element.type
    elif element.type.name in InstanceTree.BUILTIN_TYPES:
        # Builtin types are always re-instantiated from their ast_ref, so mutating
        # the resolved type's extends would be lost. Use the already-instantiated
        # redeclare_class (which carries the modifications in its symbols) directly.
        element.type = redeclare_class
        element.replaceable = redeclare.replaceable
        redeclare_class.replaceable = redeclare.replaceable
        if merged_prefixes is not None:
            element.prefixes = merged_prefixes
        return True
    else:
        # element.type is an ast.ComponentRef — fully instantiate the resolved class
        # before mutating its extends, else re-instantiation would overwrite the mutation.
        elem_parent_instance = element.parent_instance  # type: ignore[union-attr]
        assert isinstance(elem_parent_instance, InstanceClass)
        resolved = _find_name(
            elem_parent_instance,
            element.type,  # type: ignore[arg-type]
            guard,
            LookupOptions(instantiate_in_place=False),
        )
        if resolved is None:
            raise ModelicaSemanticError(
                f"Redeclared type {element.type} not found in {scope_class.full_name}"
            )
        if isinstance(resolved, ast.Symbol):
            raise ModelicaSemanticError(
                f"Redeclared type {element.type} resolved to a symbol, not a class"
            )
        redeclared = _instantiate_class(
            resolved,
            modification_environment,
            elem_parent_instance,
            guard=guard,
            opts=opts,
        )
        element.type = redeclared  # type: ignore[assignment]

    element.replaceable = redeclare.replaceable
    redeclared.replaceable = redeclare.replaceable
    if merged_prefixes is not None:
        element.prefixes = merged_prefixes  # type: ignore[union-attr]
    redeclared.extends.insert(0, redeclare_class)

    return True


_CONNECTOR_PREFIXES = frozenset({"flow", "stream"})
_CAUSALITY_PREFIXES = frozenset({"input", "output"})
_VARIABILITY_ORDER = ["constant", "parameter", "discrete"]  # strictest first


def _check_and_preserve_prefixes(
    element: InstanceSymbol, redeclare: ast.ComponentClause
) -> list[str]:
    """Merge redeclare prefixes with original element prefixes per MLS 7.3.

    Returns the merged prefix list to assign to element.prefixes.
    Raises ModelicaSemanticError on incompatible prefix changes.
    """
    orig = (
        element.ast_ref.prefixes  # type: ignore[union-attr]
    )  # element.prefixes not yet populated (pre-_copy_symbol_contents)
    new = redeclare.prefixes

    orig_conn = next((p for p in orig if p in _CONNECTOR_PREFIXES), None)
    new_conn = next((p for p in new if p in _CONNECTOR_PREFIXES), None)
    if new_conn is not None and new_conn != orig_conn:
        raise ModelicaSemanticError(
            f"Redeclare of {element.full_name!r} changes connector prefix"
            f" from {orig_conn!r} to {new_conn!r}"
        )

    orig_caus = next((p for p in orig if p in _CAUSALITY_PREFIXES), None)
    new_caus = next((p for p in new if p in _CAUSALITY_PREFIXES), None)
    if new_caus is not None and new_caus != orig_caus:
        raise ModelicaSemanticError(
            f"Redeclare of {element.full_name!r} changes causality prefix"
            f" from {orig_caus!r} to {new_caus!r}"
        )

    orig_var = next((p for p in _VARIABILITY_ORDER if p in orig), None)
    new_var = next((p for p in _VARIABILITY_ORDER if p in new), None)

    def _var_rank(v: str | None) -> int:
        return _VARIABILITY_ORDER.index(v) if v is not None else len(_VARIABILITY_ORDER)

    if new_var is not None and _var_rank(new_var) > _var_rank(orig_var):
        raise ModelicaSemanticError(
            f"Redeclare of {element.full_name!r} loosens variability"
            f" from {orig_var!r} to {new_var!r}"
        )

    return [
        p
        for p in [orig_conn, new_var if new_var is not None else orig_var, orig_caus]
        if p is not None
    ]


def _canon_class(c: ast.Class | InstanceClass) -> ast.Class:
    if isinstance(c, InstanceClass):
        assert isinstance(c.ast_ref, ast.Class)
        return c.ast_ref
    return c


def _check_type_compatibility(
    element: InstanceClass | InstanceSymbol,
    redeclare_class: ast.Class | InstanceClass,
) -> None:
    """Partial type-compatibility check for class redeclares (MLS 7.3.2, 6.4).

    Enforces that a class redeclare preserves the specialized class kind
    (model, connector, package, etc.) per MLS 6.4. Full structural subtype
    checking is not yet implemented — it requires interface comparison.

    Component redeclares (InstanceSymbol) are not checked here.
    """
    if not isinstance(element, InstanceClass):
        return

    declared_kind = _canon_class(element).type
    replacement_kind = _canon_class(redeclare_class).type
    if not declared_kind or not replacement_kind or declared_kind == replacement_kind:
        return

    raise ModelicaSemanticError(
        f"Redeclare of {element.full_name!r} changes class kind"
        f" from {declared_kind!r} to {replacement_kind!r}"
    )


def _copy_equations_contents(to_class: InstanceClass) -> None:
    """Copy equations and statements from ast_ref so extends instances carry them."""
    from_class = to_class.ast_ref
    assert isinstance(from_class, ast.Class)
    to_class.equations += from_class.equations
    to_class.initial_equations += from_class.initial_equations
    to_class.statements += from_class.statements
    to_class.initial_statements += from_class.initial_statements


def _copy_class_contents(
    to_class: InstanceClass,
    copy_extends=True,
) -> None:
    """Shallow copy of references from original to new class (excluding equations)"""
    from_class = to_class.ast_ref
    assert isinstance(from_class, ast.Class)
    if copy_extends:
        to_class.extends += from_class.extends  # type: ignore[attr-defined]
    # Equations/statements already copied by _copy_equations_contents
    if isinstance(from_class.annotation, ast.ClassModification) and to_class.annotation is not None:
        to_class.annotation.arguments += from_class.annotation.arguments
    to_class.functions.update(from_class.functions)
    to_class.comment = from_class.comment


def _copy_symbol_contents(to_symbol: InstanceSymbol) -> None:
    """Shallow copy of references from original to new symbol"""
    from_symbol = to_symbol.ast_ref
    assert isinstance(from_symbol, ast.Symbol)
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
            "replaceable",
            "prefixes",
        )
    ):
        setattr(to_symbol, attr_name, getattr(from_symbol, attr_name))
