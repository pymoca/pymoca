#!/usr/bin/env python
"""
Modelica flattening — MLS 5.6.2

Entry: flatten_class(root, class_name) → InstanceClass (new pipeline, steps 1+2)
       flatten_instance(instance) → _flatten_instance()
Adapter: flatten_to_tree(root, class_name) → ast.Tree (for backend compatibility)
"""

from __future__ import annotations

import copy
import math
import operator
import sys
from collections import OrderedDict
from typing import cast

import numpy as np

from . import LookupOptions, ModelicaSemanticError, NameLookupError, RecursionGuard
from ._instantiation import (
    _get_lexical_parent_instance,
    _instantiate_class,
    instantiate,
)
from ._listener import TreeListener, TreeWalker, logger
from ._name_lookup import _find_name
from .instance import (
    InstanceClass,
    InstanceElement,
    InstanceSymbol,
    InstanceTree,
    InstantiationState,
    element_instance_name_tuple,
)
from .. import ast


def flatten_instance(
    instance: InstanceClass,
    expand_connect: bool = True,
    evaluate_parameters: bool = False,
) -> InstanceClass:
    """Flatten an instance class

    :param instance: The instance class to flatten
    :param expand_connect: Whether to expand ``ConnectClause``s into equations. When
        False, they are left in ``flat.equations`` unexpanded. Connector member
        symbols are present in the flattened class either way.
    :param evaluate_parameters: When True, fold parameter and constant values to literals
    :return: The flattened class
    """
    if not isinstance(instance, InstanceClass):
        raise TypeError(f"Expected InstanceClass, got {type(instance)}")

    opts = LookupOptions(evaluate_parameters=evaluate_parameters)
    flat_class = _create_partial_flat_instance(instance)
    _flatten_instance(instance, flat_class, opts=opts)
    _flatten_discovered_functions(flat_class)
    _flatten_value_ref_names(flat_class)
    if evaluate_parameters:
        _evaluate_parameter_values(flat_class)
    _generate_value_equations(flat_class)
    _check_all_references_valid(flat_class)
    _process_transitions(flat_class)
    if expand_connect:
        _generate_connect_equations(flat_class)
    return flat_class


def _create_partial_flat_instance(instance: InstanceClass) -> InstanceClass:
    flat_class = InstanceClass(
        name=_flatten_name(instance),
        comment=instance.comment,
        type=instance.type,
        ast_ref=instance.ast_ref,
        parent=instance.parent,
        parent_instance=instance.parent_instance,
        instantiation_state=instance.instantiation_state,
    )

    # Populate classes and extends for name lookup
    for name, class_ in instance.classes.items():
        class_copy = copy.copy(class_)
        class_copy.parent_instance = flat_class  # type: ignore[attr-defined]
        flat_class.classes[name] = class_copy
    for extends in instance.extends:  # type: ignore[attr-defined]
        extends_copy = copy.copy(extends)
        extends_copy.parent_instance = flat_class  # type: ignore[attr-defined]
        flat_class.extends.append(extends_copy)

    return flat_class


def _flatten_instance(
    instance: InstanceClass,
    flat_class: InstanceClass,
    *,
    guard: RecursionGuard | None = None,
    opts: LookupOptions | None = None,
    prefix: str = "",
) -> None:
    """Flatten an instance class

    :param instance: The instance class to flatten
    :param flat_class: The flattened class; function calls discovered while flattening
        are accumulated directly into ``flat_class.functions``
    :param guard: Cycle detection state shared across a single operation
    :param opts: Name lookup options (instantiate_in_place, search flags, etc.)
    :param prefix: The prefix for the current instance
    :return: None

    The passed instance may be updated by fully instantiating elements as needed.
    We allow the top-level flattened instance to retain connectors so that we can
    flatten a set of components and connect them together later.
    """
    # Outline of spec 3.5 section 5.6.2 *Generation of the flat equation system*.
    # "Simple Type" below means Integer, Real, Boolean, String, enumeration, Clock,
    # or External Object.
    #
    # 1 Recursively walk the tree, beginning at the class to be flattened, and process
    # components with the following items used throughout:
    # * A. "all references by name in conditional declarations, modifications, dimension
    #   definitions, annotations, equations and algorithms are resolved to the real
    #   instance to which they are referring to ... [Either the referenced instance
    #   belongs to the model to be simulated the path starts at the model itself, or if
    #   not, it starts at the unnamed root of the instance tree, e.g. in case of a
    #   constant in a package.]"
    # * B. Names are replaced by the global unique identifier of the instance "[This
    #   identifier is normally constructed from the names of the instances along a path
    #   in the instance tree (and omitting the unnamed nodes of extends clauses),
    #   separated by dots. Either the referenced instance belongs to the model to be
    #   simulated the path starts at the model itself, or if not, it starts at the
    #   unnamed root of the instance tree, e.g. in case of a constant in a package.]"
    # * C. Classes and symbols are instantiated as needed (if not a fully instantiated instance)
    # * D. Classes are ignored unless needed by components
    # * E. Function calls encountered in the above are processed as needed:
    #   - Fill the call with default arguments (section 12.4.1) (TODO)
    #   - Look up the function in the instance tree
    #   - Recursively flatten the function call
    #   - Add to the list of functions
    #
    # 1.1 Insert components into the variables list
    # 1.2 Evaluate the conditional declaration expression and mark the declaration for
    #     deletion (TODO)
    # 1.3 Resolve dimensions, including enclosing instances
    # 1.4 Resolve modifications of value attributes of simple types and records (TODO: records)
    # 1.5 Resolve modifications of other attributes of simple types
    # 1.6 Recursively "handle" non-simple types
    # 1.7 Resolve references in equations and algorithms
    # 1.8 Recursively "handle" unnamed extends instances
    # 1.9 Check that all references now point to a valid instance (not conditionally false)
    #
    # 2 Generate connect equations for all connections in the flattened tree
    #
    # 3 Process transitions in the flattened tree (TODO)

    if guard is None:
        guard = RecursionGuard()
    if opts is None:
        opts = LookupOptions()

    # 1.1–1.6 Process local symbols per MLS 5.6.2
    for name, symbol in list(cast(dict[str, InstanceSymbol], instance.symbols).items()):

        assert isinstance(name, str)
        # Steps A-D
        if name in InstanceTree.BUILTIN_TYPES and prefix:
            flat_name = prefix
        elif prefix:
            flat_name = prefix + "." + name
        else:
            flat_name = name
        flat_symbol: InstanceSymbol = copy.copy(symbol)
        # dimensions inner lists are shared with the parsed AST via
        # _copy_symbol_contents; _resolve_dimensions mutates dim_list[i] in-place,
        # so make fresh inner lists to avoid corrupting the shared parsed AST.
        if flat_symbol.dimensions:
            flat_symbol.dimensions = [list(dl) for dl in flat_symbol.dimensions]
        # Strip input/output prefixes from nested symbols — these are
        # connector-local causality markers (MLS 9.1.1) and should not
        # propagate to the parent model's flattened namespace.
        if prefix:
            flat_symbol.prefixes = [p for p in flat_symbol.prefixes if p not in ("input", "output")]
        if symbol.type.name in InstanceTree.BUILTIN_TYPES:
            flat_symbol.name = flat_name
            flat_class.symbols[flat_name] = flat_symbol
        elif isinstance(flat_symbol.type, InstanceClass) and ast.is_enumeration(flat_symbol.type):
            # Enumeration-typed symbols are simple leaf types (MLS 4.8.5), treated
            # symmetrically with builtin types: place directly in the flat namespace.
            flat_symbol.name = flat_name
            flat_class.symbols[flat_name] = flat_symbol
        elif not isinstance(flat_symbol.type, InstanceClass):
            # scope==instance (declaring class) already differs from flat_class (root); no
            # derived-vs-base asymmetry here, so name_flat_class is not needed.
            resolved = _resolve_name(
                flat_symbol.type,
                instance,
                flat_class,
                guard=guard,
                opts=opts,
            )
            is_class = isinstance(resolved, InstanceClass)
            flat_symbol.type = resolved if is_class else resolved.parent  # type: ignore[assignment]
            # If the resolved type is an enumeration, add the symbol here now that
            # the type is known (covers the ComponentRef → InstanceClass resolution path).
            if ast.is_enumeration(flat_symbol.type) and flat_name not in flat_class.symbols:
                flat_symbol.name = flat_name
                flat_class.symbols[flat_name] = flat_symbol

        # 1.2 Evaluate conditional declarations (TODO)
        _evaluate_conditional_declarations(flat_symbol, flat_class)

        # 1.3 Resolve dimensions, including enclosing instances
        _resolve_dimensions(
            flat_symbol,
            instance,
            flat_class,
            guard=guard,
            opts=opts,
        )

        # 1.4 Resolve modifications of value attributes of simple types and records
        # 1.5 Resolve modifications of other attributes of simple types
        if (
            flat_symbol.type.name in InstanceTree.BUILTIN_TYPES
            or ast.is_enumeration(flat_symbol.type)
            or "record" in flat_symbol.prefixes
        ):
            _resolve_modifications(
                flat_symbol,
                flat_class,
                guard=guard,
                opts=opts,
            )
        else:
            # 1.6 Recursively "handle" non-simple types
            symbols_before_recursive = set(flat_class.symbols.keys())
            assert isinstance(flat_symbol.type, InstanceClass)
            _flatten_instance(
                flat_symbol.type,
                flat_class,
                guard=guard,
                opts=opts,
                prefix=flat_name,
            )

            # Propagate outer symbol dimensions to inner symbols (MLS 5.6.2 step 1.3)
            # Always propagate (including scalar [[None]]) to preserve nesting depth
            # for backend indexing.  Only propagate to symbols added by this recursive
            # call to avoid double-prepending when extends chains share component names.
            outer_dims = flat_symbol.dimensions
            flat_name_prefix = flat_name + "."
            new_sym_names = set(flat_class.symbols.keys()) - symbols_before_recursive
            for sym_name in new_sym_names:
                if sym_name.startswith(flat_name_prefix):
                    sym = flat_class.symbols[sym_name]
                    sym.dimensions = outer_dims + sym.dimensions

            # Propagate outer symbol prefixes and replaceable to the leaf builtin symbol
            inner_sym = flat_class.symbols.get(flat_name)
            if inner_sym is not None and flat_name in new_sym_names:
                for pfx in flat_symbol.prefixes:
                    if pfx not in ("input", "output") and pfx not in inner_sym.prefixes:
                        inner_sym.prefixes.insert(0, pfx)
                inner_sym.replaceable = flat_symbol.replaceable
                inner_sym.final = flat_symbol.final
                inner_sym.inner = flat_symbol.inner
                inner_sym.outer = flat_symbol.outer

    # 1.8 Recursively "handle" unnamed extends instances.
    # Extends must be processed before equations (step 1.7) so that inherited symbols
    # are present when equations of the current class are resolved.
    for extends in instance.extends:  # type: ignore[attr-defined]
        if not isinstance(extends, InstanceClass):
            continue
        _flatten_instance(
            extends,
            flat_class,
            guard=guard,
            opts=opts,
            prefix=prefix,
        )

    # 1.7 Resolve references in equations and algorithms
    _collect_and_resolve_equations(instance, flat_class, prefix)

    # Steps 1.9, 2, and 3 are done outside the recursion in the caller


def _evaluate_conditional_declarations(symbol: InstanceSymbol, parent: ast.Class):
    # TODO: 1.2 Evaluate the conditional declaration expression
    pass


def _resolve_dimensions(
    symbol: InstanceSymbol,
    instance: InstanceClass,
    flat_class: InstanceClass,
    *,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> None:
    """Resolve parametric array dimensions to concrete integers.

    Each element in symbol.dimensions is a list of dimension specifiers.
    ComponentRef and Expression elements are resolved to Primary(value=int(...)).
    """
    for dim_list in symbol.dimensions:
        for i, elem in enumerate(dim_list):
            if isinstance(elem, ast.Primary) or isinstance(elem, ast.Slice):
                continue
            elif isinstance(elem, ast.ComponentRef):
                # Dimension refs are always declared in the same class (instance);
                # no base-vs-derived asymmetry, so name_flat_class is not needed.
                resolved = _resolve_name(
                    elem,
                    instance,
                    flat_class,
                    guard=guard,
                    opts=opts,
                )
                if isinstance(resolved, InstanceSymbol) and isinstance(
                    resolved.value, (int, float)
                ):
                    dim_list[i] = ast.Primary(value=int(resolved.value))
            elif isinstance(elem, ast.Expression):
                try:
                    result = _resolve_expression(
                        elem,
                        instance,
                        flat_class,
                        guard=guard,
                        opts=opts,
                    )
                except (NotImplementedError, ModelicaSemanticError, NameLookupError):
                    result = None
                if isinstance(result, (int, float)):
                    dim_list[i] = ast.Primary(value=int(result))


def _resolve_modifications(
    symbol: InstanceSymbol,
    flat_class: InstanceClass,
    *,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> None:
    """Resolve modifications of a symbol
    :param symbol: The symbol to resolve modifications for
    :param flat_class: The flattened class
    :return: None
    """
    # TODO: Resolve modifications of records
    if "record" in symbol.prefixes:
        raise NotImplementedError("Record modifications not implemented yet")

    # TODO: Instantiation should move type class modifications down to the builtin
    if isinstance(symbol.type, ast.ComponentRef):
        # type class, modifications at this level
        modification_environment = symbol.modification_environment  # type: ignore[union-attr]
    elif ast.is_enumeration(symbol.type):
        # Enum types have no value-attribute modifications to inherit from the type class
        modification_environment = symbol.modification_environment  # type: ignore[union-attr]
    else:
        assert symbol.type.name is not None
        modification_environment = cast(
            InstanceSymbol, symbol.type.symbols[symbol.type.name]
        ).modification_environment

    for arg in modification_environment.arguments:
        assert isinstance(arg.value, ast.ElementModification)
        if arg.value.component == "value" and "record" in symbol.prefixes:
            # TODO: record value modification
            continue
        _resolve_modification_attribute(
            symbol,
            arg,
            flat_class=flat_class,
            guard=guard,
            opts=opts,
        )


def _resolve_modification_attribute(
    symbol: InstanceSymbol,
    arg: ast.ClassModificationArgument,
    *,
    flat_class: InstanceClass,
    guard: RecursionGuard,
    opts: LookupOptions,
):
    # MLS 5.6.2 step 1.4 turns value modifications on non-parameter/non-constant
    # simple-type variables into equations. Here we only set the resolved value on the
    # symbol (`setattr` below). `_generate_value_equations` emits equations in a second
    # pass after all recursion completes because a value expression may reference a
    # not-yet-processed symbol and each `_flatten_instance` recursion can mutate
    # `flat_class.symbols`.
    mod = cast(ast.ElementModification, arg.value)
    value = mod.modifications[0]
    assert not isinstance(value, ast.ClassModification)
    if isinstance(value, ast.Primary):
        value = value.value
    elif isinstance(value, ast.ComponentRef):
        assert isinstance(symbol.parent, InstanceClass)
        # 'time' is a Modelica built-in variable (MLS §2.7); leave it as-is.
        if value.name == "time" and not value.child:
            pass
        else:
            assert symbol.parent_instance is not None
            value = _resolve_name(
                value,
                arg.scope,  # type: ignore[arg-type]  # may be raw Class; _resolve_name walks tree
                symbol.parent_instance,
                name_flat_class=flat_class,
                guard=guard,
                opts=opts,
            )
            value = cast(InstanceSymbol, value)
            const_val = _get_constant_value(value)
            if const_val is not None and isinstance(const_val, ast.Primary):
                value = const_val.value
    elif isinstance(value, (ast.Array, ast.Expression)):
        # Discover function calls inside Array elements and Expressions so
        # that the function flattening pass can include them in the output.
        # Without this, functions referenced only in modifications are lost.
        _fn_scope = arg.scope if isinstance(arg.scope, InstanceClass) else symbol.parent_instance
        if isinstance(_fn_scope, InstanceClass):
            func_resolver = _FunctionCallResolver(_fn_scope, flat_class.functions)
            walker = TreeWalker()
            walker.walk(func_resolver, value)
    if isinstance(value, ast.Expression):
        expr_scope = arg.scope if isinstance(arg.scope, InstanceClass) else symbol.parent_instance
        expr_flat_class = symbol.parent_instance
        if isinstance(expr_scope, InstanceClass) and isinstance(expr_flat_class, InstanceClass):
            # Try to evaluate constant expressions (e.g. +1 → 1, -1.0 → -1.0);
            # keep the original Expression if evaluation fails (e.g. references
            # to non-constant variables or unsupported operators).
            try:
                result = _resolve_expression(
                    value,
                    expr_scope,
                    expr_flat_class,
                    guard=guard,
                    opts=opts,
                    name_flat_class=flat_class,
                )
                if result is not None:
                    value = result
            except (NotImplementedError, ModelicaSemanticError, NameLookupError):
                pass  # keep original ast.Expression
            # When the expression cannot be folded to a scalar, rewrite any
            # ComponentRef operands to their flat names so that the stored
            # Expression is consistent with the rest of the flattened tree.
            # This mirrors the name_flat_class treatment in the ComponentRef branch
            # above (MLS §5.6.2 point B).
            if isinstance(value, ast.Expression):
                value = _flatten_expression_crefs(
                    value,
                    expr_scope,
                    expr_flat_class,
                    flat_class,
                    guard,
                    opts,
                )

    setattr(symbol, mod.component.name, value)


class ExpressionEvaluator(TreeListener):
    """Calculate an ast.Expression tree containing only Primary and ComponentRef operands

    TODO: Ensure expression evaluation is according to Modelica spec"""

    # Unary and binary operator callables for Modelica → Python dispatch.
    # Includes built-in mathematical functions that can appear as Expression operators
    # in dimension expressions (e.g. n = div(shift, resolution)).
    unary_operator = {
        "+": operator.pos,
        "-": operator.neg,
        "not": operator.not_,
        # Modelica built-in scalar functions (MLS §3.7.1)
        "abs": abs,
        "integer": math.floor,  # integer(x) = floor(x) cast to Integer (MLS §3.7.1)
        "floor": math.floor,
        "ceil": math.ceil,
        "sqrt": math.sqrt,
    }
    binary_operator = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "^": operator.pow,
        "and": operator.and_,
        "or": operator.or_,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "==": operator.eq,
        "<>": operator.ne,
        # Modelica built-in integer/real functions (MLS §3.7.1)
        "div": lambda x, y: int(math.trunc(x / y)),  # truncated integer division
        "mod": lambda x, y: x - math.floor(x / y) * y,  # MLS mod (floor-based)
        "rem": lambda x, y: x - int(math.trunc(x / y)) * y,  # MLS rem (truncation-based)
        "max": max,
        "min": min,
    }

    def __init__(
        self,
        scope: InstanceClass,
        flat_class: InstanceClass,
        guard: RecursionGuard,
        opts: LookupOptions,
        name_flat_class: InstanceClass | None = None,
    ):
        self.scope = scope
        self.flat_class = flat_class
        self.guard = guard
        self.opts = opts
        self.name_flat_class = name_flat_class

        self.result = None
        super().__init__()

    def enterExpression(self, tree: ast.Expression):
        op_str = str(tree.operator)
        operands = []
        for operand in tree.operands:
            if isinstance(operand, ast.ComponentRef):
                operand = _resolve_name(
                    operand,
                    self.scope,
                    self.flat_class,
                    guard=self.guard,
                    opts=self.opts,
                    name_flat_class=self.name_flat_class,
                )
            if isinstance(operand, InstanceSymbol):
                const_val = _get_constant_value(operand)
                if const_val is not None and isinstance(const_val, ast.Primary):
                    operand = const_val
                elif self.opts.evaluate_parameters and "parameter" in operand.prefixes:
                    param_val = _get_parameter_value_from_chain(operand)
                    if param_val is not None:
                        operand = param_val
                # else: fall through with InstanceSymbol; arithmetic TypeError becomes
                # ModelicaSemanticError in the try/except below, keeping the expression
            elif not isinstance(operand, ast.Primary):
                raise NotImplementedError(f"Expression operand type not implemented: {operand!r}")
            operands.append(operand)

        try:
            if len(operands) == 1:
                op_func = self.unary_operator.get(op_str)
                if op_func is None:
                    raise NotImplementedError(f"Unsupported unary operator {op_str!r} in {tree!r}")
                self.result = op_func(operands[0].value)
            else:
                op_func = self.binary_operator.get(op_str)
                if op_func is None:
                    raise NotImplementedError(f"Unsupported binary operator {op_str!r} in {tree!r}")
                self.result = op_func(operands[0].value, operands[1].value)
        except (NotImplementedError, ModelicaSemanticError):
            raise
        except Exception as exc:
            raise ModelicaSemanticError(
                f"Unable to evaluate expression: {op_str!r} on {[o.value for o in operands]}"
            ) from exc


def _flatten_expression_crefs(
    expr: ast.Expression,
    scope: InstanceClass,
    flat_class: InstanceClass,
    name_flat_class: InstanceClass,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> ast.Expression:
    """Return a deep copy of expr with ComponentRef operands rewritten to flat names.

    Mirrors the name_flat_class treatment in _resolve_name: each ComponentRef is
    resolved in scope and renamed relative to name_flat_class (MLS §5.6.2 point B).
    Operands that cannot be resolved are left unchanged.
    """
    result = copy.deepcopy(expr)
    _rewrite_expression_crefs(result, scope, flat_class, name_flat_class, guard, opts)
    return result


def _rewrite_expression_crefs(
    expr: ast.Expression,
    scope: InstanceClass,
    flat_class: InstanceClass,
    name_flat_class: InstanceClass,
    guard: RecursionGuard,
    opts: LookupOptions,
) -> None:
    for operand in expr.operands:
        if isinstance(operand, ast.ComponentRef):
            try:
                resolved = _resolve_name(
                    operand,
                    scope,
                    flat_class,
                    name_flat_class=name_flat_class,
                    guard=guard,
                    opts=opts,
                )
                assert resolved.name is not None
                operand.name = resolved.name
                operand.child = []
            except Exception:
                pass  # leave un-resolvable refs (builtins, for-indices) unchanged
        elif isinstance(operand, ast.Expression):
            _rewrite_expression_crefs(operand, scope, flat_class, name_flat_class, guard, opts)


def _resolve_expression(
    expr: ast.Expression,
    scope: InstanceClass,
    flat_class: InstanceClass,
    *,
    guard: RecursionGuard,
    opts: LookupOptions,
    name_flat_class: InstanceClass | None = None,
) -> int | float | bool | str | None:
    """Calculate the given expression or return None if not possible"""
    assert isinstance(expr, ast.Expression)
    listener = ExpressionEvaluator(
        scope=scope,
        flat_class=flat_class,
        guard=guard,
        opts=opts,
        name_flat_class=name_flat_class,
    )
    walker = TreeWalker()
    walker.walk(listener, expr)
    return listener.result


class _EquationRefResolver(TreeListener):
    """Resolve ComponentRef names in equations to flattened dot-separated names.

    Follows the same depth/cutoff pattern as legacy ComponentRefFlattener:
    when a ComponentRef resolves to a known flat symbol, rewrite it in place;
    when it doesn't (builtins like ``time``, function names, for-loop indices),
    skip it and all its children.
    """

    def __init__(self, flat_class: InstanceClass, prefix: str):
        self.flat_class = flat_class
        self.prefix = prefix
        self.depth = 0
        self.cutoff_depth = sys.maxsize
        super().__init__()

    def reset(self):
        self.depth = 0
        self.cutoff_depth = sys.maxsize

    def enterComponentRef(self, tree: ast.ComponentRef):
        self.depth += 1
        if self.depth > self.cutoff_depth:
            return

        # Compose candidate flat name: prefix.name.child1.child2...
        if self.prefix:
            new_name = self.prefix + "." + tree.name
        else:
            new_name = tree.name
        c = tree
        while len(c.child) > 0:
            assert len(c.child) <= 1
            c = c.child[0]
            new_name += "." + c.name

        if new_name in self.flat_class.symbols:
            tree.name = new_name
            # Merge child indices into parent
            c = tree
            while len(c.child) > 0:
                assert len(c.child) <= 1
                c = c.child[0]
                tree.indices += c.indices
            tree.child = []
        else:
            # Not a known symbol — leave alone (builtin, function, for-index, etc.)
            self.cutoff_depth = self.depth

    def exitComponentRef(self, tree: ast.ComponentRef):
        self.depth -= 1
        if self.depth < self.cutoff_depth:
            self.cutoff_depth = sys.maxsize


class _FunctionCallResolver(TreeListener):
    """Discover function calls in equations/statements, resolve to fully-scoped names."""

    def __init__(self, instance: InstanceClass, functions: OrderedDict):
        self.instance = instance
        self.functions = functions
        super().__init__()

    def exitExpression(self, tree: ast.Expression):
        if not isinstance(tree.operator, ast.ComponentRef):
            return
        try:
            found = _find_name(self.instance, tree.operator, RecursionGuard(), LookupOptions())
        except Exception:
            return
        if found is None or not isinstance(found, ast.Class):
            return  # built-in function (sin, cos, etc.) or symbol
        full_name = found.full_name
        tree.operator = full_name
        self.functions[full_name] = found


def _flatten_connect_ref(ref: ast.ComponentRef, prefix: str) -> None:
    """Flatten a connect clause ComponentRef by prepending prefix and collapsing children."""
    parts = [ref.name]
    c = ref
    last_indices = ref.indices
    while len(c.child) > 0:
        assert len(c.child) <= 1
        c = c.child[0]
        parts.append(c.name)
        last_indices = c.indices
    ref.name = prefix + "." + ".".join(parts) if prefix else ".".join(parts)
    ref.indices = last_indices
    ref.child = []


def _is_inner_connector(ref: ast.ComponentRef, instance: InstanceClass) -> bool:
    """Determine if a connect ref is 'inner' (belongs to a sub-component) per MLS 9.2.

    A connector is 'inner' if the first name in the ref resolves to a non-connector
    component (i.e., the connector is a port of that component). It is 'outer' if the
    first name resolves directly to a connector declared in the current class.
    """
    first_name = ref.name
    sym = instance.symbols.get(first_name)
    if sym is None:
        return False
    if isinstance(sym.type, InstanceClass) and sym.type.type == "connector" and len(ref.child) == 0:
        # Bare connector name in this scope → outer
        return False
    # Component with sub-connector (or deeper path) → inner
    return True


def _collect_and_resolve_equations(
    instance: InstanceClass,
    flat_class: InstanceClass,
    prefix: str,
) -> None:
    """Deep-copy equations from *instance*, resolve refs, append to *flat_class*."""
    walker = TreeWalker()
    resolver = _EquationRefResolver(flat_class, prefix)
    func_resolver = _FunctionCallResolver(instance, flat_class.functions)

    # Snapshot lists before iteration to avoid mutation issues if the
    # instance's lists are modified during extends processing.
    equations = instance.equations
    initial_equations = instance.initial_equations
    statements = instance.statements
    initial_statements = instance.initial_statements

    for eq in equations:
        eq_copy = copy.deepcopy(eq)
        if isinstance(eq_copy, ast.ConnectClause):
            # Annotate inner/outer per MLS 9.2 (before flattening collapses children)
            assert isinstance(eq, ast.ConnectClause)
            eq_copy._left_inner = _is_inner_connector(cast(ast.ComponentRef, eq.left), instance)
            eq_copy._right_inner = _is_inner_connector(cast(ast.ComponentRef, eq.right), instance)
            # Manually flatten connect refs (connector names aren't in flat_class.symbols)
            _flatten_connect_ref(eq_copy.left, prefix)
            _flatten_connect_ref(eq_copy.right, prefix)
        else:
            walker.walk(func_resolver, eq_copy)
            resolver.reset()
            walker.walk(resolver, eq_copy)
        flat_class.equations.append(eq_copy)

    for eq in initial_equations:
        eq_copy = copy.deepcopy(eq)
        walker.walk(func_resolver, eq_copy)
        resolver.reset()
        walker.walk(resolver, eq_copy)
        flat_class.initial_equations.append(eq_copy)

    for stmt in statements:
        stmt_copy = copy.deepcopy(stmt)
        walker.walk(func_resolver, stmt_copy)
        resolver.reset()
        walker.walk(resolver, stmt_copy)
        flat_class.statements.append(stmt_copy)

    for stmt in initial_statements:
        stmt_copy = copy.deepcopy(stmt)
        walker.walk(func_resolver, stmt_copy)
        resolver.reset()
        walker.walk(resolver, stmt_copy)
        flat_class.initial_statements.append(stmt_copy)


def _flatten_discovered_functions(flat_class: InstanceClass) -> None:
    """Flatten discovered function classes and add to flat_class.functions."""
    functions = flat_class.functions
    processed = set()
    while set(functions.keys()) - processed:
        for full_name in list(functions.keys()):
            if full_name in processed:
                continue
            func_class = functions[full_name]

            # Instantiate if needed
            if (
                not isinstance(func_class, InstanceClass)
                or func_class.instantiation_state < InstantiationState.FULL
            ):
                func_instance_raw = _instantiate_element(
                    func_class,
                    func_class.parent,  # type: ignore[arg-type]
                    ast.ClassModification(),
                )
                func_instance = cast(InstanceClass, func_instance_raw)
            else:
                func_instance = func_class

            # Flatten using new pipeline — discovers nested function calls
            flat_func = _create_partial_flat_instance(func_instance)
            _flatten_instance(func_instance, flat_func)

            # Merge nested discoveries back
            for nested_name, nested_class in flat_func.functions.items():
                if nested_name not in functions:
                    functions[nested_name] = nested_class

            flat_class.functions[full_name] = flat_func
            processed.add(full_name)


def _check_all_references_valid(flat_class: InstanceClass):
    # TODO: 1.9 Check that all references now point to a valid instance (not conditionally false)
    pass


def _generate_connect_equations(flat_class: InstanceClass):
    """Generate equations from connect clauses (MLS 9.2).

    For each ConnectClause, expand into variable-level equations:
    - Non-flow variables (no prefix, input, output): equality equations
    - Flow variables: Kirchhoff sum-to-zero equations
    - Constant/parameter: skipped
    Disconnected flow variables default to 0.
    """
    # Track disconnected flow variables
    disconnected_flow_variables = OrderedDict()
    for name, sym in flat_class.symbols.items():
        if "flow" in sym.prefixes:
            disconnected_flow_variables[name] = name

    flow_connections = OrderedDict()
    orig_equations = flat_class.equations[:]
    flat_class.equations = []

    for equation in orig_equations:
        if not isinstance(equation, ast.ConnectClause):
            flat_class.equations.append(equation)
            continue

        left_inner = getattr(equation, "_left_inner", False)
        right_inner = getattr(equation, "_right_inner", False)
        left_name = equation.left.name
        right_name = equation.right.name

        # Find connector variables by prefix in flat_class.symbols
        left_prefix = left_name + "."
        connector_vars = []
        for sym_name, sym in flat_class.symbols.items():
            if sym_name.startswith(left_prefix):
                suffix = sym_name[len(left_prefix) :]
                type_indices = getattr(sym.type, "indices", [])
                connector_vars.append((suffix, sym.prefixes, type_indices))

        if not connector_vars:
            # Elementary type connection (e.g., connecting Reals directly)
            flat_class.equations.append(ast.Equation(left=equation.left, right=equation.right))
            continue

        for suffix, prefixes, type_indices in connector_vars:
            l_name = left_name + "." + suffix
            r_name = right_name + "." + suffix
            left = ast.ComponentRef(
                name=l_name,
                indices=equation.left.indices + type_indices,
            )
            right = ast.ComponentRef(
                name=r_name,
                indices=equation.right.indices + type_indices,
            )

            if len(prefixes) == 0 or prefixes[0] in ["input", "output"]:
                flat_class.equations.append(ast.Equation(left=left, right=right))
            elif "flow" in prefixes:
                # Build flow connection groups for sum-to-zero equations
                left_key = (
                    l_name,
                    tuple(
                        i.value if isinstance(i, ast.Primary) else str(i)
                        for index_array in left.indices
                        for i in index_array
                        if i is not None
                    ),
                    left_inner,
                )
                right_key = (
                    r_name,
                    tuple(
                        i.value if isinstance(i, ast.Primary) else str(i)
                        for index_array in right.indices
                        for i in index_array
                        if i is not None
                    ),
                    right_inner,
                )

                left_connected = flow_connections.get(left_key, OrderedDict())
                right_connected = flow_connections.get(right_key, OrderedDict())

                left_connected.update(right_connected)
                connected = left_connected
                connected[left_key] = (left, left_inner)
                connected[right_key] = (right, right_inner)

                for k in connected:
                    flow_connections[k] = connected

                disconnected_flow_variables.pop(l_name, None)
                disconnected_flow_variables.pop(r_name, None)
            elif prefixes[0] in ["constant", "parameter"]:
                pass
            else:
                raise NotImplementedError(
                    "Unsupported connector variable prefixes {}".format(prefixes)
                )

    # Generate flow sum-to-zero equations
    processed: set[int] = set()
    for connected_variables in flow_connections.values():
        if id(connected_variables) not in processed:
            processed.add(id(connected_variables))
            operand_specs = list(connected_variables.values())
            if all(not op_spec[1] for op_spec in operand_specs):
                # All outer variables — no sign inversion needed
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
            flat_class.equations.append(ast.Equation(left=expr, right=ast.Primary(value=0)))

    # Disconnected flow variables default to 0
    for name in disconnected_flow_variables.values():
        flat_class.equations.append(
            ast.Equation(left=ast.ComponentRef(name=name), right=ast.Primary(value=0))
        )


def _flatten_value_ref_names(flat_class: InstanceClass) -> None:
    """Rewrite InstanceSymbol references in symbol value attributes to flat names.

    Per MLS 5.6.2, value modifications are resolved (in source scope) before the
    enclosing extends clauses are processed.  When a value references an inherited
    symbol, ``_resolve_name`` must therefore fall back to the class tree — producing
    an InstanceSymbol whose ``.name`` is the full class path (e.g. ``Pkg.A.x``)
    rather than the flat name (``x``), because the extends instance isn't yet
    registered in ``flat_class.symbols``.

    Once all extends have been processed, every referenced inherited symbol exists
    under its flat name.  Rewrite references by matching on ``ast_ref.full_name``
    so that downstream passes (``_to_ast_value`` and ``_generate_value_equations``)
    emit equations with flat names.
    """
    flat_name_by_ref = {}
    for flat_name, sym in flat_class.symbols.items():
        ref_name = getattr(getattr(sym, "ast_ref", None), "full_name", None)
        if ref_name is not None:
            flat_name_by_ref[ref_name] = flat_name

    for sym in flat_class.symbols.values():
        for attr in _VALUE_ATTRS:
            value = getattr(sym, attr, None)
            if not isinstance(value, InstanceSymbol):
                continue
            ref_name = getattr(getattr(value, "ast_ref", None), "full_name", None)
            flat_name = flat_name_by_ref.get(ref_name)
            if flat_name is not None and value.name != flat_name:
                value.name = flat_name


def _get_parameter_value_from_chain(isym: InstanceSymbol) -> ast.Primary | None:
    """Walk an InstanceSymbol value chain to a literal, for inline parameter folding

    Returns None if literal value not found
    """
    val = isym.value
    seen = {id(isym)}
    while True:
        if val is None or (isinstance(val, ast.Primary) and val.value is None):
            return None
        if isinstance(val, ast.Primary):
            return val
        if isinstance(val, (int, float, bool, str)):
            return ast.Primary(value=val)
        if isinstance(val, InstanceSymbol):
            vid = id(val)
            if vid in seen:
                return None
            seen.add(vid)
            val = val.value
            continue
        return None


def _resolve_param_ref(name: str, flat_class: InstanceClass, seen: set[str]) -> ast.Primary | None:
    """Resolve a flat parameter/constant name to its literal value, following chains

    Returns None when the chain is unresolvable or cyclic
    """
    if name in seen:
        return None
    seen = seen | {name}
    flat_sym = flat_class.symbols.get(name)
    if flat_sym is None:
        return None
    return _fold_flat_value(flat_sym.value, flat_class, seen)


def _fold_flat_value(value, flat_class: InstanceClass, seen: set[str]) -> ast.Primary | None:
    """Fold an already-flattened value to an ast.Primary literal if possible

    Return None if any operand is unresolvable (e.g. references a non-constant variable
    or an unsupported operator).
    """
    if value is None or (isinstance(value, ast.Primary) and value.value is None):
        return None
    if isinstance(value, ast.Primary):
        return value
    if isinstance(value, (int, float, bool, str)):
        return ast.Primary(value=value)
    if isinstance(value, InstanceSymbol):
        return _resolve_param_ref(value.name, flat_class, seen)
    if isinstance(value, ast.ComponentRef) and not value.child:
        return _resolve_param_ref(value.name, flat_class, seen)
    if isinstance(value, ast.Expression):
        folded_operands: list[ast.Primary] = []
        for operand in value.operands:
            folded_operand = _fold_flat_value(operand, flat_class, seen)
            if folded_operand is None:
                return None
            folded_operands.append(folded_operand)
        operands = folded_operands
        op_str = str(value.operator)
        try:
            if len(operands) == 1:
                op_func = ExpressionEvaluator.unary_operator.get(op_str)
                if op_func is None:
                    return None
                return ast.Primary(value=op_func(operands[0].value))
            op_func = ExpressionEvaluator.binary_operator.get(op_str)
            if op_func is None:
                return None
            return ast.Primary(value=op_func(operands[0].value, operands[1].value))
        except Exception:
            return None
    return None


def _evaluate_parameter_values(flat_class: InstanceClass) -> None:
    """Fold parameter and constant symbol values to ast.Primary

    Unresolvable or cyclic references are left unchanged.
    """
    _FOLD_PREFIXES = frozenset({"parameter", "constant"})
    for sym in flat_class.symbols.values():
        if not (_FOLD_PREFIXES & set(sym.prefixes)):
            continue
        folded = _fold_flat_value(sym.value, flat_class, set())
        if folded is not None:
            sym.value = folded


def _generate_value_equations(flat_class: InstanceClass) -> None:
    """Convert resolved value modifications to equations (MLS 5.6.2 step 1.4).

    For each non-parameter/non-constant flat symbol with a resolved .value,
    emit an equation  sym = value  and clear sym.value to the sentinel.

    Parameters and constants retain their .value as a declaration binding
    (used by backends as the constant/parameter value).

    ComponentRef operands that remain in source scope (when expression
    evaluation failed) are flattened to dot-separated names by
    _EquationRefResolver using the symbol's own prefix.
    """
    _NON_EQUATION_PREFIXES = frozenset({"constant", "parameter"})
    walker = TreeWalker()
    for flat_name, sym in list(flat_class.symbols.items()):
        value = sym.value
        if value is None or (isinstance(value, ast.Primary) and value.value is None):
            continue
        if _NON_EQUATION_PREFIXES & set(sym.prefixes):
            continue

        rhs = copy.deepcopy(_to_ast_value(value))

        # Resolve any source-scope ComponentRefs remaining in rhs (happens when
        # _resolve_expression fails and keeps the original ast.Expression).
        # The prefix for source-scope refs inside this symbol's value equals the
        # instance path containing the symbol (everything before the last dot).
        prefix = flat_name.rsplit(".", 1)[0] if "." in flat_name else ""
        resolver = _EquationRefResolver(flat_class, prefix)
        walker.walk(resolver, rhs)

        flat_class.equations.append(
            ast.Equation(
                left=ast.ComponentRef(name=flat_name),
                right=rhs,
            )
        )
        sym.value = ast.Primary(value=None)


def _process_transitions(flat_class: InstanceClass):
    # TODO: 3. Process transitions in the flattened tree
    pass


def _flatten_name(
    element: InstanceElement,
    remove_prefix: str = "",
) -> str:
    """Flatten the instance path into a name str

    :param instance: The instance having the name to flatten
    :param remove_prefix: The prefix to remove from the name
    :return: The flattened name
    """
    assert isinstance(element, InstanceElement)
    element_full_name = element_instance_name_tuple(element)
    flat_name_start = 0
    for names in zip(element_full_name, remove_prefix.split(".")):
        if names[0] == names[1]:
            flat_name_start += 1
    return ".".join(element_full_name[flat_name_start:])


def _resolve_name(
    name: str | ast.ComponentRef,
    scope: InstanceClass,
    flat_class: InstanceClass,
    *,
    guard: RecursionGuard,
    opts: LookupOptions,
    name_flat_class: InstanceClass | None = None,
) -> InstanceClass | InstanceSymbol:
    """Resolve a name reference and return a flat-named element.

    Two contexts govern resolution, which usually coincide but can differ for
    modifications written in a base class that reference symbols in a derived class
    (MLS section 4.5.1 and 5.6):

    - ``scope``: where to look the name up (the syntactic scope where the reference
      appears, e.g. the base class containing a modification expression).
    - ``flat_class`` / ``name_flat_class``: the root model relative to which the
      returned element's flat path is computed.  ``flat_class`` is also where the
      element is cached/registered; ``name_flat_class`` overrides the naming root
      without touching ``flat_class.symbols``.

    Pass ``name_flat_class`` when ``scope`` and the flat-naming root differ — for
    example, when resolving a ComponentRef that appears in a modification value
    (``_resolve_modification_attribute``).  When ``name_flat_class`` is given the
    returned element is a clone with the correct flat name, but it is *not*
    registered in ``name_flat_class.symbols``; the element will be registered
    through its own definition site when it is processed in the normal flattening
    pass.
    """

    # Implements points A-C of the MLS 5.6.2 outline in _flatten_instance: resolve the
    # reference to the real instance, rename it to the global unique dot-separated
    # identifier, instantiating classes and symbols as needed along the way.

    # When scope is a raw ast.Class (not yet an InstanceClass), find the
    # corresponding InstanceClass by walking up the instance tree from
    # flat_class.  This happens for deeply-nested modifications whose scope
    # wasn't resolved to an InstanceClass during instantiation.
    if not isinstance(scope, InstanceClass):
        current = flat_class
        while current is not None:
            if isinstance(current, InstanceClass) and (
                current.ast_ref is scope or current.ast_ref.full_name == scope.full_name
            ):
                scope = current
                break
            current = getattr(current, "parent_instance", None)
        if not isinstance(scope, InstanceClass):
            raise NameLookupError(
                f"Unable to find instance for scope {scope.full_name} from {flat_class.full_name}"
            )

    # Step C: Fully instantiate the scope class if needed.
    # Unnamed extends (name='') must NOT be cached to avoid collision when a class
    # has multiple unnamed extends — two different extends would share the same key.
    # Named scopes are cached (update_parent_instance=True) for performance.
    if scope.instantiation_state < InstantiationState.FULL:
        scope = _instantiate_class(
            scope,
            ast.ClassModification(),
            scope.parent_instance,  # type: ignore[arg-type]
            guard=guard,
            opts=opts,
            update_parent_instance=bool(scope.name),
        )

    found = _find_name(
        scope,
        name,
        guard,
        LookupOptions(instantiate_in_place=opts.instantiate_in_place),
    )
    if found is None:
        raise NameLookupError(f"Unable to resolve {name} in scope {scope.full_name}")

    is_symbol = isinstance(found, ast.Symbol)

    if (
        not isinstance(found, InstanceElement)
        or found.instantiation_state < InstantiationState.FULL
    ):
        element = _instantiate_element(
            found,
            scope,
            ast.ClassModification(),
            guard=guard,
            opts=opts,
        )
    else:
        element = found

    # Deduplication against name_flat_class.symbols is intentionally skipped: entries
    # there may have been registered by _flatten_instance's direct path without updating
    # parent_instance, so returning them would give the wrong parent for
    # modification-value clones.  Multiple modifications referencing the same symbol
    # will receive distinct clones — safe as long as consumers key off the name string
    # rather than object identity (which is the current convention).
    if name_flat_class is not None:
        flat_name = _flatten_name(element, name_flat_class.full_instance_name)
        element = element.clone()
        element.name = flat_name
        element.parent_instance = name_flat_class
        element.parent = name_flat_class
        return cast(InstanceClass | InstanceSymbol, element)

    flat_name = _flatten_name(element, flat_class.full_instance_name)

    # Check to see if the name is already resolved
    if is_symbol and flat_name in flat_class.symbols:
        return cast(InstanceClass | InstanceSymbol, flat_class.symbols[flat_name])
    elif flat_name in flat_class.classes:
        return cast(InstanceClass | InstanceSymbol, flat_class.classes[flat_name])

    # Make copy so we can flatten name without changing originals.
    # Reparent under flat_class so full_instance_name / full_name reflect
    # the flat path (flat_name already encodes the ancestor components).
    element = element.clone()
    element.name = flat_name
    element.parent_instance = flat_class
    element.parent = flat_class
    if is_symbol:
        flat_class.symbols[flat_name] = cast(InstanceSymbol, element)  # type: ignore[assignment]
    else:
        flat_class.classes[flat_name] = cast(InstanceClass, element)  # type: ignore[assignment]

    return cast(InstanceClass | InstanceSymbol, element)


def _instantiate_element(
    element: ast.Class | ast.Symbol | InstanceClass | InstanceSymbol,
    scope: InstanceClass,
    modification_environment: ast.ClassModification,
    *,
    guard: RecursionGuard | None = None,
    opts: LookupOptions | None = None,
    update_parent_instance: bool = True,
    target_state: InstantiationState = InstantiationState.FULL,
) -> InstanceClass | InstanceSymbol:

    if guard is None:
        guard = RecursionGuard()
    if opts is None:
        opts = LookupOptions(instantiate_in_place=False)

    is_symbol = isinstance(element, ast.Symbol)

    if isinstance(element, InstanceElement):
        if is_symbol:
            # For symbols, class_to_instantiate (below) will be the
            # containing class (element.parent).  Its parent in the instance
            # tree is element.parent_instance.parent_instance, NOT
            # element.parent_instance (which IS the containing class).
            assert element.parent_instance is not None
            parent_instance = element.parent_instance.parent_instance
        else:
            parent_instance = element.parent_instance
        parent = element.parent
    else:
        element_class = element.parent if is_symbol else element
        assert element_class is not None
        parent = _get_lexical_parent_instance(
            element_class,
            scope,
            guard=guard,
            opts=opts,
        )
        parent_instance = parent

    class_to_instantiate = element.parent if is_symbol else element
    assert class_to_instantiate is not None
    assert parent_instance is not None
    instance = _instantiate_class(
        class_to_instantiate,
        ast.ClassModification(),
        parent_instance,
        guard=guard,
        opts=opts,
        update_parent_instance=update_parent_instance,
        target_state=target_state,
    )

    if is_symbol:
        return cast(InstanceSymbol, instance.symbols[element.name])
    else:
        return instance


# ---------------------------------------------------------------------------
# Adapter: InstanceClass → ast.Class / ast.Tree  (for backend compatibility)
# ---------------------------------------------------------------------------

_VALUE_ATTRS = (
    "start",
    "min",
    "max",
    "nominal",
    "value",
    "fixed",
    "unit",
    "quantity",
    "displayUnit",
)


def _get_constant_value(isym: InstanceSymbol):
    """Extract the resolved constant value from an InstanceSymbol, or None if unavailable."""
    if "constant" not in isym.prefixes:
        return None
    # Check if value was already resolved
    if isym.value is not None and not (
        isinstance(isym.value, ast.Primary) and isym.value.value is None
    ):
        return isym.value
    # Look in the type class's builtin symbol modification for the value
    if isinstance(isym.type, InstanceClass) and isym.type.name in InstanceTree.BUILTIN_TYPES:
        builtin_sym = isym.type.symbols.get(isym.type.name)
        if isinstance(builtin_sym, InstanceSymbol):
            for arg in builtin_sym.modification_environment.arguments:
                if (
                    isinstance(arg.value, ast.ElementModification)
                    and arg.value.component.name == "value"
                    and arg.value.modifications
                ):
                    mod_val = arg.value.modifications[0]
                    if isinstance(mod_val, ast.Primary):
                        return mod_val
    return None


def _to_ast_value(val):
    """Wrap a raw Python value or InstanceSymbol back into an AST node."""
    if isinstance(val, ast.Primary):
        return val
    if isinstance(val, (ast.Expression, ast.Array, ast.ComponentRef)):
        return val
    if isinstance(val, InstanceSymbol):
        # For constants, substitute the actual value (MLS 5.6.2)
        const_val = _get_constant_value(val)
        if const_val is not None:
            return const_val
        return ast.ComponentRef(name=val.name)
    if isinstance(val, (int, float, bool, str)) or val is None:
        return ast.Primary(value=val)
    return val


def _instance_to_ast_symbol(isym: InstanceSymbol) -> ast.Symbol:
    """Convert an InstanceSymbol to a plain ast.Symbol for TreeWalker compatibility."""
    sym = ast.Symbol()
    for attr in (
        "name",
        "type",
        "prefixes",
        "replaceable",
        "final",
        "inner",
        "outer",
        "dimensions",
        "comment",
        "id",
        "order",
        "visibility",
        "class_modification",
    ):
        setattr(sym, attr, getattr(isym, attr))
    # Wrap raw values back into ast.Primary for backend compatibility
    for attr in _VALUE_ATTRS:
        setattr(sym, attr, _to_ast_value(getattr(isym, attr)))
    # Normalize type to ComponentRef for TreeWalker compatibility
    if isinstance(sym.type, InstanceClass):
        sym.type = ast.ComponentRef(name=sym.type.name)
    return sym


def _instance_to_ast_class(flat_instance: InstanceClass) -> ast.Class:
    """Convert a flat InstanceClass to a plain ast.Class for backend consumption."""
    flat_class = ast.Class()
    flat_class.name = flat_instance.name
    flat_class.type = flat_instance.type
    flat_class.comment = flat_instance.comment
    flat_class.annotation = flat_instance.annotation

    # Convert symbols
    for name, isym in flat_instance.symbols.items():
        sym = _instance_to_ast_symbol(cast(InstanceSymbol, isym))
        sym.parent = flat_class
        flat_class.symbols[name] = sym

    # Equations are plain AST nodes — copy directly
    flat_class.equations = flat_instance.equations
    flat_class.initial_equations = flat_instance.initial_equations
    flat_class.statements = flat_instance.statements
    flat_class.initial_statements = flat_instance.initial_statements

    # Functions
    for fname, func in flat_instance.functions.items():
        if isinstance(func, InstanceClass):
            flat_class.functions[fname] = _instance_to_ast_class(func)
        else:
            flat_class.functions[fname] = func

    return flat_class


def _add_connector_symbols(
    instance: InstanceClass,
    flat_class: ast.Class,
    prefix: str,
) -> None:
    """Walk the instance tree and add connector-level symbols to *flat_class*.

    ``expand_connectors`` expects a symbol entry for each connector (e.g.
    ``a.up``) with a ``_connector_type`` attribute containing a flat
    ``ast.Class`` of the connector's variables.  The new flattening only
    produces leaf symbols (``a.up.H``, ``a.up.Q``), so we recreate the
    intermediate entries here.
    """
    for name, sym in instance.symbols.items():
        if not isinstance(sym.type, InstanceClass):
            continue
        full_name = prefix + "." + name if prefix else name
        if sym.type.type in ("connector", "expandableconnector"):
            # Create a stub symbol for this connector
            stub = ast.Symbol()
            stub.name = full_name
            stub.type = ast.ComponentRef(name=sym.type.name)
            stub.prefixes = list(sym.prefixes)
            stub.comment = sym.comment
            # Flatten the connector type to get its variables
            connector_flat = flatten_instance(sym.type)
            stub._connector_type = _instance_to_ast_class(connector_flat)
            flat_class.symbols[full_name] = stub
        # Recurse into composite types (models containing connectors)
        _add_connector_symbols(sym.type, flat_class, full_name)
    # Recurse into extends instances (connectors inherited from base classes)
    for extends in instance.extends:
        if isinstance(extends, InstanceClass):
            _add_connector_symbols(extends, flat_class, prefix)


def flatten_class(
    root: ast.Tree,
    class_name: str | ast.ComponentRef,
    *,
    expand_connect: bool = True,
    evaluate_parameters: bool = False,
) -> InstanceClass:
    """Instantiate and flatten class_name, returning the flat InstanceClass."""
    instance = instantiate(root, str(class_name))
    return flatten_instance(
        instance,
        expand_connect=expand_connect,
        evaluate_parameters=evaluate_parameters,
    )


def flatten_to_tree(root: ast.Tree, class_name: ast.ComponentRef) -> ast.Tree:
    """Flatten a class using the new instantiation/flattening pipeline,
    returning an ast.Tree compatible with legacy backends.

    Plug-compatible with legacy ``flatten(root, class_name)``.
    """
    # 1. Instantiate
    class_name_str = str(class_name)
    instance = instantiate(root, class_name_str)

    # 2. Flatten (defer connect expansion to expand_connectors below)
    flat_instance = flatten_instance(instance, expand_connect=False)

    # 3. Convert InstanceClass → ast.Class
    flat_class = _instance_to_ast_class(flat_instance)

    # 4. Add connector-level symbols with _connector_type for expand_connectors.
    #    Must happen before the ComponentRef walk so connector names (e.g. plug_p)
    #    are present in flat_class.symbols and equation refs like plug_p.pin get
    #    resolved.  The new flattening recurses into connectors producing leaf
    #    symbols (e.g. a.up.H, a.up.Q) but expand_connectors expects an
    #    intermediate symbol for each connector (e.g. a.up) with _connector_type.
    _add_connector_symbols(instance, flat_class, prefix="")

    # 5. Resolve component refs in symbol attributes (e.g. value = 2 * p1 → 2 * nested.p1).
    # Symbols are fresh objects from _instance_to_ast_class, but their value attributes
    # (start, min, max, value, …) come from _to_ast_value which returns ComponentRef /
    # Expression / Array objects BY REFERENCE from the parsed AST.  ComponentRefFlattener
    # mutates those objects in-place, which would corrupt the shared parsed AST and break
    # subsequent flattenings of other models.  Deepcopy only the non-trivial attrs before
    # walking; Primary / scalar values are immutable and safe to share.
    # _connector_type on connector stubs must be hidden from the walker: it holds
    # an ast.Class whose ComponentRefs are shared with the parsed AST and must not
    # be mutated.  Strip and restore around the walk.
    connector_types = {}
    for sym_name, sym in flat_class.symbols.items():
        ct = sym.__dict__.pop("_connector_type", None)
        if ct is not None:
            connector_types[sym_name] = ct
    w = TreeWalker()
    for sym_name, sym in flat_class.symbols.items():
        prefix = sym_name.rsplit(".", 1)[0] + "." if "." in sym_name else ""
        for attr in _VALUE_ATTRS:
            val = getattr(sym, attr, None)
            if not isinstance(val, (ast.Primary, type(None), int, float, bool, str)):
                setattr(sym, attr, copy.deepcopy(val))
        if sym.dimensions:
            sym.dimensions = copy.deepcopy(sym.dimensions)
        w.walk(ComponentRefFlattener(flat_class, prefix), sym)
    for sym_name, ct in connector_types.items():
        flat_class.symbols[sym_name]._connector_type = ct

    # 6. Post-processing (same order as legacy flatten)
    expand_connectors(flat_class)
    for func in flat_class.functions.values():
        add_variable_value_statements(func)
    annotate_states(flat_class)

    # 6. Build ast.Tree
    out = ast.Tree()
    # Drop the builtin stub classes ast.Tree pre-populates for name lookup: the flat
    # output needs only the flat class and its functions, like the legacy flatten output
    out.classes.clear()
    flat_name = class_name_str
    flat_class.name = flat_name
    out.classes[flat_name] = flat_class

    # Pull functions to top level (before the model class)
    functions_and_classes = flat_class.functions
    flat_class.functions = OrderedDict()
    functions_and_classes.update(out.classes)
    out.classes = functions_and_classes

    return out


# ---------------------------------------------------------------------------
# Post-processing helpers (used by flatten_to_tree above)
# ---------------------------------------------------------------------------


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
            new_name += "." + c.name

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

            class_left = getattr(sym_left, "_connector_type", None)
            class_right = getattr(sym_right, "_connector_type", None)
            if class_left is None or class_right is None:
                # Elementary connect (no connector type — e.g. Real): emit equation directly.
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

                for connector_variable in class_left.symbols.values():
                    left_name = equation.left.name + "." + connector_variable.name
                    right_name = equation.right.name + "." + connector_variable.name
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
                                i.value if isinstance(i, ast.Primary) else str(i)
                                for index_array in left.indices
                                for i in index_array
                                if i is not None
                            ),
                            getattr(equation, "_left_inner", False),
                        )
                        right_key = (
                            right_name,
                            tuple(
                                i.value if isinstance(i, ast.Primary) else str(i)
                                for index_array in right.indices
                                for i in index_array
                                if i is not None
                            ),
                            getattr(equation, "_right_inner", False),
                        )

                        left_connected_variables = flow_connections.get(left_key, OrderedDict())
                        right_connected_variables = flow_connections.get(right_key, OrderedDict())

                        left_connected_variables.update(right_connected_variables)
                        connected_variables = left_connected_variables
                        connected_variables[left_key] = (
                            left,
                            getattr(equation, "_left_inner", False),
                        )
                        connected_variables[right_key] = (
                            right,
                            getattr(equation, "_right_inner", False),
                        )

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
        if getattr(sym, "_connector_type", None) is not None:
            del node.symbols[i]


def add_variable_value_statements(node: ast.Class) -> None:
    for sym in node.symbols.values():
        if not (isinstance(sym.value, ast.Primary) and sym.value.value is None):
            node.statements.append(ast.AssignmentStatement(left=[sym], right=sym.value))
            sym.value = ast.Primary(value=None)


class StateAnnotator(TreeListener):
    """
    Finds all variables that are differentiated and annotates them with the state prefix
    """

    def __init__(self, node: ast.Class):
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

    def exitComponentRef(self, tree: ast.ComponentRef):
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


def annotate_states(node: ast.Class) -> None:
    """
    Finds all derivative expressions and annotates all differentiated
    symbols as states by adding state the prefix list
    :param node: node of tree to walk
    :return:
    """
    w = TreeWalker()
    w.walk(StateAnnotator(node), node)
