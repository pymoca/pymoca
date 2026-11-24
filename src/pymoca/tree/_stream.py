"""Stream connector semantics: inStream() and actualStream() expansion (MLS 15.2, 15.3)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeGuard

from . import ModelicaSemanticError
from ._connection_sets import (
    ConnectionSet,
    SetKey,
    align_groups,
    aligned_indices,
    set_key,
    sum_terms,
    unique_sets,
)
from ._listener import logger
from .. import ast

if TYPE_CHECKING:
    from .instance import InstanceClass

# Regularization of max(flow, 0) in the mixing formula (MLS 15.2)
EPS = 1e-15

_OPERATORS = ("inStream", "actualStream")


def expand_stream_operators(
    flat_class: InstanceClass, stream_connections: dict[SetKey, ConnectionSet]
) -> None:
    """Expand inStream() and actualStream(), and add the outside stream equations (MLS 15.2).

    Operators on connectors that also appear in connects left in for- or if-equations stay
    unexpanded, since their connection sets are incomplete.
    """
    nested = set()
    for eq in flat_class.equations:
        _collect_nested_connectors(eq, nested, top=True)
    expander = _StreamExpander(flat_class.symbols, stream_connections, nested)
    for name in ("equations", "initial_equations", "statements", "initial_statements"):
        setattr(flat_class, name, expander.rewrite(getattr(flat_class, name)))
    for sym in flat_class.symbols.values():
        for attr in ast.Symbol.ATTRIBUTES:
            value = getattr(sym, attr, None)
            if _has_stream_call(value):
                setattr(sym, attr, expander.rewrite(value))
    flat_class.equations.extend(expander.outside_equations())
    for name in sorted(expander.unexpanded):
        logger.warning(f"Stream operator on {name} left unexpanded: connection set not known")


def _has_stream_call(node) -> bool:
    if isinstance(node, list):
        return any(_has_stream_call(item) for item in node)
    if isinstance(node, ast.Expression):
        op = node.operator
        if isinstance(op, ast.ComponentRef) and op.name in _OPERATORS:
            return True
        return _has_stream_call(node.operands)
    if isinstance(node, ast.IfExpression):
        return _has_stream_call(node.conditions) or _has_stream_call(node.expressions)
    if isinstance(node, ast.Array):
        return _has_stream_call(node.values)
    if isinstance(node, ast.ForArray):
        return _has_stream_call(node.expression)
    return False


def _collect_nested_connectors(node, names: set[str], top: bool) -> None:
    if isinstance(node, ast.ConnectClause):
        if not top:
            names.update((node.left.name, node.right.name))
    elif isinstance(node, ast.ForEquation):
        for eq in node.equations:
            _collect_nested_connectors(eq, names, top=False)
    elif isinstance(node, (ast.IfEquation, ast.WhenEquation)):
        for block in node.blocks:
            for eq in block:
                _collect_nested_connectors(eq, names, top=False)


def _call(name: str, *operands) -> ast.Expression:
    return ast.Expression(operator=ast.ComponentRef(name=name), operands=list(operands))


def _with_last_index(ref: ast.ComponentRef, last: list) -> ast.ComponentRef:
    """Copy of a flat ref with the subscript of its last name part replaced."""
    new = copy.deepcopy(ref)
    new.indices = new.indices[:-1] + [copy.deepcopy(last)]
    return new


def _is_literal(subscript) -> TypeGuard[ast.Primary]:
    return isinstance(subscript, ast.Primary) and isinstance(subscript.value, int)


def _is_whole(subscript) -> bool:
    return subscript is None or (isinstance(subscript, ast.Primary) and subscript.value is None)


def _slice_values(subscript: ast.Slice, size: int) -> list[int] | None:
    """Element values of a slice subscript, None if a bound is not a literal."""
    bounds = []
    for bound, default in ((subscript.start, 1), (subscript.stop, size), (subscript.step, 1)):
        if isinstance(bound, ast.Primary) and bound.value is None:
            bounds.append(default)
        elif _is_literal(bound):
            bounds.append(bound.value)
        else:
            return None
    start, stop, step = bounds
    if not step:
        return None
    return list(range(start, stop + (1 if step > 0 else -1), step))


def _unsupported_subscript(subscript, size: int | None) -> bool:
    """Whether a non-literal subscript of a dimension cannot be expanded element-wise."""
    return size is None or (
        isinstance(subscript, ast.Slice) and _slice_values(subscript, size) is None
    )


class _StreamExpander:
    def __init__(self, symbols: dict, sets: dict[SetKey, ConnectionSet], nested: set[str]):
        self._symbols = symbols
        self._sets = sets
        self._nested = nested
        self.unexpanded: set[str] = set()
        self._flows: dict[str, list[str]] = {}
        for name, sym in symbols.items():
            if "flow" in sym.prefixes:
                self._flows.setdefault(name.rpartition(".")[0], []).append(name)
        for connector in sorted(
            {name.rpartition(".")[0] for name, sym in symbols.items() if "stream" in sym.prefixes}
        ):
            found = len(self._flows.get(connector, []))
            if found != 1:
                raise ModelicaSemanticError(
                    f"Connector {connector} with a stream variable must have exactly one flow"
                    f" variable, found {found}"
                )
        whole = {key: self._whole_array_key(key) for key in sets if key[2]}
        # Names connected as inside connectors under a non-literal connector subscript
        self._array_connected = {key[0] for key, is_whole in whole.items() if is_whole}
        # Of those, names also connected element-wise, so neither set is complete
        self._overlapping = {
            key[0] for key, is_whole in whole.items() if not is_whole
        } & self._array_connected

    def _whole_array_key(self, key: SetKey) -> bool:
        """Whether a set key has a non-literal subscript on an arrayed connector name part."""
        return any(
            sizes and not (subs and all(isinstance(i, int) for i in subs))
            for subs, sizes in zip(key[1], self._part_sizes(key[0]))
        )

    def _part_sizes(self, name: str) -> list[list]:
        """Array sizes of each name part of a flat symbol, None where not a literal."""
        sym = self._symbols.get(name)
        dims = [] if sym is None else list(sym.dimensions)
        return [
            (
                []
                if all(_is_whole(d) for d in group)
                else [d.value if _is_literal(d) else None for d in group]
            )
            for group in align_groups(dims, name.count(".") + 1)
        ]

    def _flow(self, stream_ref: ast.ComponentRef) -> ast.ComponentRef:
        """Ref to the flow variable in the connector of a stream variable (MLS 15.1)."""
        flow = _with_last_index(stream_ref, [None])
        (flow.name,) = self._flows[stream_ref.name.rpartition(".")[0]]
        return flow

    def _incomplete(self, name: str) -> bool:
        """Whether a stream variable's connector is in a connect not expanded yet."""
        parts = name.split(".")
        return any(".".join(parts[:k]) in self._nested for k in range(1, len(parts)))

    def _unknown_set(self, connected: ConnectionSet) -> bool:
        """Whether a connection set may be missing members."""
        return any(self._incomplete(k[0]) or k[0] in self._overlapping for k in connected)

    def _unexpanded(self, op: str, ref: ast.ComponentRef) -> ast.Expression:
        self.unexpanded.add(ref.name)
        return _call(op, copy.deepcopy(ref))

    def in_stream(self, ref: ast.ComponentRef):
        """Expansion of inStream(ref), or the unexpanded call if the set is not known."""
        return self._expand("inStream", ref, self._in_stream_element)

    def actual_stream(self, ref: ast.ComponentRef):
        """Expansion of actualStream(ref) (MLS 15.3)."""
        return self._expand("actualStream", ref, self._actual_stream_element)

    def _expand(self, op: str, ref: ast.ComponentRef, element: Callable):
        """Expansion of a stream operator, element-wise over non-literal connector subscripts."""
        if self._incomplete(ref.name):
            return self._unexpanded(op, ref)
        if set_key(ref, True) in self._sets:
            return element(ref)
        sizes = self._part_sizes(ref.name)
        groups = self._connector_groups(ref, sizes)
        if groups is None or ref.name in self._array_connected:
            return self._unexpanded(op, ref)
        non_literal = [
            (i, sizes[k][j])
            for k, g in enumerate(groups[:-1])
            if sizes[k]
            for j, i in enumerate(g)
            if not _is_literal(i)
        ]
        if not non_literal:
            return element(ref)
        if any(_unsupported_subscript(i, size) for i, size in non_literal):
            return self._unexpanded(op, ref)
        return self._elementwise(ref, groups, sizes, element)

    def _connector_groups(self, ref: ast.ComponentRef, sizes: list) -> list | None:
        """Index groups of a ref with whole connector dimensions spelled out, None if partial."""
        groups = aligned_indices(ref)
        for k, group in enumerate(groups[:-1]):
            if not sizes[k]:
                continue
            if all(i is None for i in group):
                groups[k] = [None] * len(sizes[k])
            elif len(group) != len(sizes[k]):
                return None
        return groups

    def _elementwise(self, ref: ast.ComponentRef, groups: list, sizes: list, element: Callable):
        """Array (whole dimension or range) or if-chain (expression subscript) of expansions."""
        for k, group in enumerate(groups[:-1]):
            for j, subscript in enumerate(group):
                if not sizes[k] or _is_literal(subscript):
                    continue
                is_range = subscript is None or isinstance(subscript, ast.Slice)
                values = (
                    _slice_values(subscript, sizes[k][j])
                    if isinstance(subscript, ast.Slice)
                    else range(1, sizes[k][j] + 1)
                )
                assert values is not None
                elements = []
                for value in values:
                    element_groups = copy.deepcopy(groups)
                    element_groups[k][j] = ast.Primary(value=value)
                    elements.append(self._elementwise(ref, element_groups, sizes, element))
                if is_range or not elements:
                    return ast.Array(values=elements)
                conditions = [
                    ast.Expression(
                        operator="==", operands=[copy.deepcopy(subscript), ast.Primary(value=v)]
                    )
                    for v in range(1, len(elements))
                ]
                if not conditions:
                    return elements[0]
                return ast.IfExpression(conditions=conditions, expressions=elements)
        element_ref = copy.deepcopy(ref)
        element_ref.indices = groups
        return element(element_ref)

    def _in_stream_element(self, ref: ast.ComponentRef):
        """inStream(ref) for a ref with literal connector subscripts."""
        key = set_key(ref, True)
        connected = self._sets.get(key)
        if connected is None:
            return copy.deepcopy(ref)
        if self._unknown_set(connected):
            return self._unexpanded("inStream", ref)
        return self._mix(connected, key, ref.indices[-1] if ref.indices else [None])

    def _actual_stream_element(self, ref: ast.ComponentRef):
        positive = ast.Expression(operator=">", operands=[self._flow(ref), ast.Primary(value=0)])
        return ast.IfExpression(
            conditions=[positive],
            expressions=[self._in_stream_element(ref), copy.deepcopy(ref)],
        )

    def _mix(self, connected: ConnectionSet, exclude: SetKey, subscript: list):
        """Mixing of all members of a connection set except one (MLS 15.2)."""
        inside = [
            _with_last_index(ref, subscript)
            for key, (ref, is_inside) in connected.items()
            if is_inside and key != exclude
        ]
        outside = [
            _with_last_index(ref, subscript)
            for key, (ref, is_inside) in connected.items()
            if not is_inside and key != exclude
        ]
        if not inside and not outside:
            return _with_last_index(connected[exclude][0], subscript)
        if len(inside) == 1 and not outside:
            return inside[0]
        if not inside and len(outside) == 1:
            return self.in_stream(outside[0])
        weights = [
            _call(
                "max",
                ast.Expression(operator="-", operands=[self._flow(h)]),
                ast.Primary(value=EPS),
            )
            for h in inside
        ] + [_call("max", self._flow(h), ast.Primary(value=EPS)) for h in outside]
        streams = inside + [self.in_stream(h) for h in outside]
        numerator = sum_terms(
            [ast.Expression(operator="*", operands=[w, h]) for w, h in zip(weights, streams)]
        )
        return ast.Expression(operator="/", operands=[numerator, sum_terms(copy.deepcopy(weights))])

    def outside_equations(self) -> list[ast.Equation]:
        """Equations for the stream variables of the outside connectors in each set."""
        equations = []
        for connected in unique_sets(self._sets):
            if len(connected) < 2 or self._unknown_set(connected):
                continue
            outside = [ref for ref, is_inside in connected.values() if not is_inside]
            if len(outside) == len(connected) and all(
                set_key(ref, True) not in self._sets for ref in outside
            ):
                # No connection from outside: the mirrored mixing equations are singular
                logger.warning(
                    "Stream variables of connectors "
                    + ", ".join(ref.name for ref in outside)
                    + " are only equated, since nothing is connected to them from outside"
                )
                equations.extend(
                    ast.Equation(left=copy.deepcopy(a), right=copy.deepcopy(b))
                    for a, b in zip(outside, outside[1:])
                )
                continue
            for key, (ref, is_inside) in connected.items():
                if not is_inside:
                    right = self._mix(connected, key, ref.indices[-1])
                    equations.append(ast.Equation(left=copy.deepcopy(ref), right=right))
        return equations

    def _expand_call(self, expr: ast.Expression):
        """Expansion of an inStream() or actualStream() call, or None for another expression."""
        op = expr.operator
        if not isinstance(op, ast.ComponentRef) or op.child or len(expr.operands) != 1:
            return None
        if op.name not in _OPERATORS:
            return None
        (ref,) = expr.operands
        if not isinstance(ref, ast.ComponentRef):
            raise ModelicaSemanticError(f"Argument of {op.name}() must be a component reference")
        sym = self._symbols.get(ref.name)
        if sym is None or "stream" not in sym.prefixes:
            raise ModelicaSemanticError(
                f"Argument {ref.name} of {op.name}() is not a stream variable"
            )
        if op.name == "inStream":
            return self.in_stream(ref)
        return self.actual_stream(ref)

    def rewrite(self, node):
        """Rewrite stream operator calls in place, returning the node or its replacement."""
        if isinstance(node, list):
            for i, item in enumerate(node):
                node[i] = self.rewrite(item)
        elif isinstance(node, ast.Expression):
            expanded = self._expand_call(node)
            if expanded is not None:
                return expanded
            self.rewrite(node.operands)
        elif isinstance(node, ast.IfExpression):
            self.rewrite(node.conditions)
            self.rewrite(node.expressions)
        elif isinstance(node, ast.Array):
            self.rewrite(node.values)
        elif isinstance(node, ast.ForArray):
            node.expression = self.rewrite(node.expression)
        elif isinstance(node, (ast.Equation, ast.AssignmentStatement)):
            node.right = self.rewrite(node.right)
            if isinstance(node, ast.Equation):
                node.left = self.rewrite(node.left)
        elif isinstance(
            node, (ast.IfEquation, ast.WhenEquation, ast.IfStatement, ast.WhenStatement)
        ):
            self.rewrite(node.conditions)
            self.rewrite(node.blocks)
        elif isinstance(node, ast.ForEquation):
            self.rewrite(node.equations)
        elif isinstance(node, (ast.ForStatement, ast.WhileStatement)):
            if isinstance(node, ast.WhileStatement):
                node.condition = self.rewrite(node.condition)
            self.rewrite(node.statements)
        elif isinstance(node, ast.Function):
            self.rewrite(node.arguments)
        return node
