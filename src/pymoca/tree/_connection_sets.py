"""Connection sets of connector variables (MLS 9.2)."""

from __future__ import annotations

from collections.abc import Iterator

from .. import ast

# Connection set member key: (flat name, subscripts per connector name part, is inside connector)
SetKey = tuple[str, tuple, bool]
ConnectionSet = dict[SetKey, tuple[ast.ComponentRef, bool]]


def align_groups(groups: list, n_parts: int) -> list[list]:
    """Last n_parts index groups, padded at the front with [None] to one per name part."""
    groups = [list(g) for g in groups[-n_parts:]]
    return [[None] for _ in range(n_parts - len(groups))] + groups


def aligned_indices(ref: ast.ComponentRef) -> list[list]:
    """Index groups of a flat ref, one per name part.

    A ref inside an equation of an array component carries index groups for only its
    trailing name parts, so groups are aligned to the end of the name.
    """
    return align_groups(ref.indices, ref.name.count(".") + 1)


def part_subscripts(ref: ast.ComponentRef) -> tuple[tuple, ...]:
    """Subscripts of each name part of a flat ref: a literal value or the source text.

    A name part without subscripts, or with only whole-dimension subscripts, gives ().
    """
    return tuple(
        (
            ()
            if all(i is None for i in group)
            else tuple(
                None if i is None else i.value if isinstance(i, ast.Primary) else str(i)
                for i in group
            )
        )
        for group in aligned_indices(ref)
    )


def set_key(ref: ast.ComponentRef, inside: bool) -> SetKey:
    """Key of a connect member ref in a connection set, ignoring the variable's own subscript."""
    return (ref.name, part_subscripts(ref)[:-1], inside)


def merge_into_sets(
    sets: dict[SetKey, ConnectionSet],
    left: ast.ComponentRef,
    left_inside: bool,
    right: ast.ComponentRef,
    right_inside: bool,
) -> None:
    """Merge the connection sets of two connected member refs, sharing one dict per set."""
    left_key = set_key(left, left_inside)
    right_key = set_key(right, right_inside)
    connected = sets.get(left_key, {})
    connected.update(sets.get(right_key, {}))
    connected[left_key] = (left, left_inside)
    connected[right_key] = (right, right_inside)
    for k in connected:
        sets[k] = connected


def unique_sets(sets: dict[SetKey, ConnectionSet]) -> Iterator[ConnectionSet]:
    """Each distinct connection set once, in first-seen order."""
    seen: set[int] = set()
    for connected in sets.values():
        if id(connected) not in seen:
            seen.add(id(connected))
            yield connected


def sum_terms(terms: list) -> ast.Expression:
    """Right-folded binary sum of one or more terms."""
    expr = terms[-1]
    for term in reversed(terms[:-1]):
        expr = ast.Expression(operator="+", operands=[term, expr])
    return expr
