#!/usr/bin/env python
"""
Map the quoted identifiers of flat names to plain names (MLS 2.3.1).
"""

from __future__ import annotations

import copy
import re
from collections import OrderedDict

from . import ModelicaSemanticError
from ._listener import TreeListener, TreeWalker
from .. import ast

# S-ESCAPE of MLS A.1 with the characters of MLS 2.4.4
_ESCAPES = {
    "'": "'",
    '"': '"',
    "?": "?",
    "\\": "\\",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}

_ESCAPE_RE = re.compile(r"\\(.)", re.DOTALL)
_Q_IDENT_RE = re.compile(r"'((?:[^'\\]|\\.)*)'", re.DOTALL)

_NAMED_NODES = (ast.Symbol, ast.ComponentRef, ast.ForIndex, ast.ShortClassDefinition)


def _unescape(match: re.Match) -> str:
    char = match.group(1)
    if char not in _ESCAPES:
        raise ModelicaSemanticError(f"Invalid escape sequence \\{char} in quoted identifier")
    return _ESCAPES[char]


def unquote_name(name: str) -> str:
    """Return *name* with each quoted identifier replaced by its content, escapes resolved

    For example c.'a.b' becomes c.a.b and '\\'x' becomes 'x.
    """
    if "'" not in name:
        return name
    parts = []
    for part in ast.split_name(name):
        match = _Q_IDENT_RE.match(part)
        if match:
            part = _ESCAPE_RE.sub(_unescape, match.group(1)) + part[match.end() :]
        parts.append(part)
    return ".".join(parts)


class _UnquoteWalker(TreeWalker):
    def skip_child(self, tree: ast.Node, child_name: str) -> bool:
        # Back reference to a class outside the flat class
        if isinstance(tree, ast.Symbol) and child_name == "_connector_type":
            return True
        return super().skip_child(tree, child_name)


class _UnquoteListener(TreeListener):
    @staticmethod
    def _rename_keys(scope: ast.Class, items: dict, seen: dict[str, str]) -> OrderedDict:
        renamed = OrderedDict()
        for key, value in items.items():
            new_key = unquote_name(key)
            if new_key in seen:
                raise ModelicaSemanticError(
                    f"Names {seen[new_key]} and {key} in {scope.name or 'the top level'} "
                    f"both map to {new_key} when unquoted"
                )
            seen[new_key] = key
            renamed[new_key] = value
        return renamed

    def _unquote_class(self, tree: ast.Class) -> None:
        if tree.name is not None:
            tree.name = unquote_name(tree.name)
        elements: dict[str, str] = {}
        tree.classes = self._rename_keys(tree, tree.classes, elements)
        tree.symbols = self._rename_keys(tree, tree.symbols, elements)
        tree.functions = self._rename_keys(tree, tree.functions, {})
        tree.imports = self._rename_keys(tree, tree.imports, {})

    def enterEvery(self, tree: ast.Node) -> None:
        # Dispatch on isinstance, the walker calls enter<Name> only for the exact node type
        super().enterEvery(tree)
        if isinstance(tree, ast.Class):
            self._unquote_class(tree)
        elif isinstance(tree, _NAMED_NODES):
            tree.name = unquote_name(tree.name)
        elif isinstance(tree, ast.Function):
            tree.name = unquote_name(tree.name)
        elif isinstance(tree, ast.Expression) and isinstance(tree.operator, str):
            # Flattening resolves a function call to the full name of the function
            tree.operator = unquote_name(tree.operator)


def unquote_flat_names(flat_tree: ast.Tree) -> ast.Tree:
    """Return a copy of a flattened tree with each quoted identifier in its names unquoted

    For tools that quote every name, such as 'tank.h[1]': c.'a.b' becomes c.a.b, with
    escape sequences resolved.

    Raises: ModelicaSemanticError if two distinct flat names map to the same name, such as
    'x' and x.
    """
    flat_tree = copy.deepcopy(flat_tree)
    _UnquoteWalker().walk(_UnquoteListener(), flat_tree)
    return flat_tree
