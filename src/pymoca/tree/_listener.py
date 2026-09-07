#!/usr/bin/env python
"""
TreeListener and TreeWalker for the pymoca.tree package.
"""

from __future__ import annotations

import logging
from typing import Iterable

from .. import ast

logger = logging.getLogger("pymoca")


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

    def enterForArray(self, tree: ast.ForArray) -> None:
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

    def exitForArray(self, tree: ast.ForArray) -> None:
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
            and child_name in ("parent", "parent_instance")
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

    def handle_walk(self, listener: TreeListener, tree: ast.Node | dict | list) -> None:
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
