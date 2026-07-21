#!/usr/bin/env python
"""
Tools for tree walking and visiting etc.

This package re-exports all public symbols for backward compatibility.
The implementation is split across submodules:

  instance.py        — InstantiationState, Instance{Element,Class,Symbol,Tree} (MLS 5.6.1)
  _listener.py       — TreeListener, TreeWalker
  _name_lookup.py    — find_name and helpers (MLS 5.3, other details throughout Chapter 5)
  _instantiation.py  — instantiate (MLS 5.6.1)
  _flattening.py     — flatten_instance and helpers (MLS 5.6.2)
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "FlatteningError",
    "InstanceClass",
    "InstanceElement",
    "InstanceSymbol",
    "InstanceTree",
    "InstantiationError",
    "InstantiationState",
    "ModelicaError",
    "ModelicaSemanticError",
    "NameLookupError",
    "TreeListener",
    "TreeWalker",
    "find_name",
    "flatten",
    "flatten_class",
    "flatten_instance",
    "flatten_to_tree",
    "instantiate",
]


class ModelicaError(Exception):
    """Common base class for all Modelica language errors"""

    def __init__(self, msg):
        self.msg = msg
        super().__init__(msg)

    def __str__(self) -> str:
        return str(self.msg)

    def __repr__(self) -> str:
        return type(self).__name__ + "(" + str(self) + ")"


class ModelicaSemanticError(ModelicaError):
    """Error in meaning of Modelica code"""

    pass


class NameLookupError(ModelicaError):
    """Error looking up a Modelica name"""

    pass


class InstantiationError(ModelicaError):
    """Error instantiating a Modelica element"""

    pass


class FlatteningError(ModelicaError):
    """Error flattening a Modelica model"""

    pass


@dataclass
class RecursionGuard:
    """Cycle detection for instantiation and name lookup.

    Mutable and shared across a single instantiation/lookup operation.
    """

    current_instances: set[InstanceClass] = field(default_factory=set)
    current_extends: set = field(default_factory=set)
    # Memoization cache for _find_inherited results.
    # Key: (name_str, id(scope)) — stable within a single flatten() call because
    # extends lists are populated at PARTIAL instantiation and don't grow after that.
    # Both "found" and "not found" (None) results are cached.
    _find_inherited_cache: dict = field(default_factory=dict)
    # Cache for type-class instantiation in _instantiate_symbol where modification_environment
    # is empty.  Keyed by symbol_type itself (not id()), to avoid false hits from CPython
    # reusing the address of a freed object.  Value: the fully-instantiated type InstanceClass.
    # Cache hits are shallow-cloned so each symbol gets its own parent_instance slot.
    _symbol_type_cache: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LookupOptions:
    """Per-call options for name resolution.

    Frozen so ``replace()`` is the idiom for variants.
    """

    instantiate_in_place: bool = True
    search_imports: bool = True
    search_parent: bool = True
    search_inherited: bool = True
    check_encapsulated: bool = True
    evaluate_parameters: bool = False
    # Names bound as for-loop iteration variables in the equation/statement body
    # currently being resolved (MLS 5.3.1 step 1, highest lookup precedence -- an
    # iteration variable shadows any same-named class member for the loop's extent).
    # Empty outside equation/statement resolution, so lookup is unaffected elsewhere.
    iteration_variables: frozenset[str] = frozenset()
    # Mutable set shared by reference across a single inherited-scope traversal to prevent
    # exponential re-visits of the same scope in diamond inheritance hierarchies.
    # Keyed by (name, id(scope)).  Not included in equality/hash.
    _searched_extends: set | None = field(default=None, compare=False, hash=False, repr=False)


from .instance import (  # noqa: E402,F401,I100
    InstanceClass,
    InstanceElement,
    InstanceSymbol,
    InstanceTree,
    InstantiationState,
)
from ._listener import (  # noqa: E402,F401,I100,I202
    TreeListener,
    TreeWalker,
)
from ._flattening import (  # noqa: E402,F401,I100
    flatten_class,
    flatten_instance,
    flatten_to_tree,
)
from ._instantiation import (  # noqa: E402,F401,I100
    instantiate,
)
from ._name_lookup import (  # noqa: E402,F401,I100
    find_name,
)


def flatten(root, class_name):
    """Flatten *class_name* inside *root*, returning a new ``ast.Tree``."""
    return flatten_to_tree(root, class_name)
