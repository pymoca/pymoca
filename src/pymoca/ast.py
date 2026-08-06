#!/usr/bin/env python
"""
Modelica AST definitions
"""

from __future__ import annotations

import copy
import json
import math
import sys
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "AlgorithmSection",
    "Array",
    "AssignmentStatement",
    "Class",
    "ClassModification",
    "ClassModificationArgument",
    "ComponentClause",
    "ComponentRef",
    "ConnectClause",
    "ElementModification",
    "ElementReplaceable",
    "EnumerationLiteral",
    "Equation",
    "EquationSection",
    "Expression",
    "ExtendsClause",
    "ForEquation",
    "ForIndex",
    "ForStatement",
    "Function",
    "IfEquation",
    "IfExpression",
    "IfStatement",
    "ImportClause",
    "Node",
    "Primary",
    "ShortClassDefinition",
    "Slice",
    "Symbol",
    "Tree",
    "Visibility",
    "WhenEquation",
    "WhenStatement",
    "WhileStatement",
    "element_full_name",
    "element_full_reference",
    "element_name_tuple",
    "is_enumeration",
]


class Visibility(Enum):
    PROTECTED = 1, "protected"
    PUBLIC = 2, "public"

    fullname: str

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value

    def __str__(self):
        return self.fullname

    def __lt__(self, other):
        return self.value < other.value


nan = float("nan")

"""
AST Node Type Hierarchy

Root Class
    Class
        Equation
            ComponentRef
            Expression
            Primary
        IfEquation
            Expression
            Equation
        ForEquation
            Expression
            Equation
        ConnectClause
            ComponentRef
        Symbol
"""


class Node:
    def __init__(self, **kwargs):
        self.set_args(**kwargs)

    def set_args(self, **kwargs):
        for key in kwargs.keys():
            if key not in self.__dict__.keys():
                raise KeyError("{:s} not valid arg".format(key))
            self.__dict__[key] = kwargs[key]

    def __repr__(self):
        return "{!r}".format(self.__dict__)

    def __str__(self):
        d = self.to_json(self)
        d["_type"] = self.__class__.__name__
        return json.dumps(d, indent=2, sort_keys=True)

    @classmethod
    def to_json(cls, var):
        def guard(var):
            if isinstance(var, Path):
                return var.resolve().as_uri()
            return var

        if isinstance(var, list):
            res = [cls.to_json(guard(item)) for item in var]
        elif isinstance(var, dict):
            res = {key: cls.to_json(guard(var[key])) for key in var.keys()}
        elif isinstance(var, Node):
            # Avoid infinite recursion by not handling attributes that may go
            # back up in the tree again.
            res = {
                key: cls.to_json(guard(var.__dict__[key]))
                for key in var.__dict__.keys()
                if key not in ("parent", "parent_instance", "scope", "__deepcopy__")
            }
        elif isinstance(var, Visibility):
            res = str(var)
        else:
            res = var
        return res


class Primary(Node):
    def __init__(self, **kwargs):
        self.value: bool | float | int | str | None = None
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(value={!r})".format(type(self).__name__, self.value)

    def __str__(self):
        return "{} value {}".format(type(self).__name__, self.value)


class Array(Node):
    def __init__(self, **kwargs):
        self.values: list[Expression | Primary | ComponentRef | Array] = []
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(values={!r})".format(type(self).__name__, self.values)

    def __str__(self):
        return "{} {}".format(type(self).__name__, self.values)


class Slice(Node):
    def __init__(self, **kwargs):
        self.start: Expression | Primary | ComponentRef = Primary(value=None)
        self.stop: Expression | Primary | ComponentRef = Primary(value=None)
        self.step: Expression | Primary | ComponentRef = Primary(value=1)
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(start={!r}, stop={!r}, step={!r})".format(
            type(self).__name__, self.start, self.stop, self.step
        )

    def __str__(self):
        return "{} start: {}, stop: {}, step: {}".format(
            type(self).__name__, self.start, self.stop, self.step
        )


class ComponentRef(Node):
    def __init__(self, **kwargs):
        self.name: str = ""
        self.indices: list[list[Expression | Slice | Primary | ComponentRef | None]] = [[None]]
        self.child: list[ComponentRef] = []
        # True once name holds a final flattened reference; later passes must leave it as-is
        self.resolved: bool = False
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        # TODO: indices
        if self.child:
            return "{!r}{!r}".format(self.name, self.child)
        else:
            return "{!r}".format(self.name)

    def __str__(self) -> str:
        return ".".join(self.to_tuple())

    def to_tuple(self) -> tuple[str, ...]:
        """
        Convert the nested component reference to flat tuple of names, which is
        hashable and can therefore be used as dictionary key. Note that this
        function ignores any array indices in the component reference.
        :return: flattened tuple of c's names
        """

        if self.child:
            return (self.name,) + self.child[0].to_tuple()
        else:
            return (self.name,)

    @classmethod
    def from_tuple(cls, components: tuple | list) -> ComponentRef:
        """
        Convert the tuple pointing to a component to
        a component reference.
        :param components: tuple of components name
        :return: ComponentRef
        """

        component_ref = ComponentRef(name=components[0], child=[])
        c = component_ref
        for component in components[1:]:
            c.child.append(ComponentRef(name=component, child=[]))
            c = c.child[0]
        return component_ref

    @classmethod
    def from_string(cls, s: str) -> ComponentRef:
        """
        Convert the string pointing to a component using dot notation to
        a component reference.
        :param s: string pointing to component using dot notation
        :return: ComponentRef
        """

        components = s.split(".")
        return cls.from_tuple(components)

    def concatenate(self, arg: ComponentRef) -> ComponentRef:
        """
        Helper function to append two component references to eachother, e.g.
        a "within" component ref and an "object type" component ref.
        :return: New component reference, with other appended to self.
        """

        a = copy.deepcopy(self)
        n = a
        b = arg
        while n.child:
            n = n.child[0]
        b = copy.deepcopy(b)  # Not strictly necessary
        n.child = [b]
        return a


class Expression(Node):
    def __init__(self, **kwargs):
        self.operator: str | ComponentRef | None = None
        self.operands: list[Expression | Primary | ComponentRef | Array | IfExpression] = []
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(operator={!r}, operands={!r})".format(
            type(self).__name__, self.operator, self.operands
        )


class IfExpression(Node):
    def __init__(self, **kwargs):
        self.conditions: list[Expression | Primary | ComponentRef | Array | IfExpression] = []
        self.expressions: list[Expression | Primary | ComponentRef | Array | IfExpression] = []
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(conditions={!r}, expressions={!r})".format(
            type(self).__name__, self.conditions, self.expressions
        )


class Equation(Node):
    def __init__(self, **kwargs):
        self.left: (
            Expression | Primary | ComponentRef | list[Expression | Primary | ComponentRef] | None
        ) = None
        self.right: (
            Expression | Primary | ComponentRef | list[Expression | Primary | ComponentRef] | None
        ) = None
        self.comment: str = ""
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(left={!r}, right={!r})".format(type(self).__name__, self.left, self.right)


class IfEquation(Node):
    def __init__(self, **kwargs):
        self.conditions: list[Expression | Primary | ComponentRef] = []
        self.blocks: list[list[Expression | ForEquation | ConnectClause | IfEquation]] = []
        self.comment: str = ""
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(conditions={!r}, blocks={!r})".format(
            type(self).__name__, self.conditions, self.blocks
        )


class WhenEquation(Node):
    def __init__(self, **kwargs):
        self.conditions: list[Expression | Primary | ComponentRef] = []
        self.blocks: list[list[Expression | ForEquation | ConnectClause | IfEquation]] = []
        self.comment: str = ""
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(conditions={!r}, blocks={!r})".format(
            type(self).__name__, self.conditions, self.blocks
        )


class ForIndex(Node):
    def __init__(self, **kwargs):
        self.name: str = ""
        self.expression: Expression | Primary | Slice | None = None
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(name={!r}, expression={!r})".format(
            type(self).__name__, self.name, self.expression
        )


class ForEquation(Node):
    def __init__(self, **kwargs):
        self.indices: list[ForIndex] = []
        self.equations: list[Equation | ForEquation | ConnectClause] = []
        self.comment: str | None = None
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(indices={!r}, equations={!r})".format(
            type(self).__name__, self.indices, self.equations
        )


class ConnectClause(Node):
    def __init__(self, **kwargs):
        self.left: ComponentRef = ComponentRef()
        self.right: ComponentRef = ComponentRef()
        self.comment: str = ""
        # Set by flattening (MLS 9.2 inner/outer connector annotation), not a parse-time arg.
        self._left_inner: bool = False
        self._right_inner: bool = False
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(left={!r}, right={!r})".format(type(self).__name__, self.left, self.right)


class AssignmentStatement(Node):
    def __init__(self, **kwargs):
        self.left: list[ComponentRef] = []
        self.right: Expression | IfExpression | Primary | ComponentRef | None = None
        self.comment: str = ""
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(left={!r}, right={!r})".format(type(self).__name__, self.left, self.right)


class IfStatement(Node):
    def __init__(self, **kwargs):
        self.conditions: list[Expression | Primary | ComponentRef] = []
        self.blocks: list[list[AssignmentStatement | IfStatement | ForStatement]] = []
        self.comment: str = ""
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(conditions={!r}, blocks={!r})".format(
            type(self).__name__, self.conditions, self.blocks
        )


class WhenStatement(Node):
    def __init__(self, **kwargs):
        self.conditions: list[Expression | Primary | ComponentRef] = []
        self.blocks: list[list[AssignmentStatement | IfStatement | ForStatement]] = []
        self.comment: str = ""
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(conditions={!r}, blocks={!r})".format(
            type(self).__name__, self.conditions, self.blocks
        )


class ForStatement(Node):
    def __init__(self, **kwargs):
        self.indices: list[ForIndex] = []
        self.statements: list[AssignmentStatement | IfStatement | ForStatement] = []
        self.comment: str = ""
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(indices={!r}, statements={!r})".format(
            type(self).__name__, self.indices, self.statements
        )


class WhileStatement(Node):
    def __init__(self, **kwargs):
        self.condition: Expression | Primary | ComponentRef | None = None
        self.statements: list[AssignmentStatement | IfStatement | ForStatement] = []
        self.comment: str = ""
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(condition={!r}, statements={!r})".format(
            type(self).__name__, self.condition, self.statements
        )


class Function(Node):
    def __init__(self, **kwargs):
        self.name: str = ""
        self.arguments: list[Expression | Primary | ComponentRef | Array] = []
        self.comment: str = ""
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(name={!r}, arguments={!r})".format(
            type(self).__name__, self.name, self.arguments
        )


class Symbol(Node):
    """
    A mathematical variable or state of the model
    """

    ATTRIBUTES = [
        "value",
        "min",
        "max",
        "start",
        "fixed",
        "nominal",
        "unit",
        "quantity",
        "displayUnit",
    ]

    def __init__(self, **kwargs):
        # pylint: disable=invalid-name
        self.name: str = ""
        self.type: ComponentRef = ComponentRef()
        self.prefixes: list[str] = []
        self.replaceable: bool = False
        self.final: bool = False
        self.inner: bool = False
        self.outer: bool = False
        self.dimensions: list[list[Expression | Primary | ComponentRef]] = [[Primary(value=None)]]
        self.comment: str = ""
        self.start: Expression | Primary | ComponentRef | Array = Primary(value=None)
        self.min: Expression | Primary | ComponentRef | Array = Primary(value=None)
        self.max: Expression | Primary | ComponentRef | Array = Primary(value=None)
        self.nominal: Expression | Primary | ComponentRef | Array = Primary(value=None)
        self.value: Expression | Primary | ComponentRef | Array = Primary(value=None)
        self.fixed: Primary = Primary(value=False)
        self.unit: Primary = Primary(value=None)
        self.quantity: Primary = Primary(value=None)
        self.displayUnit: Primary = Primary(value=None)
        self.id: int = 0
        self.order: int = 0
        self.visibility: Visibility = Visibility.PUBLIC
        self.class_modification: ClassModification | None = None
        self.parent: Class | None = None
        # Set by flattening on connector stub symbols, not a parse-time arg.
        self._connector_type: Class | None = None
        super().__init__(**kwargs)

    def full_reference(self) -> ComponentRef:
        return element_full_reference(self)

    @property
    def full_name(self) -> str:
        """Return fully-qualified name of this symbol"""
        return element_full_name(self)

    def __str__(self):
        return '{} {}, Type "{}"'.format(type(self).__name__, self.name, self.type)

    def __repr__(self):
        return "{}(name={!r}, type={!r})".format(type(self).__name__, self.name, self.type)


class EnumerationLiteral(Symbol):
    """An enumeration literal declared inside ``type E = enumeration(...)``.

    Subclasses :class:`Symbol` so that ``E.two`` resolves through the ordinary
    name-lookup path (which requires a ``constant`` Symbol).  Carries a 1-based
    *ordinal* value matching the declaration order (MLS 4.8.5).
    """

    def __init__(self, **kwargs):
        self.ordinal: int | None = None  # 1-based position in the enum
        super().__init__(**kwargs)
        self.prefixes = ["constant"]

    def __repr__(self):
        return "{}(name={!r}, ordinal={!r})".format(type(self).__name__, self.name, self.ordinal)


def is_enumeration(obj: Any) -> bool:
    """Return True if *obj* is a Modelica enumeration type class (or instance)."""
    return getattr(obj, "enumeration", False)


class ComponentClause(Node):
    def __init__(self, **kwargs):
        self.prefixes: list[str] = []
        self.type: ComponentRef = ComponentRef()
        self.replaceable: bool = False
        self.dimensions: list[list[Expression | Primary | ComponentRef]] = [[Primary(value=None)]]
        self.comment: list[str] = []
        self.symbol_list: list[Symbol] = []
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(prefixes={!r}, type={}, dimensions={!r}, symbol_list={!r})".format(
            type(self).__name__, self.prefixes, self.type, self.dimensions, self.symbol_list
        )

    def __str__(self):
        pre = " ".join([str(s) for s in self.prefixes])
        return "{} {!r} {!r} {!r}".format(pre, self.type, self.dimensions, self.symbol_list)


class EquationSection(Node):
    def __init__(self, **kwargs):
        self.initial: bool = False
        self.equations: list[Equation | IfEquation | ForEquation | ConnectClause] = []
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(initial={!r}, equations={!r})".format(
            type(self).__name__, self.initial, self.equations
        )


class AlgorithmSection(Node):
    def __init__(self, **kwargs):
        self.initial: bool = False
        self.statements: list[AssignmentStatement | IfStatement | ForStatement] = []
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(initial={!r}, statements={!r})".format(
            type(self).__name__, self.initial, self.statements
        )


class ImportClause(Node):
    def __init__(self, **kwargs):
        self.components: list[ComponentRef] = []
        self.short_name: str = ""
        self.unqualified: bool = False
        # Comments are rare, so ignore
        super().__init__(**kwargs)

    def __str__(self):
        star = ""
        if self.unqualified:
            star = ".*"
        return "import {!r}{!r}{!r}".format(self.short_name, self.components, star)

    def __repr__(self):
        return "{}(components={}, short_name={!r}), unqualfied={!r}".format(
            type(self).__name__, self.components, self.short_name, self.unqualified
        )


class ElementModification(Node):
    def __init__(self, **kwargs):
        self.component: ComponentRef = ComponentRef()
        self.modifications: list[
            Primary | Expression | ClassModification | Array | ComponentRef
        ] = []
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(component={}, modifications={!r})".format(
            type(self).__name__, self.component, self.modifications
        )

    def __str__(self):
        return "{!r} = {!r}".format(self.component, self.modifications)


class ShortClassDefinition(Node):
    def __init__(self, **kwargs):
        self.name: str = ""
        self.type: str = ""
        self.component: ComponentRef = ComponentRef()
        self.class_modification: ClassModification = ClassModification()
        self.replaceable: bool = False
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(name={!r}, type={!r}, component={}, class_modification={!r})".format(
            type(self).__name__, self.name, self.type, self.component, self.class_modification
        )

    def __str__(self):
        return "{!r} = {!r}".format(self.name, self.component)


class ElementReplaceable(Node):
    def __init__(self, **kwargs):
        # TODO, add fields ?
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}()".format(type(self).__name__)


class ClassModification(Node):
    def __init__(self, **kwargs):
        self.arguments: list[ClassModificationArgument] = []
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(arguments={!r})".format(type(self).__name__, self.arguments)


class ClassModificationArgument(Node):
    def __init__(self, **kwargs):
        # value holds a single modification item; the list default is a never-read sentinel
        # that lets Node.set_args validate kwarg names before value is assigned by the parser.
        self.value: (  # type: ignore[assignment]
            ElementModification | ComponentClause | ShortClassDefinition
        ) = []  # type: ignore[assignment]
        self.scope: Class | None = None
        self.redeclare = False
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(value={!r}, scope={!r}, redeclare={!r})".format(
            type(self), self.value, self.scope, self.redeclare
        )

    def __deepcopy__(self, memo):
        _scope, _deepcp = self.scope, self.__deepcopy__
        self.scope, self.__deepcopy__ = None, None  # type: ignore[method-assign]
        new = copy.deepcopy(self, memo)
        self.scope, self.__deepcopy__ = _scope, _deepcp  # type: ignore[method-assign]
        new.scope, new.__deepcopy__ = _scope, _deepcp  # type: ignore[method-assign]
        return new


class ExtendsClause(Node):
    def __init__(self, **kwargs):
        self.component: ComponentRef | None = None
        self.class_modification: ClassModification | None = None
        self.visibility: Visibility = Visibility.PUBLIC
        self.scope: Class | None = None
        # True when created from `class extends X` syntax (MLS §7.3.1).
        # The base class X must be resolved in the inherited scope of the enclosing
        # class, not the local scope (which already contains the redeclaration).
        self.is_class_extends: bool = False
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(component={}, class_modification={!r}, visibility={!r})".format(
            type(self).__name__, self.component, self.class_modification, self.visibility
        )


class Class(Node):
    BUILTIN = ("Real", "Integer", "String", "Boolean")

    def __init__(self, **kwargs):
        self.name: str | None = None
        self.imports: OrderedDict[str, ImportClause | ComponentRef] = OrderedDict()
        # Name-only here; instantiation resolves each to a class later.
        self.extends: list[ExtendsClause] = []
        self.encapsulated: bool = False
        self.partial: bool = False
        self.final: bool = False
        self.replaceable: bool = False
        self.type: str = ""
        self.comment: str = ""
        self.classes: dict[str, Class] = OrderedDict()
        self.symbols: OrderedDict[str, Symbol] = OrderedDict()
        self.functions: OrderedDict[str, Class] = OrderedDict()
        self.initial_equations: list[Equation | ForEquation] = []
        self.equations: list[Equation | ForEquation | ConnectClause] = []
        self.initial_statements: list[AssignmentStatement | IfStatement | ForStatement] = []
        self.statements: list[AssignmentStatement | IfStatement | ForStatement] = []
        self.annotation: ClassModification | None = None
        self.parent: Class | None = None
        self.visibility: Visibility = Visibility.PUBLIC
        self.is_short_class_definition: bool = False
        self.enumeration: bool = False  # True iff this is an enumeration type
        super().__init__(**kwargs)

    def full_reference(self) -> ComponentRef:
        return element_full_reference(self)

    @property
    def full_name(self) -> str:
        """Return fully-qualified lexical name of this class"""
        return element_full_name(self)

    def _extend(self, other: Class) -> None:
        for class_name in other.classes.keys():
            if class_name in self.classes.keys():
                self.classes[class_name]._extend(other.classes[class_name])
            else:
                self.classes[class_name] = other.classes[class_name]

    def _update_parent_refs(self) -> None:
        for c in self.classes.values():
            c.parent = self
            c._update_parent_refs()

    @property
    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root

    def ancestors(self) -> list[Class]:
        """Return a list of all ancestor classes, farthest first."""
        ancestors = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        ancestors.reverse()
        return ancestors

    def copy_including_children(self):
        return copy.deepcopy(self)

    def add_class(self, c: Class) -> None:
        """
        Add a (sub)class to this class.

        :param c: (Sub)class to add.
        """
        assert c.name is not None
        self.classes[c.name] = c
        c.parent = self

    def remove_class(self, c: Class) -> None:
        """
        Removes a (sub)class from this class.

        :param c: (Sub)class to remove.
        """
        assert c.name is not None
        del self.classes[c.name]
        c.parent = None

    def add_symbol(self, s: Symbol) -> None:
        """
        Add a symbol to this class.

        :param s: Symbol to add.
        """
        self.symbols[s.name] = s

    def remove_symbol(self, s: Symbol) -> None:
        """
        Removes a symbol from this class.

        :param s: Symbol to remove.
        """
        del self.symbols[s.name]

    def add_equation(self, e: Equation) -> None:
        """
        Add an equation to this class.

        :param e: Equation to add.
        """
        self.equations.append(e)

    def remove_equation(self, e: Equation) -> None:
        """
        Removes an equation from this class.

        :param e: Equation to remove.
        """
        self.equations.remove(e)

    def add_initial_equation(self, e: Equation) -> None:
        """
        Add an initial equation to this class.

        :param e: Equation to add.
        """
        self.initial_equations.append(e)

    def remove_initial_equation(self, e: Equation) -> None:
        """
        Removes an initial equation from this class.

        :param e: Equation to remove.
        """
        self.initial_equations.remove(e)

    def __deepcopy__(self, memo):
        # Avoid copying the entire tree
        if self.parent is not None and self.parent not in memo:
            memo[id(self.parent)] = self.parent

        _deepcp = self.__deepcopy__
        self.__deepcopy__ = None  # type: ignore[method-assign]
        new = copy.deepcopy(self, memo)
        self.__deepcopy__ = _deepcp  # type: ignore[method-assign]
        new.__deepcopy__ = _deepcp  # type: ignore[method-assign]
        return new

    def __repr__(self):
        return "{}(name={!r}, type={!r})".format(type(self).__name__, self.name, self.type)

    def __str__(self):
        return '{} {}, Type "{}"'.format(type(self).__name__, self.name, self.type)


def element_name_tuple(element: Class | Symbol) -> tuple[str, ...]:
    """Return fully-qualified lexical name of an element as a tuple of names"""
    names = []
    current = element
    while current.parent is not None:
        names.append(current.name)
        current = current.parent
    return tuple(reversed(names))  # type: ignore[arg-type]


def element_full_name(element: Class | Symbol) -> str:
    """Return fully-qualified lexical name of an element"""
    return ".".join(element_name_tuple(element))


def element_full_reference(element: Class | Symbol) -> ComponentRef:
    """Return fully-qualified component reference to element"""
    name_tuple = element_name_tuple(element)
    return ComponentRef.from_tuple(name_tuple) if name_tuple else ComponentRef()


class Tree(Class):
    """
    The root class of the class tree
    """

    BUILTIN_TYPES = {
        "Real": Symbol(
            name="Real",
            type=ComponentRef(name="Real"),
            start=Primary(value=0.0),
            min=Primary(value=-math.inf),
            max=Primary(value=math.inf),
            nominal=Primary(value=None),
            fixed=Primary(value=False),  # True for parameters and constants
            unit=Primary(value=""),
            quantity=Primary(value=""),
            displayUnit=Primary(value=""),
            # TODO: unbounded from spec is missing in Symbol
            # TODO: stateSelect from spec is missing in Symbol
            class_modification=ClassModification(),
        ),
        "Integer": Symbol(
            name="Integer",
            type=ComponentRef(name="Integer"),
            start=Primary(value=0.0),
            min=Primary(value=-sys.maxsize),
            max=Primary(value=sys.maxsize),
            fixed=Primary(value=False),  # True for parameters and constants
            quantity=Primary(value=""),
            class_modification=ClassModification(),
        ),
        "Boolean": Symbol(
            name="Boolean",
            type=ComponentRef(name="Boolean"),
            start=Primary(value=0.0),
            fixed=Primary(value=False),  # True for parameters and constants
            quantity=Primary(value=""),
            class_modification=ClassModification(),
        ),
        "String": Symbol(
            name="String",
            type=ComponentRef(name="String"),
            start=Primary(value=0.0),
            fixed=Primary(value=False),  # True for parameters and constants
            quantity=Primary(value=""),
            class_modification=ClassModification(),
        ),
    }

    # Predefined enumeration types per MLS 4.8.4; maps name → tuple of literal names
    BUILTIN_ENUM_TYPES = {
        "StateSelect": ("never", "avoid", "default", "prefer", "always"),
    }

    # Predefined opaque types (no attributes/literals); MLS 16.2 Clock
    BUILTIN_OPAQUE_TYPES = frozenset({"Clock"})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "package"
        self._create_builtins()

    def _create_builtins(self):
        """Add builtins to root of tree"""
        for name, symbol in self.BUILTIN_TYPES.items():
            new_symbol = copy.deepcopy(symbol)
            type_class = Class(name=name, type="type", parent=self)
            new_symbol.parent = type_class
            type_class.symbols[name] = new_symbol
            self.classes[name] = type_class
        for name, literals in self.BUILTIN_ENUM_TYPES.items():
            type_class = Class(name=name, type="type", parent=self)
            type_class.enumeration = True
            for ordinal, literal in enumerate(literals, start=1):
                enum_sym = EnumerationLiteral(
                    name=literal,
                    type=ComponentRef(name=name),
                    ordinal=ordinal,
                    class_modification=ClassModification(),
                )
                enum_sym.parent = type_class
                type_class.symbols[literal] = enum_sym
            self.classes[name] = type_class
        for name in self.BUILTIN_OPAQUE_TYPES:
            type_class = Class(name=name, type="type", parent=self)
            self.classes[name] = type_class

    def extend(self, other: Tree) -> None:
        self._extend(other)
        self.update_parent_refs()

    def update_parent_refs(self) -> None:
        self._update_parent_refs()

    def __repr__(self):
        return "{}(classes={!r})".format(type(self).__name__, self.classes)
