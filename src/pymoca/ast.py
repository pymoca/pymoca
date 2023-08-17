#!/usr/bin/env python
"""
Modelica AST definitions
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Type, Union  # noqa: F401


class ClassNotFoundError(Exception):
    pass


class ConstantSymbolNotFoundError(Exception):
    pass


class FoundElementaryClassError(Exception):
    pass


class Visibility(Enum):
    PRIVATE = 0, "private"
    PROTECTED = 1, "protected"
    PUBLIC = 2, "public"

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
        if isinstance(var, list):
            res = [cls.to_json(item) for item in var]
        elif isinstance(var, dict):
            res = {key: cls.to_json(var[key]) for key in var.keys()}
        elif isinstance(var, Node):
            # Avoid infinite recursion by not handling attributes that may go
            # back up in the tree again.
            res = {
                key: cls.to_json(var.__dict__[key])
                for key in var.__dict__.keys()
                if key not in ("parent", "scope", "__deepcopy__")
            }
        elif isinstance(var, Visibility):
            res = str(var)
        else:
            res = var
        return res


class Primary(Node):
    def __init__(self, **kwargs):
        self.value = None  # type: Union[bool, float, int, str, Type[None]]
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(value={!r})".format(type(self).__name__, self.value)

    def __str__(self):
        return "{} value {}".format(type(self).__name__, self.value)


class Array(Node):
    def __init__(self, **kwargs):
        self.values = []  # type: List[Union[Expression, Primary, ComponentRef, Array]]
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(values={!r})".format(type(self).__name__, self.values)

    def __str__(self):
        return "{} {}".format(type(self).__name__, self.values)


class Slice(Node):
    def __init__(self, **kwargs):
        self.start = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef]
        self.stop = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef]
        self.step = Primary(value=1)  # type: Union[Expression, Primary, ComponentRef]
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
        self.name = ""  # type: str
        self.indices = [[None]]  # type: List[List[Union[Expression, Slice, Primary, ComponentRef]]]
        self.child = []  # type: List[ComponentRef]
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        # TODO: indices
        if self.child:
            return "{!r}{!r}".format(self.name, self.child)
        else:
            return "{!r}".format(self.name)

    def __str__(self) -> str:
        return ".".join(self.to_tuple())

    def to_tuple(self) -> tuple:
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
    def from_tuple(cls, components: Union[tuple, list]) -> "ComponentRef":
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
    def from_string(cls, s: str) -> "ComponentRef":
        """
        Convert the string pointing to a component using dot notation to
        a component reference.
        :param s: string pointing to component using dot notation
        :return: ComponentRef
        """

        components = s.split(".")
        return cls.from_tuple(components)

    def concatenate(self, arg: "ComponentRef") -> "ComponentRef":
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
        self.operator = None  # type: Union[str, ComponentRef]
        self.operands = (
            []
        )  # type: List[Union[Expression, Primary, ComponentRef, Array, IfExpression]]
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(operator={!r}, operands={!r})".format(
            type(self).__name__, self.operator, self.operands
        )


class IfExpression(Node):
    def __init__(self, **kwargs):
        self.conditions = (
            []
        )  # type: List[Union[Expression, Primary, ComponentRef, Array, IfExpression]]
        self.expressions = (
            []
        )  # type: List[Union[Expression, Primary, ComponentRef, Array, IfExpression]]
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(conditions={!r}, expressions={!r})".format(
            type(self).__name__, self.conditions, self.expressions
        )


class Equation(Node):
    def __init__(self, **kwargs):
        self.left = (
            None
        )  # type: Union[Expression, Primary, ComponentRef, List[Union[Expression, Primary, ComponentRef]]]
        self.right = (
            None
        )  # type: Union[Expression, Primary, ComponentRef, List[Union[Expression, Primary, ComponentRef]]]
        self.comment = ""  # type: str
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(left={!r}, right={!r})".format(type(self).__name__, self.left, self.right)


class IfEquation(Node):
    def __init__(self, **kwargs):
        self.conditions = []  # type: List[Union[Expression, Primary, ComponentRef]]
        self.blocks = (
            []
        )  # type: List[List[Union[Expression, ForEquation, ConnectClause, IfEquation]]]
        self.comment = ""  # type: str
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(conditions={!r}, blocks={!r})".format(
            type(self).__name__, self.conditions, self.blocks
        )


class WhenEquation(Node):
    def __init__(self, **kwargs):
        self.conditions = []  # type: List[Union[Expression, Primary, ComponentRef]]
        self.blocks = (
            []
        )  # type: List[List[Union[Expression, ForEquation, ConnectClause, IfEquation]]]
        self.comment = ""  # type: str
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(conditions={!r}, blocks={!r})".format(
            type(self).__name__, self.conditions, self.blocks
        )


class ForIndex(Node):
    def __init__(self, **kwargs):
        self.name = ""  # type: str
        self.expression = None  # type: Union[Expression, Primary, Slice]
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(name={!r}, expression={!r})".format(
            type(self).__name__, self.name, self.expression
        )


class ForEquation(Node):
    def __init__(self, **kwargs):
        self.indices = []  # type: List[ForIndex]
        self.equations = []  # type: List[Union[Equation, ForEquation, ConnectClause]]
        self.comment = None  # type: str
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(indices={!r}, equations={!r})".format(
            type(self).__name__, self.indices, self.equations
        )


class ConnectClause(Node):
    def __init__(self, **kwargs):
        self.left = ComponentRef()  # type: ComponentRef
        self.right = ComponentRef()  # type: ComponentRef
        self.comment = ""  # type: str
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(left={!r}, right={!r})".format(type(self).__name__, self.left, self.right)


class AssignmentStatement(Node):
    def __init__(self, **kwargs):
        self.left = []  # type: List[ComponentRef]
        self.right = None  # type: Union[Expression, IfExpression, Primary, ComponentRef]
        self.comment = ""  # type: str
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(left={!r}, right={!r})".format(type(self).__name__, self.left, self.right)


class IfStatement(Node):
    def __init__(self, **kwargs):
        self.conditions = []  # type: List[Union[Expression, Primary, ComponentRef]]
        self.blocks = []  # type: List[List[Union[AssignmentStatement, IfStatement, ForStatement]]]
        self.comment = ""  # type: str
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(conditions={!r}, blocks={!r})".format(
            type(self).__name__, self.conditions, self.blocks
        )


class WhenStatement(Node):
    def __init__(self, **kwargs):
        self.conditions = []  # type: List[Union[Expression, Primary, ComponentRef]]
        self.blocks = []  # type: List[List[Union[AssignmentStatement, IfStatement, ForStatement]]]
        self.comment = ""  # type: str
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(conditions={!r}, blocks={!r})".format(
            type(self).__name__, self.conditions, self.blocks
        )


class ForStatement(Node):
    def __init__(self, **kwargs):
        self.indices = []  # type: List[ForIndex]
        self.statements = []  # type: List[Union[AssignmentStatement, IfStatement, ForStatement]]
        self.comment = ""  # type: str
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(indices={!r}, statements={!r})".format(
            type(self).__name__, self.indices, self.statements
        )


class Function(Node):
    def __init__(self, **kwargs):
        self.name = ""  # type: str
        self.arguments = []  # type: List[Union[Expression, Primary, ComponentRef, Array]]
        self.comment = ""  # type: str
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
        self.name = ""  # type: str
        self.type = ComponentRef()  # type: Union[ComponentRef, InstanceClass]
        self.prefixes = []  # type: List[str]
        self.redeclare = False  # type: bool
        self.final = False  # type: bool
        self.inner = False  # type: bool
        self.outer = False  # type: bool
        self.dimensions = [
            [Primary(value=None)]
        ]  # type: List[List[Union[Expression, Primary, ComponentRef]]]
        self.comment = ""  # type: str
        self.start = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.min = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.max = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.nominal = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.value = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.fixed = Primary(value=False)  # type: Primary
        self.unit = Primary(value=None)  # type: Primary
        self.quantity = Primary(value=None)  # type: Primary
        self.displayUnit = Primary(value=None)  # type: Primary
        self.id = 0  # type: int
        self.order = 0  # type: int
        self.visibility = Visibility.PRIVATE  # type: Visibility
        self.class_modification = None  # type: ClassModification
        super().__init__(**kwargs)

    def __str__(self):
        return '{} {}, Type "{}"'.format(type(self).__name__, self.name, self.type)

    def __repr__(self):
        return "{}(name={!r}, type={!r})".format(type(self).__name__, self.name, self.type)


class ComponentClause(Node):
    def __init__(self, **kwargs):
        self.prefixes = []  # type: List[str]
        self.type = ComponentRef()  # type: ComponentRef
        self.dimensions = [
            [Primary(value=None)]
        ]  # type: List[List[Union[Expression, Primary, ComponentRef]]]
        self.comment = []  # type: List[str]
        self.symbol_list = []  # type: List[Symbol]
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
        self.initial = False  # type: bool
        self.equations = []  # type: List[Union[Equation, IfEquation, ForEquation, ConnectClause]]
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(initial={!r}, equations={!r})".format(
            type(self).__name__, self.initial, self.equations
        )


class AlgorithmSection(Node):
    def __init__(self, **kwargs):
        self.initial = False  # type: bool
        self.statements = []  # type: List[Union[AssignmentStatement, IfStatement, ForStatement]]
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(initial={!r}, statements={!r})".format(
            type(self).__name__, self.initial, self.statements
        )


class ImportClause(Node):
    def __init__(self, **kwargs):
        self.components = []  # type: List[ComponentRef]
        self.short_name = ""  # type: str
        self.unqualified = False  # type: bool
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
        self.component = ComponentRef()  # type: ComponentRef
        self.modifications = (
            []
        )  # type: List[Union[Primary, Expression, ClassModification, Array, ComponentRef]]
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(component={}, modifications={!r})".format(
            type(self).__name__, self.component, self.modifications
        )

    def __str__(self):
        return "{!r} = {!r}".format(self.component, self.modifications)


class ShortClassDefinition(Node):
    def __init__(self, **kwargs):
        self.name = ""  # type: str
        self.type = ""  # type: str
        self.component = ComponentRef()  # type: ComponentRef
        self.class_modification = ClassModification()  # type: ClassModification
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
        self.arguments = []  # type: List[ClassModificationArgument]
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(arguments={!r})".format(type(self).__name__, self.arguments)


class ClassModificationArgument(Node):
    def __init__(self, **kwargs):
        self.value = []  # type: Union[ElementModification, ComponentClause, ShortClassDefinition]
        self.scope = None  # type: InstanceClass
        self.redeclare = False
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(value={!r}, scope={!r}, redeclare={!r})".format(
            type(self).__name__, self.value, self.scope, self.redeclare
        )

    def __deepcopy__(self, memo):
        _scope, _deepcp = self.scope, self.__deepcopy__
        self.scope, self.__deepcopy__ = None, None
        new = copy.deepcopy(self, memo)
        self.scope, self.__deepcopy__ = _scope, _deepcp
        new.scope, new.__deepcopy__ = _scope, _deepcp
        return new


class ExtendsClause(Node):
    def __init__(self, **kwargs):
        self.component = None  # type: ComponentRef
        self.class_modification = None  # type: ClassModification
        self.visibility = Visibility.PRIVATE  # type: Visibility
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}(component={}, class_modification={!r}, visibility={!r})".format(
            type(self).__name__, self.component, self.class_modification, self.visibility
        )


class Class(Node):
    BUILTIN = ("Real", "Integer", "String", "Boolean")

    def __init__(self, **kwargs):
        self.name = None  # type: str
        self.imports = OrderedDict()  # type: OrderedDict[str, Union[ImportClause, ComponentRef]]
        self.extends = []  # type: List[ExtendsClause]
        self.encapsulated = False  # type: bool
        self.partial = False  # type: bool
        self.final = False  # type: bool
        self.type = ""  # type: str
        self.comment = ""  # type: str
        self.classes = OrderedDict()  # type: OrderedDict[str, Class]
        self.symbols = OrderedDict()  # type: OrderedDict[str, Symbol]
        self.functions = OrderedDict()  # type: OrderedDict[str, Class]
        self.initial_equations = []  # type: List[Union[Equation, ForEquation]]
        self.equations = []  # type: List[Union[Equation, ForEquation, ConnectClause]]
        self.initial_statements = (
            []
        )  # type: List[Union[AssignmentStatement, IfStatement, ForStatement]]
        self.statements = []  # type: List[Union[AssignmentStatement, IfStatement, ForStatement]]
        self.annotation = []  # type: Union[Type[None], ClassModification]
        self.parent = None  # type: Class

        super().__init__(**kwargs)

    def _find_class(
        self, component_ref: ComponentRef, search_parent=True, search_imports=True
    ) -> "Class":
        """Recursively search for component_ref in self and linked classes

        Implement lookup rules per spec chapter 5, see also chapter 13.
        This is more succinctly outlined in
        https://mbe.modelica.university/components/packages/lookup/
        """
        # TODO: Get rid of exception-based algorithm or make imports have separate exception?
        # TODO: Move import logic to separate function?
        # TODO: Implement library path lookup from section 13.2.2 of spec
        # TODO: Try @functools.lru_cache if profile shows this is hotspot
        try:
            if not component_ref.child:
                return self.classes[component_ref.name]
            else:
                # Avoid infinite recursion by passing search_parent = False
                return self.classes[component_ref.name]._find_class(component_ref.child[0], False)
        except (KeyError, ClassNotFoundError):
            try:
                if search_imports:
                    if component_ref.name in self.imports:
                        # First search qualified imports (most common case)
                        import_ = self.imports[
                            component_ref.name
                        ]  # type: Union[ImportClause, ComponentRef]
                        if isinstance(import_, ImportClause):
                            # Expand short name
                            if component_ref.child:
                                import_ = import_.components[0].concatenate(component_ref.child[0])
                            else:
                                import_ = import_.components[0]
                        elif component_ref.child:
                            import_ = import_.concatenate(component_ref.child[0])
                        return self._find_class(import_)
                    else:
                        # Next search packages for unqualified imports (slow, but assuming not common)
                        if "*" in self.imports:
                            c = None
                            for package_ref in self.imports["*"].components:
                                imported_comp_ref = package_ref.concatenate(
                                    ComponentRef(name=component_ref.name)
                                )
                                # Search within the package
                                try:
                                    # Avoid infinite recursion with search_imports = False
                                    c = self._find_class(imported_comp_ref, search_imports=False)
                                except (KeyError, ClassNotFoundError):
                                    pass
                            if c is not None:
                                # Store result for next lookup
                                self.imports[component_ref.name] = imported_comp_ref
                                return c
                            else:
                                raise ClassNotFoundError
                        else:
                            raise ClassNotFoundError
                else:
                    raise ClassNotFoundError
            except (KeyError, ClassNotFoundError):
                if search_parent and self.parent is not None and not self.encapsulated:
                    return self.parent._find_class(component_ref)
                else:
                    raise ClassNotFoundError("Could not find class '{}'".format(component_ref))

    def find_class(
        self,
        component_ref: ComponentRef,
        copy=True,
        check_builtin_classes=False,
        search_imports=True,
    ) -> "Class":
        if component_ref.name in self.BUILTIN:
            if check_builtin_classes:
                type_ = component_ref.name

                c = Class(name=type_)
                c.type = "__builtin"
                c.parent = self.root

                cref = ComponentRef(name=type_)
                s = Symbol(name="__value", type=cref)
                c.symbols[s.name] = s

                return c
            else:
                raise FoundElementaryClassError()

        c = self._find_class(component_ref, search_imports)

        if copy:
            c = c.copy_including_children()

        return c

    def _find_constant_symbol(self, component_ref: ComponentRef, search_parent=True) -> Symbol:
        if component_ref.child:
            # Try classes first, and constant symbols second
            t = component_ref.to_tuple()

            try:
                node = self._find_class(ComponentRef(name=t[0]), search_parent)
                return node._find_constant_symbol(ComponentRef.from_tuple(t[1:]), False)
            except ClassNotFoundError:
                try:
                    s = self.symbols[t[0]]
                except KeyError:
                    raise ConstantSymbolNotFoundError()

                if "constant" not in s.prefixes:
                    raise ConstantSymbolNotFoundError()

                # Found a symbol. Continue lookup on type of this symbol.
                if isinstance(s.type, InstanceClass):
                    return s.type._find_constant_symbol(ComponentRef.from_tuple(t[1:]), False)
                elif isinstance(s.type, ComponentRef):
                    node = self._find_class(s.type)  # Parent lookups is OK here.
                    return node._find_constant_symbol(ComponentRef.from_tuple(t[1:]), False)
                else:
                    raise Exception("Unknown object type of symbol type: {}".format(type(s.type)))
        else:
            try:
                return self.symbols[component_ref.name]
            except KeyError:
                raise ConstantSymbolNotFoundError()

    def find_constant_symbol(self, component_ref: ComponentRef) -> Symbol:
        return self._find_constant_symbol(component_ref)

    def full_reference(self):
        names = []

        c = self
        while True:
            names.append(c.name)
            if c.parent is None:
                break
            else:
                c = c.parent

        # Exclude the root node's name
        return ComponentRef.from_tuple(tuple(reversed(names[:-1])))

    def _extend(self, other: "Class") -> None:
        for class_name in other.classes.keys():
            if class_name in self.classes.keys():
                self.classes[class_name]._extend(other.classes[class_name])
            else:
                self.classes[class_name] = other.classes[class_name]

    @property
    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root

    def copy_including_children(self):
        return copy.deepcopy(self)

    def add_class(self, c: "Class") -> None:
        """
        Add a (sub)class to this class.

        :param c: (Sub)class to add.
        """
        self.classes[c.name] = c
        c.parent = self

    def remove_class(self, c: "Class") -> None:
        """
        Removes a (sub)class from this class.

        :param c: (Sub)class to remove.
        """
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
        self.__deepcopy__ = None
        new = copy.deepcopy(self, memo)
        self.__deepcopy__ = _deepcp
        new.__deepcopy__ = _deepcp
        return new

    def __repr__(self):
        return "{}(type={!r}, name={!r})".format(type(self).__name__, self.type, self.name)

    def __str__(self):
        return '{} {}, Type "{}"'.format(type(self).__name__, self.name, self.type)


class InstanceClass(Class):
    """
    Class used during instantiation/expansion of the model. Modififcations on
    symbols and extends clauses are shifted to the modification environment of
    this InstanceClass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modification_environment = ClassModification()

    def __repr__(self):
        return "{}(type={!r}, name={!r}, modification_environment={!r})".format(
            type(self).__name__, self.type, self.name, self.modification_environment
        )


class Tree(Class):
    """
    The root class.
    """

    def extend(self, other: "Tree") -> None:
        self._extend(other)
        self.update_parent_refs()

    def _update_parent_refs(self, parent: Class) -> None:
        for c in parent.classes.values():
            c.parent = parent
            self._update_parent_refs(c)

    def update_parent_refs(self) -> None:
        self._update_parent_refs(self)

    def __repr__(self):
        return "{}(classes={!r})".format(type(self).__name__, self.classes)
