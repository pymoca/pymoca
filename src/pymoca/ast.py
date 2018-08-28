#!/usr/bin/env python
"""
Modelica AST definitions
"""
from __future__ import print_function, absolute_import, division, print_function, unicode_literals

import copy
import json
from enum import Enum
from typing import List, Union, Dict
from collections import OrderedDict


class ClassNotFoundError(Exception):
    pass

class ConstantSymbolNotFoundError(Exception):
    pass

class FoundElementaryClassError(Exception):
    pass


class Visibility(Enum):
    PRIVATE = 0, 'private'
    PROTECTED = 1, 'protected'
    PUBLIC = 2, 'public'

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


nan = float('nan')

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
                raise KeyError('{:s} not valid arg'.format(key))
            self.__dict__[key] = kwargs[key]

    def __repr__(self):
        d = self.to_json(self)
        d['_type'] = self.__class__.__name__
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
            res = {key: cls.to_json(var.__dict__[key]) for key in var.__dict__.keys()
                   if key not in ('parent', 'scope', '__deepcopy__')}
        elif isinstance(var, Visibility):
            res = str(var)
        else:
            res = var
        return res

    __str__ = __repr__


class Primary(Node):
    def __init__(self, **kwargs):
        self.value = None  # type: Union[bool, float, int, str, type(None)]
        super().__init__(**kwargs)

    def __str__(self):
        return '{} value {}'.format(type(self).__name__, self.value)


class Array(Node):
    def __init__(self, **kwargs):
        self.values = []  # type: List[Union[Expression, Primary, ComponentRef, Array]]
        super().__init__(**kwargs)

    def __str__(self):
        return '{} {}'.format(type(self).__name__, self.values)


class Slice(Node):
    def __init__(self, **kwargs):
        self.start = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef]
        self.stop = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef]
        self.step = Primary(value=1)  # type: Union[Expression, Primary, ComponentRef]
        super().__init__(**kwargs)

    def __str__(self):
        return '{} start: {}, stop: {}, step: {}'.format(
            type(self).__name__, self.start, self.stop, self.step)


class ComponentRef(Node):
    def __init__(self, **kwargs):
        self.name = ''  # type: str
        self.indices = [[None]]  # type: List[List[Union[Expression, Slice, Primary, ComponentRef]]]
        self.child = []  # type: List[ComponentRef]
        super().__init__(**kwargs)

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
            return (self.name, ) + self.child[0].to_tuple()
        else:
            return self.name,

    @classmethod
    def from_tuple(cls, components: tuple) -> 'ComponentRef':
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
    def from_string(cls, s: str) -> 'ComponentRef':
        """
        Convert the string pointing to a component using dot notation to
        a component reference.
        :param s: string pointing to component using dot notation
        :return: ComponentRef
        """

        components = s.split('.')
        return cls.from_tuple(components)

    @classmethod
    def concatenate(cls, *args: List['ComponentRef']) -> 'ComponentRef':
        """
        Helper function to append two component references to eachother, e.g.
        a "within" component ref and an "object type" component ref.
        :return: New component reference, with other appended to self.
        """

        a = copy.deepcopy(args[0])
        n = a
        for b in args[1:]:
            while n.child:
                n = n.child[0]
            b = copy.deepcopy(b)  # Not strictly necessary
            n.child = [b]
        return a


class Expression(Node):
    def __init__(self, **kwargs):
        self.operator = None  # type: Union[str, ComponentRef]
        self.operands = []  # type: List[Union[Expression, Primary, ComponentRef, Array, IfExpression]]
        super().__init__(**kwargs)


class IfExpression(Node):
    def __init__(self, **kwargs):
        self.conditions = []  # type: List[Union[Expression, Primary, ComponentRef, Array, IfExpression]]
        self.expressions = []  # type: List[Union[Expression, Primary, ComponentRef, Array, IfExpression]]
        super().__init__(**kwargs)


class Equation(Node):
    def __init__(self, **kwargs):
        self.left = None  # type: Union[Expression, Primary, ComponentRef, List[Union[Expression, Primary, ComponentRef]]]
        self.right = None  # type: Union[Expression, Primary, ComponentRef, List[Union[Expression, Primary, ComponentRef]]]
        self.comment = ''  # type: str
        super().__init__(**kwargs)


class IfEquation(Node):
    def __init__(self, **kwargs):
        self.conditions = []  # type: List[Union[Expression, Primary, ComponentRef]]
        self.blocks = []  # type: List[List[Union[Expression, ForEquation, ConnectClause, IfEquation]]]
        self.comment = ''  # type: str
        super().__init__(**kwargs)


class WhenEquation(Node):
    def __init__(self, **kwargs):
        self.conditions = []  # type: List[Union[Expression, Primary, ComponentRef]]
        self.blocks = []  # type: List[List[Union[Expression, ForEquation, ConnectClause, IfEquation]]]
        self.comment = ''  # type: str
        super().__init__(**kwargs)


class ForIndex(Node):
    def __init__(self, **kwargs):
        self.name = ''  # type: str
        self.expression = None  # type: Union[Expression, Primary, Slice]
        super().__init__(**kwargs)


class ForEquation(Node):
    def __init__(self, **kwargs):
        self.indices = []  # type: List[ForIndex]
        self.equations = []  # type: List[Union[Equation, ForEquation, ConnectClause]]
        self.comment = None  # type: str
        super().__init__(**kwargs)


class ConnectClause(Node):
    def __init__(self, **kwargs):
        self.left = ComponentRef()  # type: ComponentRef
        self.right = ComponentRef()  # type: ComponentRef
        self.comment = ''  # type: str
        super().__init__(**kwargs)


class AssignmentStatement(Node):
    def __init__(self, **kwargs):
        self.left = []  # type: List[ComponentRef]
        self.right = None  # type: Union[Expression, IfExpression, Primary, ComponentRef]
        self.comment = ''  # type: str
        super().__init__(**kwargs)


class IfStatement(Node):
    def __init__(self, **kwargs):
        self.conditions = []  # type: List[Union[Expression, Primary, ComponentRef]]
        self.blocks = []  # type: List[List[Union[AssignmentStatement, IfStatement, ForStatement]]]
        self.comment = ''  # type: str
        super().__init__(**kwargs)


class WhenStatement(Node):
    def __init__(self, **kwargs):
        self.conditions = []  # type: List[Union[Expression, Primary, ComponentRef]]
        self.blocks = []  # type: List[List[Union[AssignmentStatement, IfStatement, ForStatement]]]
        self.comment = ''  # type: str
        super().__init__(**kwargs)


class ForStatement(Node):
    def __init__(self, **kwargs):
        self.indices = []  # type: List[ForIndex]
        self.statements = []  # type: List[Union[AssignmentStatement, IfStatement, ForStatement]]
        self.comment = ''  # type: str
        super().__init__(**kwargs)


class Function(Node):
    def __init__(self, **kwargs):
        self.name = '' # type: str
        self.arguments = []  # type: List[Union[Expression, Primary, ComponentRef, Array]]
        self.comment = ''  # type: str
        super().__init__(**kwargs)


class Symbol(Node):
    """
    A mathematical variable or state of the model
    """
    ATTRIBUTES = ['value', 'min', 'max', 'start', 'fixed', 'nominal', 'unit']

    def __init__(self, **kwargs):
        self.name = ''  # type: str
        self.type = ComponentRef()  # type: Union[ComponentRef, InstanceClass]
        self.prefixes = []  # type: List[str]
        self.redeclare = False  # type: bool
        self.final = False  # type: bool
        self.inner = False  # type: bool
        self.outer = False  # type: bool
        self.dimensions = [Primary(value=None)]  # type: List[Union[Expression, Primary, ComponentRef]]
        self.comment = ''  # type: str
        # params start value is 0 by default from Modelica spec
        self.start = Primary(value=0)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.min = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.max = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.nominal = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.value = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.fixed = Primary(value=False)  # type: Primary
        self.unit = Primary(value="")  # type: Primary
        self.id = 0  # type: int
        self.order = 0  # type: int
        self.visibility = Visibility.PRIVATE  # type: Visibility
        self.class_modification = None  # type: ClassModification
        super().__init__(**kwargs)

    def __str__(self):
        return '{} {}, Type "{}"'.format(type(self).__name__, self.name, self.type)


class ComponentClause(Node):
    def __init__(self, **kwargs):
        self.prefixes = []  # type: List[str]
        self.type = ComponentRef()  # type: ComponentRef
        self.dimensions = [Primary(value=None)]  # type: List[Union[Expression, Primary, ComponentRef]]
        self.comment = []  # type: List[str]
        self.symbol_list = []  # type: List[Symbol]
        super().__init__(**kwargs)


class EquationSection(Node):
    def __init__(self, **kwargs):
        self.initial = False  # type: bool
        self.equations = []  # type: List[Union[Equation, IfEquation, ForEquation, ConnectClause]]
        super().__init__(**kwargs)


class AlgorithmSection(Node):
    def __init__(self, **kwargs):
        self.initial = False  # type: bool
        self.statements = []  # type: List[Union[AssignmentStatement, IfStatement, ForStatement]]
        super().__init__(**kwargs)


class ImportAsClause(Node):
    def __init__(self, **kwargs):
        self.component = ComponentRef()  # type: ComponentRef
        self.name = ''  # type: str
        super().__init__(**kwargs)


class ImportFromClause(Node):
    def __init__(self, **kwargs):
        self.component = ComponentRef()  # type: ComponentRef
        self.symbols = []  # type: List[str]
        super().__init__(**kwargs)


class ElementModification(Node):
    def __init__(self, **kwargs):
        self.component = ComponentRef()  # type: Union[ComponentRef]
        self.modifications = []  # type: List[Union[Primary, Expression, ClassModification, Array, ComponentRef]]
        super().__init__(**kwargs)


class ShortClassDefinition(Node):
    def __init__(self, **kwargs):
        self.name = ''  # type: str
        self.type = ''  # type: str
        self.component = ComponentRef()  # type: ComponentRef
        self.class_modification = ClassModification()  # type: ClassModification
        super().__init__(**kwargs)


class ElementReplaceable(Node):
    def __init__(self, **kwargs):
        # TODO, add fields ?
        super().__init__(**kwargs)


class ClassModification(Node):
    def __init__(self, **kwargs):
        self.arguments = []  # type: List[ClassModificationArgument]
        super().__init__(**kwargs)


class ClassModificationArgument(Node):
    def __init__(self, **kwargs):
        self.value = []  # type: Union[ElementModification, ComponentClause, ShortClassDefinition]
        self.scope = None  # type: InstanceClass
        self.redeclare = False
        super().__init__(**kwargs)

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


class Class(Node):
    def __init__(self, **kwargs):
        self.name = None  # type: str
        self.imports = []  # type: List[Union[ImportAsClause, ImportFromClause]]
        self.extends = []  # type: List[ExtendsClause]
        self.encapsulated = False  # type: bool
        self.partial = False  # type: bool
        self.final = False  # type: bool
        self.type = ''  # type: str
        self.comment = ''  # type: str
        self.classes = OrderedDict()  # type: OrderedDict[str, Class]
        self.symbols = OrderedDict()  # type: OrderedDict[str, Symbol]
        self.functions = OrderedDict()  # type: OrderedDict[str, Class]
        self.initial_equations = []  # type: List[Union[Equation, ForEquation]]
        self.equations = []  # type: List[Union[Equation, ForEquation, ConnectClause]]
        self.initial_statements = []  # type: List[Union[AssignmentStatement, IfStatement, ForStatement]]
        self.statements = []  # type: List[Union[AssignmentStatement, IfStatement, ForStatement]]
        self.annotation = []  # type: Union[NoneType, ClassModification]
        self.parent = None  # type: Class

        super().__init__(**kwargs)

    def _find_class(self, component_ref: ComponentRef, search_parent=True) -> 'Class':
        try:
            if not component_ref.child:
                return self.classes[component_ref.name]
            else:
                # Avoid infinite recursion by passing search_parent = False
                return self.classes[component_ref.name]._find_class(component_ref.child[0], False)
        except (KeyError, ClassNotFoundError):
            if search_parent and self.parent is not None:
                return self.parent._find_class(component_ref)
            else:
                raise ClassNotFoundError("Could not find class '{}'".format(component_ref))

    def find_class(self, component_ref: ComponentRef, copy=True, check_builtin_classes=False) -> 'Class':
        # TODO: Remove workaround for Modelica / Modelica.SIUnits
        if component_ref.name in ["Real", "Integer", "String", "Boolean", "Modelica", "SI"]:
            if check_builtin_classes:
                type_ = component_ref.name
                if component_ref.name in ["Modelica", "SI"]:
                    type_ = "Real"

                c = Class(name=type_)
                c.type = "__builtin"
                c.parent = self.root

                cref = ComponentRef(name=type_)
                s = Symbol(name="__value", type=cref)
                c.symbols[s.name] = s

                return c
            else:
                raise FoundElementaryClassError()

        c = self._find_class(component_ref)

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

                if 'constant' not in s.prefixes:
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

    def _extend(self, other: 'Class') -> None:
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

    def add_class(self, c: 'Class') -> None:
        """
        Add a (sub)class to this class.

        :param c: (Sub)class to add.
        """
        self.classes[c.name] = c
        c.parent = self

    def remove_class(self, c: 'Class') -> None:
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


class Tree(Class):
    """
    The root class.
    """
    def extend(self, other: 'Tree') -> None:
        self._extend(other)
        self.update_parent_refs()

    def _update_parent_refs(self, parent: Class) -> None:
        for c in parent.classes.values():
            c.parent = parent
            self._update_parent_refs(c)

    def update_parent_refs(self) -> None:
        self._update_parent_refs(self)
