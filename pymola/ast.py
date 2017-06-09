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


class Node(object):
    def __init__(self, **kwargs):
        self.set_args(**kwargs)

    def set_args(self, **kwargs):
        for key in kwargs.keys():
            if key not in self.__dict__.keys():
                raise KeyError('{:s} not valid arg'.format(key))
            self.__dict__[key] = kwargs[key]

    def __repr__(self):
        return json.dumps(self.to_json(self), indent=2, sort_keys=True)

    @classmethod
    def to_json(cls, var):
        if isinstance(var, list):
            res = [cls.to_json(item) for item in var]
        elif isinstance(var, dict):
            res = {key: cls.to_json(var[key]) for key in var.keys()}
        elif isinstance(var, Node):
            res = {key: cls.to_json(var.__dict__[key]) for key in var.__dict__.keys()}
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


class Array(Node):
    def __init__(self, **kwargs):
        self.values = []  # type: List[Union[Expression, Primary, ComponentRef, Array]]
        super().__init__(**kwargs)


class Slice(Node):
    def __init__(self, **kwargs):
        self.start = Primary(value=0)  # type: Union[Expression, Primary, ComponentRef]
        self.stop = Primary(value=-1)  # type: Union[Expression, Primary, ComponentRef]
        self.step = Primary(value=1)  # type: Union[Expression, Primary, ComponentRef]
        super().__init__(**kwargs)


class ComponentRef(Node):
    def __init__(self, **kwargs):
        self.name = ''  # type: str
        self.indices = []  # type: List[Union[Expression, Slice, Primary, ComponentRef]]
        self.child = []  # type: List[ComponentRef]
        super().__init__(**kwargs)


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
        self.left = None  # type: Union[Expression, Primary, ComponentRef]
        self.right = None  # type: Union[Expression, Primary, ComponentRef]
        self.comment = ''  # type: str
        super().__init__(**kwargs)


class IfEquation(Node):
    def __init__(self, **kwargs):
        self.conditions = []  # type: List[Union[Expression, Primary, ComponentRef]]
        self.equations = []  # type: List[Union[Expression, ForEquation, ConnectClause, IfEquation]]
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
        self.expressions = []  # type: List[Union[Expression, Primary, ComponentRef]]
        self.statements = []  # type: List[Union[AssignmentStatement, ForStatement]]
        self.comment = ''  # type: str
        super().__init__(**kwargs)


class ForStatement(Node):
    def __init__(self, **kwargs):
        self.indices = []  # type: List[ForIndex]
        self.statements = []  # type: List[Union[AssignmentStatement, ForStatement]]
        self.comment = ''  # type: str
        super().__init__(**kwargs)


class Symbol(Node):
    """
    A mathematical variable or state of the model
    """
    ATTRIBUTES = ['value', 'min', 'max', 'start', 'fixed', 'nominal']

    def __init__(self, **kwargs):
        self.name = ''  # type: str
        self.type = ComponentRef()  # type: ComponentRef
        self.prefixes = []  # type: List[str]
        self.redeclare = False  # type: bool
        self.final = False  # type: bool
        self.inner = False  # type: bool
        self.outer = False  # type: bool
        self.dimensions = [Primary(value=1)]  # type: List[Union[Expression, Primary, ComponentRef]]
        self.comment = ''  # type: str
        # params start value is 0 by default from Modelica spec
        self.start = Primary(value=0)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.min = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.max = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.nominal = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.value = Primary(value=None)  # type: Union[Expression, Primary, ComponentRef, Array]
        self.fixed = Primary(value=False)  # type: Primary
        self.id = 0  # type: int
        self.order = 0  # type: int
        self.visibility = Visibility.PRIVATE  # type: Visibility
        self.class_modification = None  # type: ClassModification
        super().__init__(**kwargs)


class ComponentClause(Node):
    def __init__(self, **kwargs):
        self.prefixes = []  # type: List[str]
        self.type = ComponentRef()  # type: ComponentRef
        self.dimensions = [Primary(value=1)]  # type: List[Union[Expression, Primary, ComponentRef]]
        self.comment = []  # type: List[str]
        self.symbol_list = []  # type: List[Symbol]
        super().__init__(**kwargs)


class EquationSection(Node):
    def __init__(self, **kwargs):
        self.initial = False  # type: bool
        self.equations = []  # type: List[Union[Equation, ForEquation, ConnectClause]]
        super().__init__(**kwargs)


class AlgorithmSection(Node):
    def __init__(self, **kwargs):
        self.initial = False  # type: bool
        self.statements = []  # type: List[Union[AssignmentStatement, ForStatement]]
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
    # TODO: Check if ComponentRef modifiers are handled correctly. For example,
    # check HomotopicLinear which extends PartialHomotopic with the modifier
    # "H(min = H_b)".
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
        self.arguments = []  # type: List[Union[ElementModification, ComponentClause, ShortClassDefinition]]
        super().__init__(**kwargs)


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
        self.symbols = OrderedDict()  # type: OrderedDict[str, Symbol]
        self.initial_equations = []  # type: List[Union[Equation, ForEquation]]
        self.equations = []  # type: List[Union[Equation, ForEquation, ConnectClause]]
        self.initial_statements = []  # type: List[Union[AssignmentStatement, ForStatement]]
        self.statements = []  # type: List[Union[AssignmentStatement, ForStatement]]
        self.within = []  # type: List[ComponentRef]
        super().__init__(**kwargs)


class File(Node):
    """
    Represents a .mo file for use in pre-processing before flattening to a single class.
    """

    def __init__(self, **kwargs):
        self.within = []  # type: List[ComponentRef]
        self.classes = OrderedDict()  # type: OrderedDict[str, Class]
        super().__init__(**kwargs)

    def find_class(self, component_ref: ComponentRef) -> Class:
        """
        Find the class that a component is defined in
        :param component_ref: component reference
        :return: the class that contains the ref
        """
        name = component_ref.name
        for c_name in self.classes.keys():
            c = self.classes[c_name]
            if component_ref.name in c.symbols.keys():
                return c
        raise KeyError('symbol {:s} not found'.format(name))

    def find_symbol(self, c: Class, component_ref: ComponentRef) -> Symbol:
        """
        Given a component ref, lookup the symbol in the given class
        :param c: class to look in
        :param component_ref: component reference
        :return: the symbol
        """
        sym = c.symbols[component_ref.name]
        if len(component_ref.child) > 0:
            c = self.find_class(sym.type)
            return self.find_symbol(c, component_ref.child[0])
        else:
            return sym


class Collection(Node):
    """
    A list of modelica files, used in pre-processing packages etc. before flattening
    to a single class.
    """

    def __init__(self, **kwargs):
        self.files = []  # type: List[File]
        super().__init__(**kwargs)

    def extend(self, other):
        self.files.extend(other.files)

    def find_class(self, component_ref: Union[ComponentRef, str], within: list = None):

        if within is None:
            within = []

        if isinstance(component_ref, str):
            assert component_ref.find('.') == -1
            component_ref = ComponentRef(name=component_ref)

        # First we try to the find the file matching the right 'within'
        if not component_ref.child and not within:
            for f in self.files:
                if not f.within:
                    try:
                        return f.classes[component_ref.name]
                    except KeyError:
                        continue

            # Could not find symbol. Assume it is an elementary type
            # TODO: Is this a correct assumption? What if we have an undefined type that is not elementary?
            raise KeyError(component_ref.name)
        else:
            # TODO: Should we move this to a 'get_parent' method in the ComponentRef class
            c_within = copy.deepcopy(component_ref)

            n = c_within
            while n.child[0].child:
                n = n.child[0]
            class_name = n.child[0].name
            n.child = []

            # Merge the within passed in, and the within we split from the
            # class to be looked up
            extended_within = copy.deepcopy(within)

            if extended_within:
                n = extended_within[0]
                while n.child:
                    n = n.child[0]
                n.child.append(c_within)
                extended_within = extended_within[0]
            else:
                extended_within = c_within

            c = next((f.classes[class_name] for f in self.files if
                      f.within and compare_component_ref(f.within[0], extended_within) and class_name in f.classes),
                     None)

            # TODO: This could probably be cleaner if we do nested classes.
            # Then we could traverse up the tree until we found a match,
            # instead of just trying twice (once with, once without prepending
            # the passed-in 'within').
            if c is None:
                # Try again with root node lookup instead of relative
                c = next((f.classes[class_name] for f in self.files if
                          f.within and compare_component_ref(f.within[0], c_within) and class_name in f.classes), None)
                if c is None:
                    # TODO: How long do we traverse? Do we somehow force a stop at Real, Boolean, etc?
                    #       Now a force is stopped on anything in the Modelica library.
                    # FIXME: The "SI" part should be removed when we can handle import statements.
                    if c_within.name in ("Modelica", "SI"):
                        raise KeyError(c_within.name)
                    else:
                        raise Exception("Could not find class {} in {}".format(class_name, c_within))
                else:
                    return c
            else:
                return c

    def find_symbol(self, node, component_ref: ComponentRef) -> Symbol:
        sym = node.symbols[component_ref.name]
        if len(component_ref.child) > 0:
            node = self.find_class(sym.type)
            return self.find_symbol(node, component_ref.child[0])
        else:
            return sym


def compare_component_ref(this: ComponentRef, other: ComponentRef) -> bool:
    """
    Helper function to compare component references to each other without converting to JSON
    :param this:
    :param other:
    :return: boolean, true if match
    """
    if len(this.child) != len(other.child):
        return False

    if this.child and other.child:
        return compare_component_ref(this.child[0], other.child[0])

    return this.__dict__ == other.__dict__
