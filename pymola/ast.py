#!/usr/bin/env python
"""
Modelica AST definitions
"""
from __future__ import print_function, absolute_import, division, print_function, unicode_literals

import copy
import json
import sys

VALIDATE_AST = True

"""
AST Node Type Hierarchy

File
    Class
        Equation
            ComponentRef
            Expression
            Primary
        ConnectClause
            ComponentRef
        Symbol
"""

def to_json(var):
    if isinstance(var, list):
        res = [to_json(item) for item in var]
    elif isinstance(var, dict):
        res = {key: to_json(var[key]) for key in var.keys()}
    elif isinstance(var, Node):
        res = {key: to_json(var.__dict__[key]) for key in var.__dict__.keys()}
    else:
        res = var
    return res


class Field(object):
    def __init__(self, types, default=None):
        if default == None:
            if types == dict:
                default = {}
            elif types == list:
                default = []
        if type(types) is type or not hasattr(types, '__iter__'):
            types = [types]
        types = list(types)
        if sys.version_info < (3,):
            if str in types:
                types += [unicode]
        self.types = types
        self.default = default

    def validate(self, name, key, val, throw=True):
        if not type(val) in self.types:
            if throw:
                raise IOError('{:s}.{:s} requires types {:s}, but got {:s}'.format(
                    name, key, self.types, type(val)))
            else:
                return False
        else:
            return True


# TODO, need to finish implementing this for list type checking
class List(list):
    def __init__(self, types):
        super(List, self).__init__()
        if not isinstance(types, list):
            types = list([types])
        self.types = types

    def __add__(self, other):
        if other in self.types:
            super(List, self).__add__(other)
        else:
            raise IOError('List requires elements of type', self.types)


class Node(object):
    ast_spec = {}

    def __init__(self, **kwargs):
        for key in self.ast_spec.keys():
            # make sure we don't share ast_spec default data by using deep copy
            self.__dict__[key] = copy.deepcopy(self.ast_spec[key].default)
        for key in kwargs.keys():
            types = self.ast_spec[key].types
            val = kwargs[key]
            self.set_field(key, val)

    def set_field(self, key, value):
        if VALIDATE_AST:
            name = self.__class__.__name__
            if key not in self.ast_spec.keys():
                raise IOError('{:s} not a child of {:s}'.format(key, name))
            self.ast_spec[key].validate(name, key, value)
        self.__dict__[key] = value

    def __setattr__(self, key, value):
        self.set_field(key, value)

    def __repr__(self):
        return json.dumps(to_json(self), indent=2, sort_keys=True)

    __str__ = __repr__


class Primary(Node):
    ast_spec = {
        'value': Field((bool, float, int, str))
    }

    def __init__(self, **kwargs):
        super(Primary, self).__init__(**kwargs)


class ComponentRef(Node):
    ast_spec = {
        'name': Field(str),
    }

    def __init__(self, **kwargs):
        super(ComponentRef, self).__init__(**kwargs)


# so expression can reference itself
class Expression(Node):
    pass


# noinspection PyRedeclaration
class Expression(Node):
    ast_spec = {
        'operator': Field(str),
        'operands': Field(list),  # list of Expressions
    }

    def __init__(self, **kwargs):
        super(Expression, self).__init__(**kwargs)


class Equation(Node):
    ast_spec = {
        'left': Field((Expression, Primary, ComponentRef)),
        'right': Field((Expression, Primary, ComponentRef)),
        'comment': Field(str),
    }

    def __init__(self, **kwargs):
        super(Equation, self).__init__(**kwargs)


class ConnectClause(Node):
    ast_spec = {
        'left': Field(ComponentRef),
        'right': Field(ComponentRef),
        'comment': Field(str),
    }

    def __init__(self, **kwargs):
        super(ConnectClause, self).__init__(**kwargs)


class Symbol(Node):
    ast_spec = {
        'name': Field(str, ''),
        'type': Field(str, ''),
        'prefixes': Field(list, []),  # (str)
        'redeclare': Field(bool, False),
        'final': Field(bool, False),
        'inner': Field(bool, False),
        'outer': Field(bool, False),
        'dimensions': Field(list, [1]),  # (int)
        'comment': Field(str, ''),
        'start': Field(Primary, ''),
    }

    def __init__(self, **kwargs):
        super(Symbol, self).__init__(**kwargs)


class ComponentClause(Node):
    ast_spec = {
        'prefixes': Field(list, []),  # (str)
        'type': Field(str, ''),
        'dimensions': Field(list, [1]),  # (int)
        'comment': Field(str, ''),
        'symbol_list': Field(list, []),  # (Symbol)
    }

    def __init__(self, **kwargs):
        super(ComponentClause, self).__init__(**kwargs)


class EquationSection(Node):
    ast_spec = {
        'initial': Field(bool, False),
        'equations': Field(list, []),  # (Equation)
    }

    def __init__(self, **kwargs):
        super(EquationSection, self).__init__(**kwargs)


class Class(Node):
    ast_spec = {
        'name': Field(str),
        'encapsulated': Field(bool, False),
        'partial': Field(bool, False),
        'final': Field(bool, False),
        'type': Field(str, ''),
        'comment': Field(str, ''),
        'symbols': Field(dict, {}),  # (Symbol)
        'initial_equations': Field(list, []),
        'equations': Field(list, []),  # (Equation)
        'parameters': Field(list, []),  # (ComponentRef)
        'constants': Field(list, []),  # (ComponentRef)
        'inputs': Field(list, []),  # (ComponentRef)
        'outputs': Field(list, []),  # (ComponentRef)
        'states': Field(list, []),  # (ComponentRef)
        'variables': Field(list, []),  # (ComponentRef)
    }

    def __init__(self, **kwargs):
        super(Class, self).__init__(**kwargs)


class File(Node):
    ast_spec = {
        'within': Field(str),
        'classes': Field(dict),
    }

    def __init__(self, **kwargs):
        super(File, self).__init__(**kwargs)
