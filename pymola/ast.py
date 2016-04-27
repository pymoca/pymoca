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
        if default is None:
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
                raise IOError('{:s}.{:s} requires types ({:s}), but got {:s}'.format(
                    name, key, ','.join([t.__name__ for t in self.types]), type(val).__name__))
            else:
                return False
        else:
            return True


class List(list):
    def __init__(self, types, default=[]):
        super(List, self).__init__()
        if type(types) is type or not hasattr(types, '__iter__'):
            types = [types]
        types = list(types)
        if sys.version_info < (3,):
            if str in types:
                types += [unicode]
        self.types = types
        self.default = default

    def validate(self, name, key, val_list, throw=True):
        if type(val_list) != list:
            if throw:
                raise IOError('{:s}.{:s} requires types ({:s}), but got {:s}'.format(
                name, key, 'list', type(val).__name__))
            else:
                return False

        for val in val_list:
            if not type(val) in self.types:
                if throw:
                    raise IOError('{:s}.{:s} requires list items of type ({:s}), but got {:s}'.format(
                        name, key, ','.join([t.__name__ for t in self.types]), type(val).__name__))
                else:
                    return False
            else:
                return True

    def __add__(self, other):
        if other in self.types:
            super(List, self).__add__(other)
        else:
            raise IOError('List requires elements of type', self.types)


class Dict(dict):
    def __init__(self, types, default=[]):
        super(Dict, self).__init__()
        if type(types) is type or not hasattr(types, '__iter__'):
            types = [types]
        types = list(types)
        if sys.version_info < (3,):
            if str in types:
                types += [unicode]
        self.types = types
        self.default = default

    def validate(self, name, key, val_dict, throw=True):
        if type(val_dict) != dict:
            if throw:
                raise IOError('{:s}.{:s} requires types ({:s}), but got {:s}'.format(
                name, key, 'dict', type(val).__name__))
            else:
                return False

        for val in val_dict.values():
            print('dict val', val)
            if not type(val) in self.types:
                if throw:
                    raise IOError('{:s}.{:s} requires list items of type ({:s}), but got {:s}'.format(
                        name, key, ','.join([t.__name__ for t in self.types]), type(val).__name__))
                else:
                    return False
            else:
                return True


class Node(object):
    ast_spec = {}

    def __init__(self, **kwargs):
        for key in self.ast_spec.keys():
            # make sure we don't share ast_spec default data by using deep copy
            self.__dict__[key] = copy.deepcopy(self.ast_spec[key].default)
        for key in kwargs.keys():
            # types = self.ast_spec[key].types
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

    def __init__(self, **kwargs):
        super(Primary, self).__init__(**kwargs)


class ComponentRef(Node):

    def __init__(self, **kwargs):
        super(ComponentRef, self).__init__(**kwargs)


class Expression(Node):

    def __init__(self, **kwargs):
        super(Expression, self).__init__(**kwargs)


class Equation(Node):

    def __init__(self, **kwargs):
        super(Equation, self).__init__(**kwargs)


class ConnectClause(Node):

    def __init__(self, **kwargs):
        super(ConnectClause, self).__init__(**kwargs)


class Symbol(Node):
    def __init__(self, **kwargs):
        super(Symbol, self).__init__(**kwargs)


class ComponentClause(Node):
    def __init__(self, **kwargs):
        super(ComponentClause, self).__init__(**kwargs)


class EquationSection(Node):
    def __init__(self, **kwargs):
        super(EquationSection, self).__init__(**kwargs)


class Class(Node):
    def __init__(self, **kwargs):
        super(Class, self).__init__(**kwargs)


class File(Node):
    def __init__(self, **kwargs):
        super(File, self).__init__(**kwargs)


# Here we define the AST specifications for all nodes
# these are static variables shared between class instances
# and are defined here to allow a class to list itself in
# the allowed field types, this self referencing is not
# possible when initially declaring a class in python

Primary.ast_spec = {
    'value': Field((bool, float, int, str))
}

ComponentRef.ast_spec = {
    'name': Field(str),
}

Expression.ast_spec = {
    'operator': Field(str),
    'operands': List((Expression, ComponentRef)),
}

Equation.ast_spec = {
    'left': Field((Expression, Primary, ComponentRef)),
    'right': Field((Expression, Primary, ComponentRef)),
    'comment': Field(str),
}

ConnectClause.ast_spec = {
    'left': Field(ComponentRef),
    'right': Field(ComponentRef),
    'comment': Field(str),
}

Symbol.ast_spec = {
    'name': Field(str, ''),
    'type': Field(str, ''),
    'prefixes': List((str), []),
    'redeclare': Field(bool, False),
    'final': Field(bool, False),
    'inner': Field(bool, False),
    'outer': Field(bool, False),
    'dimensions': Field(list, [1]),
    'comment': Field(str, ''),
    'start': Field(Primary),
    'id': Field(int),
    'order': Field(int),
}

ComponentClause.ast_spec = {
    'prefixes': Field(list, []),
    'type': Field(str, ''),
    'dimensions': List((int), [1]),
    'comment': List((str), []),
    'symbol_list': List((Symbol), []),
}

EquationSection.ast_spec = {
    'initial': Field(bool, False),
    'equations': List((Equation, ConnectClause), []),
}

Class.ast_spec = {
    'name': Field(str),
    'encapsulated': Field(bool, False),
    'partial': Field(bool, False),
    'final': Field(bool, False),
    'type': Field(str, ''),
    'comment': Field(str, ''),
    'symbols': Dict((ComponentClause), {}),
    'initial_equations': List((Equation), []),
    'equations': List((Equation, ConnectClause), []),
}

File.ast_spec = {
    'within': Field(str),
    'classes': Dict((Class), {}),
}


