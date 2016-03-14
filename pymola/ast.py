#!/usr/bin/env python
"""
Modelica AST definitions
"""
from __future__ import print_function, absolute_import, division, print_function, unicode_literals
import json


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


def field(types, default=None):
    if types == list and default is None:
        default = list()
    elif types == dict and default is None:
        default = dict()
    return {
        'types': types,
        'default': default,
    }


class Node(object):
    ast_spec = {}

    def __init__(self, **kwargs):
        for key in self.ast_spec.keys():
            default = self.ast_spec[key]['default']
            types = self.ast_spec[key]['types']
            # make sure we create new lists nad dicts for each class
            # so they don't share the something passed as default
            if types == list:
                self.__dict__[key] = list(default)
            elif types == dict:
                self.__dict__[key] = dict(default)
            else:
                self.__dict__[key] = default
        for key in kwargs.keys():
            types = self.ast_spec[key]['types']
            val = kwargs[key]
            if key in self.ast_spec.keys():
                # make sure we create new lists nad dicts for each class
                # so they don't share the something passed as default
                if types == list:
                    self.__dict__[key] = list(val)
                elif types == dict:
                    self.__dict__[key] = dict(val)
                else:
                    self.__dict__[key] = val
            else:
                raise IOError('{:s} not a child of {:s}'.format(key, self.__class__.__name__))

    def __setattr__(self, key, value):
        if key not in self.ast_spec:
            raise IOError('{:s} not a child of {:s}'.format(key, self.__class__.__name__))
        else:
            super(Node, self).__setattr__(key, value)

    def __repr__(self):
        return json.dumps(to_json(self), indent=2, sort_keys=True)

    __str__ = __repr__


class Primary(Node):
    ast_spec = {
        'value': field((bool, float, int, str))
    }

    def __init__(self, **kwargs):
        super(Primary, self).__init__(**kwargs)


class ComponentRef(Node):
    ast_spec = {
        'name': field(str),
    }

    def __init__(self, **kwargs):
        super(ComponentRef, self).__init__(**kwargs)


# so expression can reference itself
class Expression(Node):
    pass


# noinspection PyRedeclaration
class Expression(Node):
    ast_spec = {
        'operator': field(str),
        'operands': field((Expression, Primary, ComponentRef)),
    }

    def __init__(self, **kwargs):
        super(Expression, self).__init__(**kwargs)


class Equation(Node):
    ast_spec = {
        'left': field((Expression, Primary, ComponentRef)),
        'right': field((Expression, Primary, ComponentRef)),
        'comment': field(str),
    }

    def __init__(self, **kwargs):
        super(Equation, self).__init__(**kwargs)


class ConnectClause(Node):
    ast_spec = {
        'left': field(ComponentRef),
        'right': field(ComponentRef),
        'comment': field(str),
    }

    def __init__(self, **kwargs):
        super(ConnectClause, self).__init__(**kwargs)


class Symbol(Node):
    ast_spec = {
        'name': field(str, ''),
        'type': field(str, ''),
        'prefixes': field(list, []),
        'redeclare': field(bool, False),
        'final': field(bool, False),
        'inner': field(bool, False),
        'outer': field(bool, False),
        'dimensions': field(list, [1]),
        'comment': field(str, ''),
    }

    def __init__(self, **kwargs):
        super(Symbol, self).__init__(**kwargs)


class ComponentClause(Node):
    ast_spec = {
        'prefixes': field(list, []),
        'type': field(str, ''),
        'dimensions': field(list, [1]),
        'comment': field(str, ''),
        'symbol_list': field(list, []),
    }

    def __init__(self, **kwargs):
        super(ComponentClause, self).__init__(**kwargs)


class EquationSection(Node):
    ast_spec = {
        'initial': field(bool, False),
        'equation_list': field(list, []),
    }

    def __init__(self, **kwargs):
        super(EquationSection, self).__init__(**kwargs)


class Class(Node):
    ast_spec = {
        'name': field(str),
        'encapsulated': field(bool, False),
        'partial': field(bool, False),
        'final': field(bool, False),
        'type': field(str, ''),
        'comment': field(str, ''),
        'symbols': field(dict, {}),
        'equations': field(list, []),
        'parameters': field(list, []),
        'constants': field(list, []),
        'inputs': field(list, []),
        'outputs': field(list, []),
        'states': field(list, []),
    }

    def __init__(self, **kwargs):
        super(Class, self).__init__(**kwargs)


class File(Node):
    ast_spec = {
        'within': field(str),
        'classes': field(dict),
    }

    def __init__(self, **kwargs):
        super(File, self).__init__(**kwargs)
