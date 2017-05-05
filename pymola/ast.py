#!/usr/bin/env python
"""
Modelica AST definitions
"""
from __future__ import print_function, absolute_import, division, print_function, unicode_literals

import copy
import json
import sys
from collections import OrderedDict

VALIDATE_AST = True

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


class FieldList(list):
    def __init__(self, types, default=[]):
        super(FieldList, self).__init__()
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
                raise IOError('{:s}.{:s} requires types {:s}, but got {:s}'.format(
                name, key, 'list', type(val_list).__name__))
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
            super(FieldList, self).__add__(other)
        else:
            raise IOError('List requires elements of type', self.types)


# TODO get dict validation working
class FieldDict(OrderedDict):
    def __init__(self, types, default):
        super(FieldDict, self).__init__()
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
            if not type(val) in self.types:
                if throw:
                    raise IOError('{:s}.{:s} requires list items of type ({:s}), but got {:s}'.format(
                        name, key, ','.join([t.__name__ for t in self.types]), type(val).__name__))
                else:
                    return False
            else:
                return True

    def __setitem__(self, key, value):
        if VALIDATE_AST:
            if not type(value) in self.types:
                raise IOError('{:s} requires dict values of type ({:s}), but got {:s}'.format(
                    key, ','.join([t.__name__ for t in self.types]), type(value).__name__))
        super(FieldDict, self).__setitem__(key, value)


class Node(object):
    ast_spec = {}

    def __init__(self, **kwargs):
        for key in self.ast_spec.keys():
            # make sure we don't share ast_spec default data by using deep copy
            field_type = type(self.ast_spec[key])
            default = self.ast_spec[key].default
            self.__dict__[key] = copy.deepcopy(default)
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


class Array(Node):
    def __init__(self, **kwargs):
        super(Array, self).__init__(**kwargs)


class Slice(Node):
    def __init__(self, **kwargs):
        super(Slice, self).__init__(**kwargs)


class ComponentRef(Node):
    def __init__(self, **kwargs):
        super(ComponentRef, self).__init__(**kwargs)


class Expression(Node):
    def __init__(self, **kwargs):
        super(Expression, self).__init__(**kwargs)


class IfExpression(Node):
    def __init__(self, **kwargs):
        super(IfExpression, self).__init__(**kwargs)


class Equation(Node):
    def __init__(self, **kwargs):
        super(Equation, self).__init__(**kwargs)


class IfEquation(Node):
    def __init__(self, **kwargs):
        super(IfEquation, self).__init__(**kwargs)


class ForIndex(Node):
    def __init__(self, **kwargs):
        super(ForIndex, self).__init__(**kwargs)


class ForEquation(Node):
    def __init__(self, **kwargs):
        super(ForEquation, self).__init__(**kwargs)


class ConnectClause(Node):
    def __init__(self, **kwargs):
        super(ConnectClause, self).__init__(**kwargs)


class AssignmentStatement(Node):
    def __init__(self, **kwargs):
        super(AssignmentStatement, self).__init__(**kwargs)


class IfStatement(Node):
    def __init__(self, **kwargs):
        super(IfStatement, self).__init__(**kwargs)


class ForStatement(Node):
    def __init__(self, **kwargs):
        super(ForStatement, self).__init__(**kwargs)


class Symbol(Node):
    def __init__(self, **kwargs):
        super(Symbol, self).__init__(**kwargs)

    ATTRIBUTES = ['min', 'max', 'start', 'fixed', 'nominal']


class ComponentClause(Node):
    def __init__(self, **kwargs):
        super(ComponentClause, self).__init__(**kwargs)


class EquationSection(Node):
    def __init__(self, **kwargs):
        super(EquationSection, self).__init__(**kwargs)


class AlgorithmSection(Node):
    def __init__(self, **kwargs):
        super(AlgorithmSection, self).__init__(**kwargs)


class ImportAsClause(Node):
    def __init__(self, **kwargs):
        super(ImportAsClause, self).__init__(**kwargs)


class ImportFromClause(Node):
    def __init__(self, **kwargs):
        super(ImportFromClause, self).__init__(**kwargs)


class ElementModification(Node):
    def __init__(self, **kwargs):
        super(ElementModification, self).__init__(**kwargs)


class ShortClassDefinition(Node):
    def __init__(self, **kwargs):
        super(ShortClassDefinition, self).__init__(**kwargs)


class ElementReplaceable(Node):
    def __init__(self, **kwargs):
        super(ElementReplaceable, self).__init__(**kwargs)


class ClassModification(Node):
    def __init__(self, **kwargs):
        super(ClassModification, self).__init__(**kwargs)


class ExtendsClause(Node):
    def __init__(self, **kwargs):
        super(ExtendsClause, self).__init__(**kwargs)


class Class(Node):
    def __init__(self, **kwargs):
        super(Class, self).__init__(**kwargs)


class File(Node):
    def __init__(self, **kwargs):
        super(File, self).__init__(**kwargs)

    def find_class(self, component_ref):
        for c in self.classes:
            if not component_ref.child:
                return self.classes[component_ref.name]

        return self.classes[component_ref.name]

    def find_symbol(self, c, component_ref):
        sym = c.symbols[component_ref.name]
        if len(component_ref.child) > 0:
            c = self.find_class(sym.type)
            return self.find_symbol(c, component_ref.child[0])
        else:
            return sym


class Collection(Node):
    def __init__(self, **kwargs):
        super(Collection, self).__init__(**kwargs)

    def extend(self, other):
        self.files.extend(other.files)

    def find_class(self, component_ref, within=[]):
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
            raise KeyError
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

            c = next((f.classes[class_name] for f in self.files if f.within and repr(f.within[0]) == repr(extended_within) and class_name in f.classes), None)

            # TODO: This could probably be cleaner if we do nested classes.
            # Then we could traverse up the tree until we found a match,
            # instead of just trying twice (once with, once without prepending
            # the passed-in 'within').
            if c is None:
                # Try again with root node lookup instead of relative
                # NOTE: We are using repr() to compare, because implementing
                # __eq__ on the ComponentRef class will make it unhashable.
                c = next((f.classes[class_name] for f in self.files if f.within and repr(f.within[0]) == repr(c_within) and class_name in f.classes), None)
                if c is None:
                    # TODO: How long do we traverse? Do we somehow force a stop at Real, Boolean, etc?
                    #       Now a force is stopped on anything in the Modelica library.
                    # FIXME: The "SI" part should be removed when we can handle import statements.
                    if c_within.name in ("Modelica", "SI"):
                        raise KeyError
                    else:
                        raise Exception("Could not find class {} in {}".format(class_name, c_within))
                else:
                    return c
            else:
                return c



    def find_symbol(self, c, component_ref):
        sym = c.symbols[component_ref.name]
        if len(component_ref.child) > 0:
            c = self.find_class(sym.type)
            return self.find_symbol(c, component_ref.child[0])
        else:
            return sym


VISIBILITY_PRIVATE = 0
VISIBILITY_PROTECTED = 1
VISIBILITY_PUBLIC = 2


# Here we define the AST specifications for all nodes
# these are static variables shared between class instances
# and are defined here to allow a class to list itself in
# the allowed field types, this self referencing is not
# possible when initially declaring a class in python

Primary.ast_spec = {
    'value' : Field([bool, float, int, str]),
}

Array.ast_spec = {
    'values' : FieldList([Expression, Primary, ComponentRef, Array]),
}

Slice.ast_spec = {
    'start' : Field([Expression, Primary, ComponentRef], Primary(value=0)),
    'stop' : Field([Expression, Primary, ComponentRef], Primary(value=-1)),
    'step' : Field([Expression, Primary, ComponentRef], Primary(value=1)),
}

ComponentRef.ast_spec = {
    'name' : Field([str]),
    'indices' : FieldList([Expression, Slice, Primary, ComponentRef], []),
    'child' : FieldList([ComponentRef], []),
}

Expression.ast_spec = {
    'operator' : Field([str, ComponentRef]),
    'operands' : FieldList([Expression, Primary, ComponentRef, Array, IfExpression]),
}

IfExpression.ast_spec = {
    'conditions' : FieldList([Expression, Primary, ComponentRef, Array, IfExpression]),
    'expressions' : FieldList([Expression, Primary, ComponentRef, Array, IfExpression]),
}

Equation.ast_spec = {
    'left' : Field([Expression, Primary, ComponentRef]),
    'right' : Field([Expression, Primary, ComponentRef]),
    'comment' : Field([str]),
}

IfEquation.ast_spec = {
    'conditions' : FieldList([Expression, Primary, ComponentRef]),
    'equations' : FieldList([Equation, ForEquation, ConnectClause, IfEquation], []),
    'comment' : Field([str]),
}

ForIndex.ast_spec = {
    'name' : Field([str]),
    'expression' : Field([Expression, Primary, Slice]),
}

ForEquation.ast_spec = {
    'indices' : FieldList([ForIndex]),
    'equations' : FieldList([Equation, ForEquation, ConnectClause], []),
    'comment' : Field([str]),
}

ConnectClause.ast_spec = {
    'left' : Field([ComponentRef]),
    'right' : Field([ComponentRef]),
    'comment' : Field([str]),
}

AssignmentStatement.ast_spec = {
    'left' : FieldList([ComponentRef]),
    'right' : Field([Expression, IfExpression, Primary, ComponentRef]),
    'comment' : Field([str]),
}

IfStatement.ast_spec = {
    'expressions' : FieldList([Expression, Primary, ComponentRef]),
    'statements' : FieldList([AssignmentStatement, ForStatement], []),
    'comment' : Field([str]),
}

ForStatement.ast_spec = {
    'indices' : FieldList([ForIndex]),
    'statements' : FieldList([AssignmentStatement, ForStatement], []),
    'comment' : Field([str]),
}

Symbol.ast_spec = {
    'name' : Field([str], ''),
    'type' : Field([ComponentRef], ComponentRef()),
    'prefixes' : FieldList([str], []),
    'redeclare' : Field([bool], False),
    'final' : Field([bool], False),
    'inner' : Field([bool], False),
    'outer' : Field([bool], False),
    'dimensions' : FieldList([Expression, Primary, ComponentRef], [Primary(value=1)]),
    'comment' : Field([str], ''),
    'start' : Field([Expression, Primary, ComponentRef, Array], Primary(value=nan)),
    'min' : Field([Expression, Primary, ComponentRef, Array], Primary(value=nan)),
    'max' : Field([Expression, Primary, ComponentRef, Array], Primary(value=nan)),
    'nominal' : Field([Expression, Primary, ComponentRef, Array], Primary(value=nan)),
    'value' : Field([Expression, Primary, ComponentRef, Array], Primary(value=nan)),
    'fixed' : Field([Primary], False),
    'id' : Field([int], 0),
    'order' : Field([int], 0),
    'visibility' : Field(int, VISIBILITY_PRIVATE),
    'class_modification' : Field(ClassModification),
}

ComponentClause.ast_spec = {
    'prefixes' : FieldList([str], []),
    'type' : Field([ComponentRef], ComponentRef()),
    'dimensions' : FieldList([Expression, Primary, ComponentRef], [Primary(value=1)]),
    'comment' : FieldList([str], []),
    'symbol_list' : FieldList([Symbol], []),
}

EquationSection.ast_spec = {
    'initial' : Field([bool], False),
    'equations' : FieldList([Equation, ForEquation, ConnectClause], []),
}

AlgorithmSection.ast_spec = {
    'initial' : Field([bool], False),
    'statements' : FieldList([AssignmentStatement, ForStatement], []),
}

ImportAsClause.ast_spec = {
    'component' : Field([ComponentRef]),
    'name' : Field([str]),
}

ImportFromClause.ast_spec = {
    'component' : Field([ComponentRef]),
    'symbols' : FieldList([str]),
}

# TODO: Check if ComponentRef modifiers are handled correctly. For example,
# check HomotopicLinear which extends PartialHomotopic with the modifier
# "H(min = H_b)".
ElementModification.ast_spec = {
    'component': Field([ComponentRef], [ComponentRef()]),
    'modifications' : FieldList([Primary, Expression, ClassModification, Array, ComponentRef], []),
}

ShortClassDefinition.ast_spec = {
    'name' : Field(str),
    'type' : Field(str, ''),
    'component' : Field(ComponentRef),
    'class_modification' : Field([ClassModification]),
}

ClassModification.ast_spec = {
    'arguments' : FieldList([ElementModification, ComponentClause, ShortClassDefinition], []),
}

ExtendsClause.ast_spec = {
    'component' : Field([ComponentRef]),
    'class_modification' : Field([ClassModification]),
    'visibility': Field(int, VISIBILITY_PRIVATE),
}

Class.ast_spec = {
    'name' : Field(str),
    'imports' : FieldList([ImportAsClause, ImportFromClause], []),
    'extends' : FieldList([ExtendsClause], []),
    'encapsulated' : Field([bool], False),
    'partial' : Field([bool], False),
    'final' : Field([bool], False),
    'type' : Field([str], ''),
    'comment' : Field(str, ''),
    'symbols' : FieldDict([Symbol], {}),
    'initial_equations' : FieldList([Equation, ForEquation], []),
    'equations' : FieldList([Equation, ForEquation, ConnectClause], []),
    'initial_statements' : FieldList([AssignmentStatement, ForStatement], []),
    'statements' : FieldList([AssignmentStatement, ForStatement], []),
    'within' : FieldList([ComponentRef], []),
}

File.ast_spec = {
    'within' : FieldList([ComponentRef], []),
    'classes' : FieldDict([Class], {}),
}

# TODO: The Modelica specification (v3.3, Ch. 4) seems to suggest that
# everything can/should be stored in one single (root) class, that can contain
# other classes (each in turn also storing classes). That would make both File
# and Collection types obsolete, but could make processing certain files (e.g.
# Package.mo) before others important. For example, a 'within' statement would
# only make sense if that package actually already exists in our tree of
# classes, otherwise we would have nowhere to add the classes in that file.
Collection.ast_spec = {
    'files': FieldList([File], []),
}
