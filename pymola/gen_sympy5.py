#!/usr/bin/env python
"""
Modelica compiler.
"""
from __future__ import print_function
import sys
import antlr4
import antlr4.Parser
import json
import pyast as ast
import pyast.dump.js

# compiler
from generated.ModelicaLexer import ModelicaLexer
from generated.ModelicaParser import ModelicaParser
from generated.ModelicaListener import ModelicaListener

def handle_walk(var, listener):
    if isinstance(var, Node):
        var.walk(listener)
    elif isinstance(var, ast.TypedDict):
        for k in var.keys():
            handle_walk(var[k], listener)
    elif isinstance(var, ast.TypedList):
        for c in var:
            handle_walk(c, listener)
    else:
        pass

class Node(ast.Node):

    def __repr__(self):
        return pyast.dump.js.dump(self)

    def get_children(self):
        return [self.__dict__[field] for field in self._fields]

    def walk(self, listener):
        name = self.__class__.__name__
        if hasattr(listener, 'enter_' + name):
            listener.__getattribute__('enter_' + name)(self)
        for child in self.get_children():
            handle_walk(child, listener)
        if hasattr(listener, 'exit_' + name):
            listener.__getattribute__('exit_' + name)(self)

    def visit(self, visitor):
        name = self.__class__.__name__
        if hasattr(visitor, 'visit_' + name):
            return visitor.__getattribute__('visit_' + name)(self)
        else:
            res = []
            for child in self.get_children():
                if isinstance(child, Node):
                    res += [child.visit(visitor)]
            return res

class Operator(Node):
    token = ast.field(('+', '=', '-', '==', '!=', '>', '<'))

class Expression(Node):
    operator = ast.field(Operator)
    operands = ast.seq(Node)

class Equation(Node):
    left = ast.field(str, null=True) # TODO switch to expr
    right = ast.field(str, null=True)

class Symbol(Node):
    type =  ast.field((str, unicode))
    prefixes = ast.seq((str, unicode), null=True)

class Class(Node):
    symbols = ast.dict(Symbol, null=True)
    equations = ast.seq(Equation, null=True)

class File(Node):
    classes = ast.dict(Class, null=True)


class DAEListener(ModelicaListener):

    def __init__(self):
        self.file = None
        self.class_node = None
        self.comp_clause = None
        self.comp_decl_node = None

    def enterStored_definition(self, ctx):
        """
        Create a new stored definition context.
        """
        self.file = File()

    def enterClass_spec_comp(self, ctx):
        """
        Create a new class context.
        """
        class_name = ctx.IDENT()[0].getText()
        self.class_node = Class()
        self.file.classes[class_name] = self.class_node

    def enterComponent_clause(self, ctx):
        """
        Creaet a component clause context.
        """
        self.comp_clause = {
            'prefix': str(ctx.type_prefix().getText()),
            'type': str(ctx.type_specifier().getText())
        }

    def enterDeclaration(self, ctx):
        """
        Record symbols.
        """
        name = str(ctx.IDENT().getText())
        type = self.comp_clause['type']
        prefixes = self.comp_clause['prefix'].split()
        assert name not in self.class_node.symbols.keys()
        sym = Symbol(type=type, prefixes=prefixes)
        self.class_node.symbols[name] = sym
        self.comp_decl_node = sym

    def exitEquation_simple(self, ctx):
        self.class_node.equations += [Equation(
            str(ctx.simple_expression().getText()),
            str(ctx.expression().getText()))]

input_stream = antlr4.FileStream('./test/Aircraft.mo')
lexer = ModelicaLexer(input_stream)
stream = antlr4.CommonTokenStream(lexer)
parser = ModelicaParser(stream)
tree = parser.stored_definition()
# print(tree.toStringTree(recog=parser))
daeListener = DAEListener()
walker = antlr4.ParseTreeWalker()
walker.walk(daeListener, tree)

def flatten(file_node, main_class_name):
    res = Class()
    root = file_node.classes[main_class_name]
    res.equations = root.equations
    res.symbols = root.symbols
    for sym_key in root.symbols.keys():
        for class_type in file_node.classes.keys():
            if root.symbols[sym_key].type == class_type:
                class_def = file_node.classes[class_type]
                for cls_sym_key in class_def.symbols.keys():
                    new_name = '{:s}.{:s}'.format(sym_key, cls_sym_key)
                    res.symbols[new_name] = class_def.symbols[cls_sym_key]
                    for key in ['input', 'output']:
                        if key in res.symbols[new_name].prefixes:
                            res.symbols[new_name].prefixes.remove(key)

                #TODO need to make replacement more intelligent by using context
                for eq in class_def.equations:
                    for cls_sym_key in class_def.symbols.keys():
                        new_name = '{:s}.{:s}'.format(sym_key, cls_sym_key)
                        #eq = eq.replace(cls_sym_key, new_name)
                    res.equations += [eq]
                res.symbols.pop(sym_key)
    return res

tree = daeListener.file
flat_tree = flatten(daeListener.file, 'Aircraft')
print(flat_tree)

class Listener(object):

    def enter_File(self, tree):
        print('enter file')

    def exit_File(self, tree):
        print('exit file')

    def enter_Class(self, tree):
        print('enter class')

    def exit_Class(self, tree):
        print('exit class')

    def enter_Experssion(self, tree):
        print('enter expr')

    def exit_Expression(self, tree):
        print('exit expr')


listener = Listener()
tree.walk(listener)
