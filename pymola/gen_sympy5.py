#!/usr/bin/env python
"""
Modelica compiler.
"""
from __future__ import print_function
import sys
import antlr4
import antlr4.Parser
import pprint

# compiler
from generated.ModelicaLexer import ModelicaLexer
from generated.ModelicaParser import ModelicaParser
from generated.ModelicaListener import ModelicaListener

class DAEListener(ModelicaListener):

    def __init__(self):
        self.file = None
        self.class_node = None
        self.comp_clause = None
        self.model = None

    def enterStored_definition(self, ctx):
        """
        Create a new stored definition context.
        """
        self.file = {
            '_ast' : 'file',
            'classes': {},
        }

    def enterClass_spec_comp(self, ctx):
        """
        Create a new class context.
        """
        class_name = ctx.IDENT()[0].getText()
        self.class_node = {
            '_ast' : 'class',
            'symbols': {},
            'equations': [],
            'states': [],
        }
        self.file['classes'][class_name] = self.class_node

    def enterComponent_clause(self, ctx):
        """
        Creaet a component clause context.
        """
        self.comp_clause = {
            'prefix': str(ctx.type_prefix().getText()),
            'type': str(ctx.type_specifier().getText())
        }

    def exitDeclaration(self, ctx):
        """
        Record symbols.
        """
        name = str(ctx.IDENT().getText())
        assert name not in self.class_node['symbols'].keys()
        self.class_node['symbols'][name] = {
            '_ast' : 'symbol',
            'type': self.comp_clause['type'],
            'prefix': self.comp_clause['prefix'],
        }

    def exitEquation_simple(self, ctx):
        self.class_node['equations'] += [str(ctx.getText())]

    def exitPrimary_derivative(self, ctx):
        var = str(ctx.function_call_args().function_arguments().getText())
        assert var in self.class_node['symbols']
        self.class_node['states'] += [var]

input_stream = antlr4.FileStream('./test/BouncingBall.mo')
lexer = ModelicaLexer(input_stream)
stream = antlr4.CommonTokenStream(lexer)
parser = ModelicaParser(stream)
tree = parser.stored_definition()
# print(tree.toStringTree(recog=parser))
daeListener = DAEListener()
walker = antlr4.ParseTreeWalker()
walker.walk(daeListener, tree)

pprint.pprint(daeListener.file)
