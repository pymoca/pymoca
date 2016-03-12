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

class Ast(object):

    @staticmethod
    def file_node():
        return {
            '_ast' : 'file',
            'classes': {},
        }

    @staticmethod
    def class_node():
        return {
            '_ast' : 'class',
            'symbols': {},
            'equations': [],
        }

    @staticmethod
    def symbol_node():
        return {
            '_ast' : 'symbol',
            'type': None,
            'prefix' : None,
        }

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
        self.file = Ast.file_node()

    def enterClass_spec_comp(self, ctx):
        """
        Create a new class context.
        """
        class_name = ctx.IDENT()[0].getText()
        self.class_node = Ast.class_node()
        self.file['classes'][class_name] = self.class_node

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
        prefix = self.comp_clause['prefix'].split()
        assert name not in self.class_node['symbols'].keys()
        comp_decl_node = Ast.symbol_node()
        comp_decl_node['type'] = type
        comp_decl_node['prefix'] = prefix
        self.class_node['symbols'][name] = comp_decl_node
        self.comp_decl_node = comp_decl_node

    def exitEquation_simple(self, ctx):
        self.class_node['equations'] += [str(ctx.getText())]

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
    res = Ast.class_node()
    root = file_node['classes'][main_class_name]
    res['equations'] += root['equations']
    res['symbols'].update(root['symbols'])

    for sym_key in root['symbols'].keys():
        for class_type in file_node['classes'].keys():
            if root['symbols'][sym_key]['type'] == class_type:
                class_def = file_node['classes'][class_type]
                for cls_sym_key in class_def['symbols'].keys():
                    new_name = '{:s}.{:s}'.format(sym_key, cls_sym_key)
                    res['symbols'][new_name] = class_def['symbols'][cls_sym_key]
                    for key in ['input', 'output']:
                        if key in res['symbols'][new_name]['prefix']:
                            res['symbols'][new_name]['prefix'].remove(key)

                #TODO need to make replacement more intelligent by using context
                for eq in class_def['equations']:
                    for cls_sym_key in class_def['symbols'].keys():
                        new_name = '{:s}.{:s}'.format(sym_key, cls_sym_key)
                        #eq = eq.replace(cls_sym_key, new_name)
                    res['equations'] += [eq]
                print('res_keys', res['symbols'].keys())
                res['symbols'].pop(sym_key)
    return res

pprint.pprint(daeListener.file)
flat_tree = flatten(daeListener.file, 'Aircraft')
pprint.pprint(flat_tree)
