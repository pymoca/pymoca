#!/usr/bin/env python

from __future__ import print_function
import antlr4
import antlr4.Parser
import json

# compiler
from generated.ModelicaLexer import ModelicaLexer
from generated.ModelicaParser import ModelicaParser
from generated.ModelicaListener import ModelicaListener
import copy


def newFile():
    return {
        '_node_type' : 'file',
        'classes': [],
    }

def newClass():
    return {
        '_node_type' : 'class',
        'equations': [],
        'parameters': [],
        'constants': [],
        'variables': [],
    }

def newEq():
    return {
        '_node_type' : 'equation',
        'expr': None,
    }

def newComponentClause():
    return {
        '_node_type' : 'component_clause',
        'prefix': '',
        'type': '',
        'components': [],
    }

def newComponent():
    return {
        '_node_type' : 'component',
        'name': '',
        'prefix': '',
        'type': '',
        'value': '',
    }

def newAdd():
    return {
        '_node_type' : 'add',
        'left': None,
        'right': None,
    }

def newNeg():
    return {
        '_node_type' : 'neg',
        'expr': None,
    }

def newMul():
    return {
        '_node_type' : 'mul',
        'left': None,
        'right': None,
    }

def newAnd():
    return {
        '_node_type' : 'and',
        'left': None,
        'right': None,
    }

def newExp():
    return {
        '_node_type' : 'exp',
        'left': None,
        'right': None,
    }

def newNot():
    return {
        '_node_type' : 'not',
        'expr': None,
    }

def newExpr():
    return {
        '_node_type' : 'expr',
        'expr': None,
    }

def newPrimary():
    return {
        '_node_type' : 'primary',
        'value': None,
    }

class DefListener(ModelicaListener):

    def __init__(self):
        self.ast = None
        self.scope_stack = []
        self.scope_dict = {}
        self.c_file = None
        self.c_class = None

    def push_scope(self, ctx, node):
        self.scope_stack.append(node)
        self.scope_dict[ctx] = node
        return node

    def pop_scope(self, ctx):
        node = self.scope_stack.pop()
        self.scope_dict[ctx] = node
        return node

    def scope(self):
        return self.scope_stack[-1]

    def enterEveryRule(self, ctx):
        if ctx.parentCtx in self.scope_dict.keys():
            self.scope_dict[ctx] = self.scope_dict[ctx.parentCtx]

    def enterStored_definition(self, ctx):
        file_node = self.push_scope(ctx, newFile())
        self.c_file = file_node

    def exitStored_definition(self, ctx):
        self.ast = self.pop_scope(ctx)

    def enterStored_definition_class(self, ctx):
        class_node = self.push_scope(ctx, newClass())
        self.c_class = class_node

    def exitStored_definition_class(self, ctx):
        class_node = self.pop_scope(ctx)
        self.scope()['classes'] += [class_node]

    def enterComponent_clause(self, ctx):
        # handle everything in exit when sub contexts fully walked
        c = self.push_scope(ctx, newComponentClause())
        c['prefix'] = ctx.type_prefix().getText()
        c['type'] = ctx.type_specifier().name().getText()

    def exitComponent_clause(self, ctx):
        c = self.pop_scope(ctx)
        if c['prefix'] == 'parameter':
            self.scope()['parameters'] += c['components']
        elif c['prefix'] == 'constant':
            self.scope()['constants'] += c['components']
        elif c['prefix'] == '':
            self.scope()['variables'] += c['components']

    def enterComponent_declaration(self, ctx):
        prefix = self.scope()['prefix']
        type = self.scope()['type']
        c = self.push_scope(ctx, newComponent())
        c['name'] = ctx.declaration().IDENT().getText()
        c['prefix'] = prefix
        c['type'] = type

    def exitComponent_declaration(self, ctx):
        c = self.pop_scope(ctx)
        self.scope()['components'] += [c]

    def enterModification_assignment(self, ctx):
        self.scope()['value'] = ctx.expression().getText()

    def enterEquation_simple(self, ctx):
        eq_node = self.push_scope(ctx, newEq())

    def exitEquation_simple(self, ctx):
        eq_node = self.pop_scope(ctx)
        self.scope()['equations'] += [eq_node]

    def enterSimple_expression(self, ctx):
        c = self.push_scope(ctx, newExpr())

    def exitSimple_expression(self, ctx):
        c = self.pop_scope(ctx)
        self.scope()['expr'] = c

    def enterExpr_add(self, ctx):
        c = self.push_scope(ctx, newAdd())
        c['left'] = ctx.expr()[0].getText()
        c['right'] = ctx.expr()[1].getText()

    def exitExpr_add(self, ctx):
        c = self.pop_scope(ctx)
        self.scope()['expr'] = c

    def enterExpr_neg(self, ctx):
        c = self.push_scope(ctx, newNeg())
        c['expr'] = ctx.expr().getText()

    def exitExpr_neg(self, ctx):
        c = self.pop_scope(ctx)
        self.scope()['expr'] = c

    def enterExpr_mul(self, ctx):
        c = self.push_scope(ctx, newMul())
        c['left'] = ctx.expr()[0].getText()
        c['right'] = ctx.expr()[1].getText()

    def exitExpr_mul(self, ctx):
        c = self.pop_scope(ctx)
        self.scope()['expr'] = c

    def enterExpr_and(self, ctx):
        c = self.push_scope(ctx, newAdd())
        c['left'] = ctx.expr()[0].getText()
        c['right'] = ctx.expr()[1].getText()

    def exitExpr_and(self, ctx):
        c = self.pop_scope(ctx)
        self.scope()['expr'] = c

    def enterExpr_or(self, ctx):
        c = self.push_scope(ctx, newAdd())
        c['left'] = ctx.expr()[0].getText()
        c['right'] = ctx.expr()[1].getText()

    def exitExpr_or(self, ctx):
        c = self.pop_scope(ctx)
        self.scope()['expr'] = c

    def enterExpr_exp(self, ctx):
        c = self.push_scope(ctx, newExp())
        c['left'] = ctx.primary()[0].getText()
        c['right'] = ctx.primary()[1].getText()

    def exitExpr_exp(self, ctx):
        c = self.pop_scope(ctx)
        self.scope()['expr'] = c

    def enterExpr_not(self, ctx):
        c['expr'] = ctx.primary()[0].getText()

    def exitExpr_not(self, cts):
        c = self.pop_scope(ctx)
        self.scope()['expr'] = c

    def enterExpr_primary(self, ctx):
        c = self.push_scope(ctx, newPrimary())

    def exitExpr_primary(self, ctx):
        c = self.pop_scope(ctx)
        #self.scope()['primary'] = c

input_stream = antlr4.FileStream('./test/BouncingBall.mo')
lexer = ModelicaLexer(input_stream)
stream = antlr4.CommonTokenStream(lexer)
parser = ModelicaParser(stream)
tree = parser.stored_definition()
# print(tree.toStringTree(recog=parser))
defListener = DefListener()
walker = antlr4.ParseTreeWalker()
walker.walk(defListener, tree)
ast = defListener.ast
print(json.dumps(ast, indent=2, sort_keys=True))