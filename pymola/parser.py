#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function
import sys
import antlr4
import antlr4.Parser
import yaml

from .generated.ModelicaLexer import ModelicaLexer
from .generated.ModelicaParser import ModelicaParser
from .generated.ModelicaListener import ModelicaListener

from . import ast


class ASTListener(ModelicaListener):

    def __init__(self):
        self.ast = {}
        self.ast_result = None
        self.file_node = None
        self.class_node = None
        self.comp_clause_node = None
        self.eq_sect = None
        self.symbol_node = None
        self.eq_comment = None

    def enterStored_definition(self, ctx):
        file_node = ast.File()
        file_node.within = ctx.WITHIN() != None
        self.ast[ctx] = file_node
        self.file_node = file_node

    def exitStored_definition(self, ctx):
        for class_node in [self.ast[e] for e in ctx.stored_definition_class()]:
            self.ast[ctx].classes[class_node.name] = class_node
        self.ast_result = self.ast[ctx]

    def enterStored_definition_class(self, ctx):
        class_node = ast.Class()
        class_node.final = ctx.FINAL() != None
        self.class_node = class_node
        self.ast[ctx] = class_node

    def exitStored_definition_class(self, ctx):
        pass

    def enterClass_definition(self, ctx):
        class_node = self.class_node
        class_node.encapsulated = ctx.ENCAPSULATED() != None
        class_node.partial = ctx.class_prefixes().PARTIAL() != None
        class_node.type = str(ctx.class_prefixes().class_type().getText())

    def enterClass_spec_comp(self, ctx):
        class_node = self.class_node
        class_node.name = str(ctx.IDENT()[0].getText())
        class_node.comment = str(ctx.string_comment().getText()[1:-1])

    def exitComposition(self, ctx):
        elist_comb = []
        for elist in [ctx.epriv, ctx.epub, ctx.epro]:
            if elist is not None:
                for e in [self.ast[e] for e in elist.element()]:
                    elist_comb += e.symbol_list
        for symbol in elist_comb:
            self.class_node.symbols[symbol.name] = symbol

        eqlist_comb = []
        for eqlist in [self.ast[e] for e in ctx.equation_section()]:
            if eqlist is not None:
                eqlist_comb += eqlist.equation_list
        self.class_node.equations += eqlist_comb

    def enterEquation_section(self, ctx):
        eq_sect = ast.EquationSection(
            initial=ctx.INITIAL() != None
        )
        self.ast[ctx] = eq_sect
        self.eq_sect = eq_sect

    def exitEquation_section(self, ctx):
        eq_sect = self.ast[ctx]
        eq_sect.equation_list += [self.ast[e] for e in ctx.equation()]

    def enterEquation(self, ctx):
        self.eq_comment = str(ctx.comment().getText())

    def exitEquation(self, ctx):
        self.ast[ctx] = self.ast[ctx.equation_options()]

    def exitEquation_simple(self, ctx):
        self.ast[ctx] = ast.Equation(
            left=self.ast[ctx.simple_expression()],
            right=self.ast[ctx.expression()],
            comment=self.eq_comment)

    def exitEquation_connect_clause(self, ctx):
        self.ast[ctx] = self.ast[ctx.connect_clause()]

    def exitConnect_clause(self, ctx):
        self.ast[ctx] = ast.ConnectClause(
            left=str(ctx.component_reference()[0].getText()),
            right=str(ctx.component_reference()[1].getText()),
            comment=self.eq_comment)

    def exitSimple_expression(self, ctx):
        #TODO only using first expression
        self.ast[ctx] = self.ast[ctx.expr()[0]]

    def exitExpression_simple(self, ctx):
        self.ast[ctx] = self.ast[ctx.simple_expression()]

    def exitExpr_primary(self, ctx):
        self.ast[ctx] = self.ast[ctx.primary()]

    def exitPrimary_unsigned_number(self, ctx):
        self.ast[ctx] = yaml.load(ctx.getText())

    def exitPrimary_string(self, ctx):
        self.ast[ctx] = ast.Primary(value=ctx.getText())

    def exitPrimary_false(self, ctx):
        self.ast[ctx] = ast.Primary(value=False)

    def exitPrimary_true(self, ctx):
        self.ast[ctx] = ast.Primary(value=True)

    def exitPrimary_function(self, ctx):
        self.ast[ctx] = ast.Primary(value=ctx.getText())

    def exitPrimary_derivative(self, ctx):
        self.ast[ctx] = ast.Primary(value=ctx.getText())

        self.ast[ctx] = ast.Expression(
            operator='der',
            operands=[ast.ComponentRef(
                name=str(ctx.function_call_args().function_arguments().function_argument()[0].getText()))]
        )
        self.class_node.states += self.ast[ctx].operands

    def exitPrimary_component_reference(self, ctx):
        self.ast[ctx] = ast.ComponentRef(
            name=str(ctx.getText())
        )

    def exitExpr_add(self, ctx):
        self.ast[ctx] = ast.Expression(
            operator=str(ctx.op.text),
            operands=[self.ast[e] for e in ctx.expr()]
        )

    def exitExpr_mul(self, ctx):
        self.ast[ctx] = ast.Expression(
            operator=str(ctx.op.text),
            operands=[self.ast[e] for e in ctx.expr()]
        )

    def exitExpr_neg(self, ctx):
        self.ast[ctx] = ast.Expression(
            operator=str(ctx.op.text),
            operands=[self.ast[e] for e in ctx.expr()]
        )

    def exitElement_list(self, ctx):
        self.ast[ctx] = [self.ast[e] for e in ctx.element()]

    def exitElement(self, ctx):
        self.ast[ctx] = self.ast[ctx.regular_element()]

    def exitRegular_element(self, ctx):
        self.ast[ctx] = self.ast[ctx.comp_elem]

    def enterComponent_clause(self, ctx):
        dimensions = None
        if ctx.array_subscripts() is not None:
            dimensions=[int(s) for s in ctx.array_subscripts().subscript().getText()]
        self.ast[ctx] = ast.ComponentClause(
            prefixes=str(ctx.type_prefix().getText()).split(' '),
            type=ctx.type_specifier().getText(),
            dimensions=dimensions
        )
        self.comp_clause = self.ast[ctx]

    def exitComponent_clause(self, ctx):
        self.ast[ctx].symbol_list = self.ast[ctx.component_list()]

    def exitComponent_list(self, ctx):
        self.ast[ctx] = [self.ast[e] for e in ctx.component_declaration()]

    def enterComponent_declaration(self, ctx):
        sym = ast.Symbol(
            name='',
            type=self.comp_clause.type,
            prefixes=self.comp_clause.prefixes,
            comment=ctx.comment().getText()
        )
        self.symbol_node = sym
        self.ast[ctx] = sym

    def enterDeclaration(self, ctx):
        sym = self.symbol_node
        dimensions = None
        if ctx.array_subscripts() is not None:
            dimensions = [int(s) for s in ctx.array_subscripts().subscript().getText()]
        elif self.comp_clause.dimensions is not None:
            dimensions = self.comp_clause.dimensions
        sym.name = ctx.IDENT().getText()
        sym.dimensions = dimensions
        if 'input' in sym.prefixes:
            self.class_node.inputs += [ast.ComponentRef(name=str(sym.name))]
        elif 'output' in sym.prefixes:
            self.class_node.outputs += [ast.ComponentRef(name=str(sym.name))]
        elif 'constant' in sym.prefixes:
            self.class_node.constants += [ast.ComponentRef(name=str(sym.name))]


def parse(file):
    input_stream = antlr4.FileStream(file)
    lexer = ModelicaLexer(input_stream)
    stream = antlr4.CommonTokenStream(lexer)
    parser = ModelicaParser(stream)
    parse_tree = parser.stored_definition()
    astListener = ASTListener()
    parse_walker = antlr4.ParseTreeWalker()
    parse_walker.walk(astListener, parse_tree)
    ast_tree = astListener.ast_result
    return ast_tree