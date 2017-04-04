#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import antlr4
import antlr4.Parser

from . import ast
from .generated.ModelicaLexer import ModelicaLexer
from .generated.ModelicaListener import ModelicaListener
from .generated.ModelicaParser import ModelicaParser

# TODO
#  - Functions
#  - ComponentRef 'within'
#  - Make sure slice indices (eventually) evaluate to integers

#  - Import
#  - min, max, start


class ASTListener(ModelicaListener):

    def __init__(self):
        self.ast = {}
        self.ast_result = None
        self.file_node = None
        self.class_node = None
        self.comp_clause = None
        self.eq_sect = None
        self.symbol_node = None
        self.eq_comment = None
        self.sym_count = 0

    # FILE ===========================================================

    def enterStored_definition(self, ctx):
        file_node = ast.File()
        self.ast[ctx] = file_node
        self.file_node = file_node

    def exitStored_definition(self, ctx):
        within = []
        if ctx.component_reference() is not None:
            within = [self.ast[ctx.component_reference()]]
        self.file_node.within = within

        for class_node in [self.ast[e] for e in ctx.stored_definition_class()]:
            self.ast[ctx].classes[class_node.name] = class_node
        self.ast_result = self.ast[ctx]

    # CLASS ===========================================================

    def enterStored_definition_class(self, ctx):
        class_node = ast.Class()
        class_node.final = ctx.FINAL() is not None
        self.class_node = class_node
        self.ast[ctx] = class_node

    def exitStored_definition_class(self, ctx):
        pass

    def enterClass_definition(self, ctx):
        class_node = self.class_node
        class_node.encapsulated = ctx.ENCAPSULATED() is not None
        class_node.partial = ctx.class_prefixes().PARTIAL() is not None
        class_node.type = ctx.class_prefixes().class_type().getText()

    def enterClass_spec_comp(self, ctx):
        class_node = self.class_node
        class_node.name = ctx.IDENT()[0].getText()

    def exitClass_spec_comp(self, ctx):
        class_node = self.class_node
        class_node.comment = self.ast[ctx.string_comment()]

    def exitComposition(self, ctx):
        for eqlist in [self.ast[e] for e in ctx.equation_section()]:
            if eqlist is not None:
                if eqlist.initial:
                    self.class_node.initial_equations += eqlist.equations
                else:
                    self.class_node.equations += eqlist.equations

    def exitArgument(self, ctx):
        self.ast[ctx] = self.ast[ctx.getChild(ctx.getAltNumber())]

    def exitArgument_list(self, ctx):
        self.ast[ctx] = [self.ast[a] for a in ctx.argument()]

    def exitClass_modification(self, ctx):
        arguments = []
        if ctx.argument_list() is not None:
            arguments = self.ast[ctx.argument_list()]
        self.ast[ctx] = ast.ClassModification(arguments=arguments)

    def enterEquation_section(self, ctx):
        eq_sect = ast.EquationSection(
            initial=ctx.INITIAL() is not None
        )
        self.ast[ctx] = eq_sect
        self.eq_sect = eq_sect

    def exitEquation_section(self, ctx):
        eq_sect = self.ast[ctx]
        if eq_sect.initial:
            eq_sect.equations += [self.ast[e] for e in ctx.equation()]
        else:
            eq_sect.equations += [self.ast[e] for e in ctx.equation()]

    # EQUATION ===========================================================

    def enterEquation(self, ctx):
        pass

    def exitEquation(self, ctx):
        self.ast[ctx] = self.ast[ctx.equation_options()]
        try:
            self.ast[ctx].comment = self.ast[ctx.comment()]
        except AttributeError:
            pass

    def exitEquation_simple(self, ctx):
        self.ast[ctx] = ast.Equation(
            left=self.ast[ctx.simple_expression()],
            right=self.ast[ctx.expression()])

    def exitEquation_if(self, ctx):
        self.ast[ctx] = ast.IfEquation(
            expressions=[self.ast[s] for s in ctx.if_equation().expression()],
            equations=[self.ast[s] for s in ctx.if_equation().equation()])

    def exitEquation_for(self, ctx):
        self.ast[ctx] = ast.ForEquation(
            indices=[self.ast[s] for s in ctx.for_equation().for_indices().for_index()],
            equations=[self.ast[s] for s in ctx.for_equation().equation()])

    def exitEquation_connect_clause(self, ctx):
        self.ast[ctx] = self.ast[ctx.connect_clause()]

    def exitConnect_clause(self, ctx):
        self.ast[ctx] = ast.ConnectClause(
            left=self.ast[ctx.component_reference()[0]],
            right=self.ast[ctx.component_reference()[1]])

    # EXPRESSIONS ===========================================================

    def exitSimple_expression(self, ctx):
        if len(ctx.expr()) > 1:
            if len(ctx.expr()) > 2:
                step = self.ast[ctx.expr()[2]]
            else:
                step = ast.Primary(value=1)
            self.ast[ctx] = ast.Slice(start=self.ast[ctx.expr()[0]], stop=self.ast[ctx.expr()[1]], step=step)
        else:
            self.ast[ctx] = self.ast[ctx.expr()[0]]

    def exitExpression_simple(self, ctx):
        self.ast[ctx] = self.ast[ctx.simple_expression()]

    def exitExpression_if(self, ctx):
        all_expr = [self.ast[s] for s in ctx.expression()]
        # Note that an else block is guaranteed to exist.
        conditions = all_expr[:-1:2]
        expressions = all_expr[1::2] + all_expr[-1:]

        self.ast[ctx] = ast.IfExpression(
            conditions=conditions,
            expressions=expressions)

    def exitExpr_primary(self, ctx):
        self.ast[ctx] = self.ast[ctx.primary()]

    def exitExpr_add(self, ctx):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,
            operands=[self.ast[e] for e in ctx.expr()]
        )

    def exitExpr_exp(self, ctx):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,
            operands=[self.ast[e] for e in ctx.primary()]
        )

    def exitExpr_mul(self, ctx):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,
            operands=[self.ast[e] for e in ctx.expr()]
        )

    def exitExpr_rel(self, ctx):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,
            operands=[self.ast[e] for e in ctx.expr()]
        )

    def exitExpr_neg(self, ctx):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,
            operands=[self.ast[ctx.expr()]]
        )

    def exitSubscript(self, ctx):
        if ctx.expression() is not None:
            self.ast[ctx] = self.ast[ctx.expression()]
        else:
            self.ast[ctx] = ast.Slice()

    def exitArray_subscripts(self, ctx):
        self.ast[ctx] = [self.ast[s] for s in ctx.subscript()]

    def exitFor_index(self, ctx):
        self.ast[ctx] = ast.ForIndex(name=ctx.IDENT().getText(), expression=self.ast[ctx.expression()])

    def exitFor_indices(self, ctx):
        self.ast[ctx] = [self.ast[s] for s in ctx.for_index()]

    # PRIMARY ===========================================================

    def exitPrimary_unsigned_number(self, ctx):
        self.ast[ctx] = ast.Primary(value=float(ctx.getText()))

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
        comp_name = ctx.function_call_args().function_arguments().function_argument()[0].getText()
        self.ast[ctx] = ast.Expression(
            operator='der',
            operands=[ast.ComponentRef(name=comp_name)]
        )
        # TODO 'state' is not a standard prefix;  disable this for now as it does not work
        # when differentiating states defined in superclasses.
        #if 'state' not in self.class_node.symbols[comp_name].prefixes:
        #    self.class_node.symbols[comp_name].prefixes += ['state']

    def exitComponent_reference(self, ctx):
        name = ctx.IDENT().getText()
        indices = []
        if ctx.array_subscripts() is not None:
            indices = [self.ast[x] for x in ctx.array_subscripts().subscript()]
        child = [self.ast[ctx.child]] if ctx.child is not None else []

        self.ast[ctx] = ast.ComponentRef(
            name=name,
            indices=indices,
            child=child
        )

    def exitPrimary_component_reference(self, ctx):
        self.ast[ctx] = self.ast[ctx.component_reference()]

    def exitPrimary_output_expression_list(self, ctx):
        self.ast[ctx] = [self.ast[x] for x in ctx.output_expression_list().expression()]
        # Collapse lists containing a single expression
        if len(self.ast[ctx]) == 1:
            self.ast[ctx] = self.ast[ctx][0]

    def exitPrimary_function_arguments(self, ctx):
        # TODO: This does not support for generators, or function() calls yet.
        #       Only expressions are supported, e.g. {1.0, 2.0, 3.0}.
        v = [self.ast[x.expression()] for x in ctx.function_arguments().function_argument()]
        self.ast[ctx] = ast.Array(values=v)

    def exitEquation_function(self, ctx):
        # TODO, add function ast
        self.ast[ctx] = ctx.getText()

    def exitEquation_when(self, ctx):
        # TODO, add when ast
        self.ast[ctx] = ctx.getText()

    # COMPONENTS ===========================================================

    def exitElement_list(self, ctx):
        self.ast[ctx] = [self.ast[e] for e in ctx.element()]

    def exitElement(self, ctx):
        self.ast[ctx] = self.ast[ctx.getChild(ctx.getAltNumber())]

    def exitImport_list(self, ctx):
        self.ast[ctx] = [ctx.IDENT()] + self.ast[ctx.import_list()]

    def exitImport_clause(self, ctx):
        if ctx.IDENT() is not None:
            self.ast[ctx] = ast.ImportAsClause(component=ast.ComponentRef(name=ctx.name().getText()), name=ctx.IDENT().getText())
        else:
            symbols = self.ast[ctx.import_list()]
            self.ast[ctx] = ast.ImportFromClause(component=ctx.name().getText(), symbols=symbols)
        self.class_node.imports += [self.ast[ctx]]

    def exitExtends_clause(self, ctx):
        if ctx.class_modification() is not None:
            class_modification = self.ast[ctx.class_modification()]
        else:
            class_modification = ast.ClassModification()
        self.ast[ctx] = ast.ExtendsClause(component=ast.ComponentRef(name=ctx.name().getText()),
            class_modification=class_modification)
        self.class_node.extends += [self.ast[ctx]]

    def exitRegular_element(self, ctx):
        self.ast[ctx] = self.ast[ctx.comp_elem]

    def exitReplaceable_element(self, ctx):
        self.ast[ctx] = self.ast[ctx.comp_elem]

    def enterComponent_clause(self, ctx):
        prefixes = ctx.type_prefix().getText().split(' ')
        if prefixes[0] == '':
            prefixes = []
        self.ast[ctx] = ast.ComponentClause(
            prefixes=prefixes,
            type=ctx.type_specifier().getText()
        )
        self.comp_clause = self.ast[ctx]

    def enterComponent_clause1(self, ctx):
        prefixes = ctx.type_prefix().getText().split(' ')
        if prefixes[0] == '':
            prefixes = []
        self.ast[ctx] = ast.ComponentClause(
            prefixes=prefixes,
            type=ctx.type_specifier().getText()
        )
        self.comp_clause = self.ast[ctx]

    def exitComponent_clause(self, ctx):
        clause = self.ast[ctx]
        if ctx.array_subscripts() is not None:
            clause.dimensions = self.ast[ctx.array_subscripts()]

    def exitComponent_clause1(self, ctx):
        clause = self.ast[ctx]

    def enterComponent_declaration(self, ctx):
        sym = ast.Symbol(order = self.sym_count, start=ast.Primary(value=0.0))
        self.sym_count += 1
        self.ast[ctx] = sym
        self.symbol_node = sym
        self.comp_clause.symbol_list += [sym]

    def enterComponent_declaration1(self, ctx):
        sym = ast.Symbol(order = self.sym_count, start=ast.Primary(value=0.0))
        self.sym_count += 1
        self.ast[ctx] = sym
        self.symbol_node = sym
        self.comp_clause.symbol_list += [sym]

    def enterElement_modification(self, ctx):
        sym = ast.Symbol(order = self.sym_count, start=ast.Primary(value=0.0))
        self.sym_count += 1
        self.ast[ctx] = sym
        self.symbol_node = sym

    def exitComponent_declaration(self, ctx):
        self.ast[ctx].comment = self.ast[ctx.comment()]

    def enterDeclaration(self, ctx):
        sym = self.symbol_node
        dimensions = None
        if self.comp_clause.dimensions is not None:
            dimensions = self.comp_clause.dimensions
        sym.name = ctx.IDENT().getText()
        sym.dimensions = dimensions
        sym.prefixes = self.comp_clause.prefixes
        sym.type = self.comp_clause.type
        if sym.name in self.class_node.symbols:
            raise IOError(sym.name, 'already defined')
        self.class_node.symbols[sym.name] = sym

    def exitDeclaration(self, ctx):
        sym = self.symbol_node
        if ctx.array_subscripts() is not None:
            sym.dimensions = self.ast[ctx.array_subscripts()]

    def exitElement_modification(self, ctx):
        if ctx.modification() is not None:
            modifications = self.ast[ctx.modification()]
        else:
            modifications = []
        self.ast[ctx] = ast.ElementModification(name=ctx.name().getText(), modifications=modifications)
        sym = self.symbol_node
        if ctx.name().getText() == 'start':
            sym.start = self.ast[ctx.modification().expression()]

    def exitModification_class(self, ctx):
        self.ast[ctx] = [self.ast[ctx.class_modification()]]
        if ctx.expression() is not None:
            self.ast[ctx] += [self.ast[ctx.expression()]]

    def exitModification_assignment(self, ctx):
        self.ast[ctx] = [self.ast[ctx.expression()]]

        # TODO wrong
        sym = self.symbol_node
        sym.start = self.ast[ctx.expression()]

    def exitModification_assignment2(self, ctx):
        self.ast[ctx] = [self.ast[ctx.expression()]]

    def exitElement_replaceable(self, ctx):
        self.ast[ctx] = self.ast[ctx.getChild(ctx.getAltNumber())]

    def exitElement_modification_or_replaceable(self, ctx):
        self.ast[ctx] = self.ast[ctx.getChild(ctx.getAltNumber())]

    def exitElement_redeclaration(self, ctx):
        self.ast[ctx] = []
        for symbol in self.ast[ctx.component_clause1()].symbol_list:
            symbol.redeclare = True

    # COMMENTS ==============================================================

    def exitComment(self, ctx):
        # TODO handle annotation
        self.ast[ctx] = self.ast[ctx.string_comment()]

    def exitString_comment(self, ctx):
        self.ast[ctx] = ctx.getText()[1:-1]


# UTILITY FUNCTIONS ========================================================

def parse(text):
    input_stream = antlr4.InputStream(text)
    lexer = ModelicaLexer(input_stream)
    stream = antlr4.CommonTokenStream(lexer)
    parser = ModelicaParser(stream)
    parse_tree = parser.stored_definition()
    ast_listener = ASTListener()
    parse_walker = antlr4.ParseTreeWalker()
    parse_walker.walk(ast_listener, parse_tree)
    ast_tree = ast_listener.ast_result
    return ast_tree
