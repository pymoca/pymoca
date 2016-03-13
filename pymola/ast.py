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
import itertools

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

class Expression(Node):
    pass

class Primary(Node):
    value =  ast.field((str, unicode), null=True)

class Expression(Node):
    operator = ast.field((str, unicode), null=True)
    operands = ast.seq((Primary, Expression), null=True)


class Equation(Node):
    left = ast.field((Expression, Node), null=True) # TODO switch to expr
    right = ast.field((Expression, Node), null=True)


class ConnectClause(Node):
    left = ast.field(str, null=True)
    right = ast.field(str, null=True)


class Symbol(Node):
    name = ast.field((str, unicode))
    type =  ast.field((str, unicode), null=True)
    prefixes = ast.seq((str, unicode), null=True)
    redeclare =  ast.field(bool, default=False)
    final =  ast.field(bool, default=False)
    inner =  ast.field(bool, default=False)
    outer =  ast.field(bool, default=False)
    dimensions =  ast.seq(int, null=True)


# This class is just an intermediate AST step
class ComponentClause(Node):
    prefixes = ast.seq((str, unicode), null=True)
    type =  ast.field((str, unicode), null=True)
    array_subscripts =  ast.seq(int, null=True)
    dimensions =  ast.seq(int, null=True)
    comment = ast.field(str, default='')
    symbol_list = ast.seq(Symbol, null=True)


# This class is just an intermediate AST step
class EquationSection(Node):
    initial =  ast.field(bool, default=False)
    equation_list = ast.seq((Equation, ConnectClause), null=True)


class Class(Node):
    name = ast.field((str, unicode), null=True)
    final =  ast.field(bool, default=False)
    encapsulated =  ast.field(bool, default=False)
    partial =  ast.field(bool, default=False)
    type =  ast.field((str, unicode), null=True)
    comment = ast.field(str, default='')
    symbols = ast.dict(Symbol, null=True)
    equations = ast.seq((Equation, ConnectClause), null=True)


class File(Node):
    within =  ast.field(bool, null=True)
    classes = ast.dict(Class, null=True)


class DAEListener(ModelicaListener):

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
        file_node = File()
        file_node.within = ctx.WITHIN() != None
        self.ast[ctx] = file_node
        self.file_node = file_node

    def exitStored_definition(self, ctx):
        for class_node in [self.ast[e] for e in ctx.stored_definition_class()]:
            self.ast[ctx].classes[class_node.name] = class_node
        self.ast_result = self.ast[ctx]

    def enterStored_definition_class(self, ctx):
        class_node = Class()
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
        eq_sect = EquationSection(
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
        print(type(self.ast[ctx.simple_expression()]))
        self.ast[ctx] = Equation(
            left=self.ast[ctx.simple_expression()],
            right=self.ast[ctx.expression()],
            comment=self.eq_comment)

    def exitEquation_connect_clause(self, ctx):
        self.ast[ctx] = self.ast[ctx.connect_clause()]

    def exitConnect_clause(self, ctx):
        self.ast[ctx] = ConnectClause(
            left=str(ctx.component_reference()[0].getText()),
            right=str(ctx.component_reference()[1].getText()),
            comment=self.eq_comment)

    def exitSimple_expression(self, ctx):
        #TODO only using first expression
        self.ast[ctx] = self.ast[ctx.expr()[0]]

    def exitExpression_simple(self, ctx):
        self.ast[ctx] = self.ast[ctx.simple_expression()]

    def enterExpr_primary(self, ctx):
        self.ast[ctx] = Primary(ctx.getText())

    def exitExpr_add(self, ctx):
        self.ast[ctx] = Expression(
            operator=str(ctx.op.text),
            operands=[self.ast[e] for e in ctx.expr()]
        )

    def exitExpr_mul(self, ctx):
        self.ast[ctx] = Expression(
            operator=str(ctx.op.text),
            operands=[self.ast[e] for e in ctx.expr()]
        )

    def exitExpr_neg(self, ctx):
        self.ast[ctx] = Expression(
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
        self.ast[ctx] = ComponentClause(
            prefixes=str(ctx.type_prefix().getText()).split(' '),
            type=ctx.type_specifier().getText(),
            dimensions=dimensions
        )
        self.comp_clause = self.ast[ctx]

    def exitComponent_clause(self, ctx):
        self.ast[ctx].symbol_list = self.ast[ctx.component_list()]

    def exitComponent_list(self, ctx):
        self.ast[ctx] = [self.ast[e] for e in ctx.component_declaration()]

    def exitComponent_declaration(self, ctx):
        self.ast[ctx] = self.ast[ctx.declaration()]

    def enterDeclaration(self, ctx):
        dimensions = None
        if ctx.array_subscripts() is not None:
            dimensions = [int(s) for s in ctx.array_subscripts().subscript().getText()]
        elif self.comp_clause.dimensions is not None:
            dimensions = self.comp_clause.dimensions
        sym = Symbol(
            name=ctx.IDENT().getText(),
            dimensions=dimensions,
            type=self.comp_clause.type,
            prefixes=self.comp_clause.prefixes
        )
        self.ast[ctx] = sym

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

def flatten(root, class_name, instance_name=''):
    """
    This function takes and flattens it so that all subclasses instances
    are replaced by the their equations and symbols with name mangling
    of the instance name passed.
    :param root: The root of the tree that contains all class definitions
    :param class_name: The class we want to flatten
    :param instance_name:
    :return: flat_class, the flattened class of type Class
    """

    # extract the original class of interest
    orig_class = root.classes[class_name]

    # create the returned class
    flat_class = Class(
        equations=orig_class.equations
    )

    # append period to non empty instance_name
    if instance_name != '':
        instance_name += '.'

    # for all symbols in the original class
    for sym_name, sym in orig_class.symbols.items():
        # if the symbol type is a class
        if sym.type in root.classes:
            # recursively call flatten on the sub class
            flat_sub_class = flatten(root, sym.type, instance_name=sym_name)
            # add sub_class members symbols and equations
            for sub_sym_name, sub_sym in flat_sub_class.symbols.items():
                flat_class.symbols[instance_name + sub_sym_name] = sub_sym
            flat_class.equations += flat_sub_class.equations

        # else if the symbols is not a class name
        else:
            # append original symbol to flat class
            flat_class.symbols[instance_name + sym_name] = sym
    return flat_class

if __name__ == "__main__":

    input_stream = antlr4.FileStream('./test/Aircraft.mo')
    lexer = ModelicaLexer(input_stream)
    stream = antlr4.CommonTokenStream(lexer)
    parser = ModelicaParser(stream)
    tree = parser.stored_definition()
    #print(tree.toStringTree(recog=parser))
    daeListener = DAEListener()
    walker = antlr4.ParseTreeWalker()
    walker.walk(daeListener, tree)

    tree = daeListener.ast_result
    #print(tree)
    flat_tree = flatten(daeListener.ast_result, 'Aircraft')
    print(flat_tree)

    #listener = Listener()
    #tree.walk(listener)


#print(flatten(tree, 'Aircraft', ''))