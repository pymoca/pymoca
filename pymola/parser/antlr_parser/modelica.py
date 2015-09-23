#!/usr/bin/env python
"""
Modelica compiler.
"""
from __future__ import print_function
import sys
import antlr4
import antlr4.Parser
from generated.ModelicaLexer import ModelicaLexer
from generated.ModelicaParser import ModelicaParser
from generated.ModelicaListener import ModelicaListener
import argparse
from pprint import pprint

#pylint: disable=invalid-name, no-self-use, missing-docstring

class TraceListener(ModelicaListener):

    def __init__(self, parser):
        self._parser = parser
        self._ctx = None

    def enterEveryRule(self, ctx):
        self._ctx = ctx
        print(" "*ctx.depth(), "enter   " +
              self._parser.ruleNames[ctx.getRuleIndex()] + ", LT(1)=" +
              self._parser._input.LT(1).text)

    def visitTerminal(self, node):
        print(" "*self._ctx.depth() + "consume " + str(node.symbol) +
              " rule " + self._parser.ruleNames[self._ctx.getRuleIndex()])

    def visitErrorNode(self, node):
        pass

    def exitEveryRule(self, ctx):
        print(" "*ctx.depth(), "exit    " + self._parser.ruleNames[ctx.getRuleIndex()] +
              ", LT(1)=" + self._parser._input.LT(1).text)


class SympyPrinter(ModelicaListener):

    #-------------------------------------------------------------------------
    # Setup
    #-------------------------------------------------------------------------

    def __init__(self, parser):
        """
        Constructor
        """
        self._val_dict = {}
        self.result = None
        self._data = {
            'parameters': {},
            'variables': [],
            'equations': [],
            'init': {},
        }
        self._parser = parser

    def setValue(self, ctx, val):
        """
        Sets tree values.
        """

        self._val_dict[ctx] = val

    def getValue(self, ctx):
        """
        Gets tree values.
        """
        return self._val_dict[ctx]

    #-------------------------------------------------------------------------
    # Top Level
    #-------------------------------------------------------------------------

    def exitStored_definition(self, ctx):
        c = ctx.class_definition()[0].class_specifier().composition()
        self.result = self.getValue(c)

    def exitComposition(self, ctx):
        s = "# declaration\n"
        s += self.getValue(ctx.element_list()[0])
        s += "\n"
        s += "# equations\n"
        s += self.getValue(ctx.equation_section()[0])
        self.setValue(ctx, s)

    def exitElement_list(self, ctx):
        s = ""
        for element in ctx.element():
            s += self.getValue(element)
        self.setValue(ctx, s)

    def exitElement(self, ctx):
        self.setValue(ctx, self.getValue(ctx.component_clause()))

    def exitEquation_section(self, ctx):
        val = "eqs=["
        for eq in ctx.equation():
            data = self.getValue(eq)
            if data is not None:
                val += "\n\t{:s},".format(data)
        val += "\n\t]\n"
        self.setValue(ctx, val)

    def exitComponent_clause(self, ctx):
        s = ""
        if ctx.type_prefix().getText() == 'parameter':
            # store all parameters
            for comp in ctx.component_list().component_declaration():
                name = comp.declaration().IDENT().getText()
                val = comp.declaration().modification().expression().getText()
                self._data['parameters'][name] = float(val)
                s += "{:s} = {:s}".format(name, val)
        else:
            # store all variables
            for comp in ctx.component_list().component_declaration():
                name = comp.declaration().IDENT().getText()
                mod = comp.declaration().modification().class_modification()
                val = mod.argument_list().getText()
                self._data['init'][name] = val
                s += "{:s} = {:s}".format(name, val)
        self.setValue(ctx, s)

    #-------------------------------------------------------------------------
    # Equation
    #-------------------------------------------------------------------------

    def exitEquation(self, ctx):
        self.setValue(ctx, self.getValue(ctx.equation_options()))

    def exitEquation_simple(self, ctx):
        self.setValue(
            ctx,
            self.getValue(ctx.simple_expression()) + '='
            + self.getValue(ctx.expression()))

    def exitEquation_function(self, ctx):
        self.setValue(ctx, None)

    def exitEquation_when_clause(self, ctx):
        self.setValue(ctx, None)

    #-------------------------------------------------------------------------
    # Expression
    #-------------------------------------------------------------------------

    def exitExpression_simple(self, ctx):
        self.setValue(ctx, self.getValue(ctx.simple_expression()))

    def exitSimple_expression(self, ctx):
        self.setValue(ctx, self.getValue(ctx.expr()[0]))

    def exitExpr_primary(self, ctx):
        self.setValue(ctx, self.getValue(ctx.primary()))

    def exitExpr_neg(self, ctx):
        self.setValue(ctx, '-({:s})'.format(self.getValue(ctx.expr())))

    def exitExpr_rel(self, ctx):
        self.setValue(ctx, '({:s} {:s} {:s})'.format(
            ctx.expr()[0], ctx.op, ctx.expr()[1]))

    def exitExpr_mul(self, ctx):
        self.setValue(ctx, '({:s} {:s} {:s})'.format(
            ctx.expr()[0], ctx.op, ctx.expr()[1]))

    #-------------------------------------------------------------------------
    # Primary
    #-------------------------------------------------------------------------

    def exitPrimary_unsigned_number(self, ctx):
        self.setValue(ctx, ctx.getText())

    def exitPrimary_component_reference(self, ctx):
        self.setValue(ctx, ctx.getText())

    def exitPrimary_derivative(self, ctx):
        name = ctx.function_call_args().function_arguments().function_argument().getText()
        self._data['variables'] += [name]
        self.setValue(ctx, 'diff({:s}, t)'.format(name))

    def exitPrimary_string(self, ctx):
        self.setValue(ctx, ctx.getText())

    def exitPrimary_false(self, ctx):
        self.setValue(ctx, ctx.getText())

def main(argv):
    #pylint: disable=unused-argument
    "The main function"
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    text = antlr4.FileStream(args.filename)
    lexer = ModelicaLexer(text)
    stream = antlr4.CommonTokenStream(lexer)
    parser = ModelicaParser(stream)
    tree = parser.stored_definition()
    # print(tree.toStringTree(recog=parser))
    sympyPrinter = SympyPrinter(parser)
    walker = antlr4.ParseTreeWalker()
    walker.walk(sympyPrinter, tree)

    print(sympyPrinter.result)

    # trace = TraceListener(parser)
    # walker.walk(trace, tree)

if __name__ == '__main__':
    main(sys.argv)

# vim: set et ft=python fenc=utf-8 ff=unix sts=0 sw=4 ts=4 :
