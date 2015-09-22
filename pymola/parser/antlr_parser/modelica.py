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

    def __init__(self):
        self._equations = []
        self._variables = []
        print ('Hello')

    def exitEveryRule(self, ctx):
        # pprint(ctx.__dict__)
        depth = ctx.depth()
        # print('-'*depth)
        # pprint(ctx.__dict__)
        # pprint(ctx.parser.__dict__)
        #print(ctx.name)
        ctx.value = 1
        # print('-'*depth, ctx.getText())

    def exitElementList(self, ctx):
        print ('element list:', ctx.getText())
        ctx.value = ctx.getText()
 
    def exitEquation_section(self, ctx):
        # print ('equation section:', ctx.getText())
        # ctx.value = ctx.getText()
        pass

    def exitAlgorithm_section(self, ctx):
        # print ('algorithm section:', ctx.getText())
        # ctx.value = ctx.getText()
        pass

    def exitEquation(self, ctx):
        # print ('equation:', ctx.getText())
        pass

    def exitExpression(self, ctx):
        pass
        # print('expression: ', ctx.getText())

    def exitLogical_factor(self, ctx):
        # print('logical factor', ctx.getText())
        pass

    def exitStored_definition(self, ctx):
        pass

    def exitClass_definition(self, ctx):
        pass

    def exitClass_prefixes(self, ctx):
        pass

    def exitPrimary(self, ctx):
        ctx.value = ctx.getText()
        print('primary')
        print('\ttext:', ctx.getText())
        print('\tval:', ctx.value)

    def exitTerm(self, ctx):
        ctx.value = ctx.factors[0].value
        for i in range(1, len(ctx.factors)):
            ctx.value += ctx.ops[i-1].value * ctx.factors[i].value
        print('term')
        print('\ttext:', ctx.getText())
        print('\tval:', ctx.value)

    def exitFactor(self, ctx):
        print('factor text:', ctx.getText())
        print('base text:', ctx.base.getText())
        print('base value:', ctx.base.value)
        base = ctx.base.value
        print('base:', base)
        if ctx.exp:
            exp = ctx.exp.value
            ctx.value = "{:s}**{:s}".format(base, exp)
        else:
            ctx.value = "{:s}".format(ctx.base.getText())

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
    sympyPrinter = SympyPrinter()
    walker = antlr4.ParseTreeWalker()
    walker.walk(sympyPrinter, tree)

    # trace = TraceListener(parser)
    # walker.walk(trace, tree)

if __name__ == '__main__':
    main(sys.argv)

# vim: set et ft=python fenc=utf-8 ff=unix sts=0 sw=4 ts=4 :
