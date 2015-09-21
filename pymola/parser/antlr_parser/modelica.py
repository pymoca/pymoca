#!/usr/bin/env python
"""
Modelica compiler.
"""
from __future__ import print_function
import sys
import antlr4
from generated.ModelicaLexer import ModelicaLexer
from generated.ModelicaParser import ModelicaParser
from generated.ModelicaListener import ModelicaListener
import argparse
from pprint import pprint

#pylint: disable=invalid-name, no-self-use, missing-docstring

template="""
class Test(object):
    
    def __init__(self):
        pass
"""

class KeyPrinter(ModelicaListener):

    def enterEveryRule(self, ctx):
        # pprint(ctx.__dict__)
        depth = ctx.depth()
        # print('-'*depth)
        # pprint(ctx.__dict__)
        ctx.value = 1

    def exitStored_definition(self, ctx):
        pass

    def exitClass_definition(self, ctx):
        pass

    def exitClass_prefixes(self, ctx):
        pass

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
    print(tree.toStringTree(recog=parser))
    printer = KeyPrinter()
    walker = antlr4.ParseTreeWalker()
    walker.walk(printer, tree)

if __name__ == '__main__':
    main(sys.argv)

# vim: set et ft=python fenc=utf-8 ff=unix sts=0 sw=4 ts=4 :
