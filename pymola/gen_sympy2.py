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
import jinja2
import os


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

#pylint: disable=invalid-name, no-self-use, missing-docstring, unused-variable, protected-access
#pylint: disable=too-many-public-methods

class SympyPrinter(ModelicaListener):

    #-------------------------------------------------------------------------
    # Setup
    #-------------------------------------------------------------------------

    def __init__(self, parser, trace):
        """
        Constructor
        """
        self._parser = parser
        self._trace = trace
        self.result = None

    def exitStored_definition(self, ctx):
        d = locals()
        d['walker'] = d['self']
        d.pop('self', None)
        env = jinja2.Environment(
            loader=jinja2.PackageLoader('pymola', 'templates'),
            extensions=['jinja2.ext.do']
            )   
        template = env.get_template('sympy.jinja')
        self.result = template.render(**d)

def main(argv):
    #pylint: disable=unused-argument
    "The main function"
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('out')
    parser.add_argument('-t', '--trace', action='store_true')
    parser.set_defaults(trace=False)
    args = parser.parse_args(argv)
    text = antlr4.FileStream(args.src)
    lexer = ModelicaLexer(text)
    stream = antlr4.CommonTokenStream(lexer)
    parser = ModelicaParser(stream)
    tree = parser.stored_definition()
    # print(tree.toStringTree(recog=parser))
    sympyPrinter = SympyPrinter(parser, args.trace)
    walker = antlr4.ParseTreeWalker()
    walker.walk(sympyPrinter, tree)

    with open(args.out, 'w') as f:
        f.write(sympyPrinter.result)

if __name__ == '__main__':
    main(sys.argv[1:])

# vim: set et ft=python fenc=utf-8 ff=unix sts=0 sw=4 ts=4 :
