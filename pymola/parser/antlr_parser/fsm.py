#!/usr/bin/env python
"""
Fsm compiler.
"""
from __future__ import print_function
import sys
import antlr4
from generated.FsmLexer import FsmLexer
from generated.FsmParser import FsmParser
from generated.FsmListener import FsmListener
import argparse

class KeyPrinter(FsmListener):
    "Simple example"
    def exitFsm_state(self, ctx):
        "print msg when leaving state"
        print("leaving state")

def main(argv):
    "The main function"
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    text = antlr4.FileStream(args.filename)
    lexer = FsmLexer(text)
    stream = antlr4.CommonTokenStream(lexer)
    parser = FsmParser(stream)
    tree = parser.fsm_main()
    print(tree.toStringTree(recog=parser))
    printer = KeyPrinter()
    walker = antlr4.ParseTreeWalker()
    walker.walk(printer, tree)

if __name__ == '__main__':
    main(sys.argv)

# vim: set et ft=python fenc=utf-8 ff=unix sts=0 sw=4 ts=4 :
