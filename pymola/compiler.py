#!/usr/bin/env python
"""
Compiler tool.
"""

from optparse import OptionParser
import sys
import os
import fnmatch

# Parse command line arguments
usage = "usage: %prog [options] MODEL_FOLDER MODEL_NAME"
parser = OptionParser(usage)
parser.add_option("-c", "--casadi", dest="casadi_folder",
                  help="CasADi installation folder")
parser.add_option("-f", "--flatten_only",
                  action="store_true", dest="flatten_only")
(options, args) = parser.parse_args()
if len(args) != 2:
    parser.error("incorrect number of arguments")

model_folder = args[0]
model_name = args[1]

# Set CasADi installation folder
if options.casadi_folder is not None:
    sys.path.append(options.casadi_folder)

# Import rest of pymola
from . import parser, tree, gen_casadi

# Load folder
S = ''
for root, dir, files in os.walk(model_folder):
    for items in fnmatch.filter(files, "*.mo"):
        with open(os.path.join(root, items), 'r') as f:
            S += f.read()

# Compile
ast = parser.parse(S)
if options.flatten_only:
    ast = tree.flatten(ast, model_name)
    print(ast)
else:
    gen_casadi.generate(ast, model_name)