#!/usr/bin/env python
"""
Compiler tool.
"""

from optparse import OptionParser
import sys
import os
import fnmatch
import logging

logger = logging.getLogger("pymola")

# Parse command line arguments
usage = "usage: %prog [options] MODEL_FOLDER MODEL_NAME"
parser = OptionParser(usage)
parser.add_option("-c", "--casadi", dest="casadi_folder",
                  help="CasADi installation folder")
parser.add_option("-f", "--flatten_only",
                  action="store_true", dest="flatten_only")
parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose")
(options, args) = parser.parse_args()
if len(args) != 2:
    parser.error("incorrect number of arguments")

model_folder = args[0]
model_name = args[1]

# Set log level
if options.verbose:
    logging.basicConfig(level=logging.DEBUG)

# Set CasADi installation folder
if options.casadi_folder is not None:
    sys.path.append(options.casadi_folder)

# Import rest of pymola
from pymola import parser, tree
if not options.flatten_only:
    from pymola import gen_casadi
    import casadi as ca

# Load folder
ast = None
for root, dir, files in os.walk(model_folder, followlinks=True):
    for item in fnmatch.filter(files, "*.mo"):
        logger.info("Parsing {}".format(item))

        with open(os.path.join(root, item), 'r') as f:
            if ast is None:
                ast = parser.parse(f.read())
            else:
                ast.extend(parser.parse(f.read()))

# Compile
if options.flatten_only:
    ast = tree.flatten(ast, model_name)
    print(ast)
else:
    model = gen_casadi.generate(ast, model_name)
    model.check_balanced()
    
    f = model.get_function()
    f.print_dimensions()

    # Generate C code
    cg = ca.CodeGenerator(model_name)
    cg.add(f)
    cg.add(f.forward(1))
    cg.add(f.reverse(1))
    cg.add(f.reverse(1).forward(1))
    cg.generate()

    file_name = model_name + '.c'

    # Compile shared library
    if os.name == 'posix':
        ext = 'so'
    else:
        ext = 'dll'
    try:
        os.system("clang -shared {} -o lib{}.{}".format(file_name, model_name, ext))
    except:
        raise
    finally:
        os.remove(file_name)
