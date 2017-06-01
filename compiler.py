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
logging.basicConfig(level=logging.DEBUG if options.verbose else logging.INFO)

# Import rest of pymola
from pymola import parser, tree    

# Compile
if options.flatten_only:
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

    logger.info("Flattening")

    ast = tree.flatten(ast, model_name)
    print(ast)
else:
    # Set CasADi installation folder
    if options.casadi_folder is not None:
        sys.path.append(options.casadi_folder)

    from pymola.backends.casadi.api import transfer_model
    import casadi as ca

    logger.info("Generating CasADi model")
    
    compiler_options = \
        {'replace_constants': True,
         'replace_parameter_expressions': True,
         'eliminable_variable_expression': r'_\w+',
         'detect_aliases': True,
         'cache': False}

    model = transfer_model(model_folder, model_name, compiler_options)
    print(model)
