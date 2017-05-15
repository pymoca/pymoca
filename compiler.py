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
    logger.info("Flattening")

    ast = tree.flatten(ast, model_name)
    print(ast)
else:
    logger.info("Generating CasADi model")
    
    model = gen_casadi.generate(ast, model_name)
    model.check_balanced()
    
    f = model.get_function(replace_constants=True)
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
    so_file_name = '{}.{}'.format(model_name, ext)
    try:
        os.system("clang -shared {} -o {}".format(file_name, so_file_name))
    except:
        raise
    finally:
        os.remove(file_name)

    # Output metadata        
    from collections import namedtuple
    import shelve
    with shelve.open(model_name, 'n') as db:
        # Include a reference to the shared library
        db['library'] = so_file_name
        db['library_os'] = os.name

        # Describe variables per category
        Variable = namedtuple('Variable', ['name', 'aliases'])
        for key in ['states', 'der_states', 'alg_states', 'parameters', 'inputs', 'outputs']:
            db[key] = [Variable(x.name(), []) for x in getattr(model, key)]

        DelayedVariable = namedtuple('DelayedVariable', ['name', 'origin', 'delay'])
        db['delayed_states'] = [DelayedVariable(t[0].name(), t[1].name(), t[2]) for t in model.delayed_states]
