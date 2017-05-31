#!/usr/bin/env python
"""
Compiler tool.
"""

from optparse import OptionParser
from collections import namedtuple
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
    model.simplify(replace_constants=True, replace_parameter_expressions=True)

    # Compile shared libraries
    if os.name == 'posix':
        ext = 'so'
    else:
        ext = 'dll'

    class ObjectData:
        def __init__(self, key, derivatives, library):
            self.key = key
            self.derivatives = derivatives
            self.library = library
            
    objects = {'dae_residual': ObjectData('dae_residual', True, ''), 'initial_residual': ObjectData('initial_residual', True, ''), 'state_metadata': ObjectData('state_metadata', False, '')}
    for o, d in objects.items():
        f = getattr(model, o + '_function')(group_arguments=True)
        print(f.name())
        f.print_dimensions()

        # Generate C code
        library_name = '{}_{}'.format(model_name, o)

        cg = ca.CodeGenerator(library_name)
        cg.add(f)
        if d.derivatives:
            cg.add(f.forward(1))
            cg.add(f.reverse(1))
            cg.add(f.reverse(1).forward(1))
        cg.generate()

        file_name = library_name + '.c'

        d.library = '{}.{}'.format(library_name, ext)
        try:
            os.system("clang -shared {} -o {}".format(file_name, d.library))
        except:
            raise
        finally:
            os.remove(file_name)

    # Output metadata        
    import shelve
    with shelve.open(model_name, 'n') as db:
        # Include references to the shared libraries
        for o, d in objects.items():
            db[d.key] = d.library
        db['library_os'] = os.name

        # Describe variables per category
        Variable = namedtuple('Variable', ['name', 'value', 'aliases'])
        for key in ['states', 'der_states', 'alg_states', 'inputs', 'outputs']:
            db[key] = [Variable(e.name(), None, []) for e in getattr(model, key)]

        db['parameters'] = [Variable(e.name(), v, []) for e, v in zip(model.parameters, model.parameter_values)]

        DelayedVariable = namedtuple('DelayedVariable', ['name', 'origin', 'delay'])
        db['delayed_states'] = [DelayedVariable(t[0].name(), t[1].name(), t[2]) for t in model.delayed_states]
