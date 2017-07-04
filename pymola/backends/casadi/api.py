from collections import namedtuple
import casadi as ca
import numpy as np
import sys
import os
import fnmatch
import logging
import shelve

from pymola import parser, tree, ast
from pymola.version import version as pymola_version
from . import generator
from .model import Model, Variable

logger = logging.getLogger("pymola")


class CachedModel(Model):
    def __init__(self):
        self.states = []
        self.der_states = []
        self.alg_states = []
        self.inputs = []
        self.outputs = []
        self.constants = []
        self.parameters = []
        self.time = ca.MX.sym('time')
        self.delayed_states = []

        self._dae_residual_function = None
        self._initial_residual_function = None
        self._variable_metadata_function = None

    def __str__(self):
        r = ""
        r += "Model\n"
        r += "time: " + str(self.time) + "\n"
        r += "states: " + str(self.states) + "\n"
        r += "der_states: " + str(self.der_states) + "\n"
        r += "alg_states: " + str(self.alg_states) + "\n"
        r += "inputs: " + str(self.inputs) + "\n"
        r += "outputs: " + str(self.outputs) + "\n"
        r += "constants: " + str(self.constants) + "\n"
        r += "parameters: " + str(self.parameters) + "\n"
        return r

    @property
    def dae_residual_function(self):
        return self._dae_residual_function

    @property
    def initial_residual_function(self):
        return self._initial_residual_function

    @property
    def variable_metadata_function(self):
        return self._variable_metadata_function

    @property
    def equations(self):
        raise NotImplementedError("Cannot access individual equations on cached model.  Use residual function instead.")

    @property
    def initial_equations(self):
        raise NotImplementedError("Cannot access individual equations on cached model.  Use residual function instead.")

    def simplify(self, options):
        raise NotImplementedError("Cannot simplify cached model")


class InvalidCacheError(Exception):
    pass


class ObjectData:
    # This is not a named tuple, since we need read/write access to 'library'
    def __init__(self, key, library):
        self.key = key
        self.library = library


def _compile_model(model_folder, model_name, compiler_options):
    # Load folders
    tree = None
    for folder in [model_folder] + compiler_options.get('library_folders', []):
        for root, dir, files in os.walk(folder, followlinks=True):
            for item in fnmatch.filter(files, "*.mo"):
                logger.info("Parsing {}".format(item))

                with open(os.path.join(root, item), 'r') as f:
                    if tree is None:
                        tree = parser.parse(f.read())
                    else:
                        tree.extend(parser.parse(f.read()))

    # Compile
    logger.info("Generating CasADi model")
    
    model = generator.generate(tree, model_name)
    if compiler_options.get('check_balanced', True):
        model.check_balanced()

    model.simplify(compiler_options)

    if compiler_options.get('verbose', False):
        model.check_balanced()

    return model

def _save_model(model_folder, model_name, model):
    # Compile shared libraries
    if os.name == 'posix':
        ext = 'so'
    else:
        ext = 'dll'
            
    objects = {'dae_residual': ObjectData('dae_residual', ''), 'initial_residual': ObjectData('initial_residual', ''), 'variable_metadata': ObjectData('variable_metadata', '')}
    for o, d in objects.items():
        f = getattr(model, o + '_function')

        # Generate C code
        library_name = '{}_{}'.format(model_name, o)

        cg = ca.CodeGenerator(library_name)
        cg.add(f)
        cg.add(f.forward(1))
        cg.add(f.reverse(1))
        cg.add(f.reverse(1).forward(1))
        cg.generate(model_folder + '/')

        file_name = os.path.join(model_folder, library_name + '.c')

        d.library = os.path.join(model_folder, '{}.{}'.format(library_name, ext))
        cc = os.getenv('CC', 'gcc')
        cflags = os.getenv('CFLAGS', '-O3 -fPIC')
        try:
            os.system("{} {} -shared {} -o {}".format(cc, cflags, file_name, d.library))
        except:
            raise
        finally:
            os.remove(file_name)

    # Output metadata     
    shelve_file = os.path.join(model_folder, model_name)   
    with shelve.open(shelve_file, 'n') as db:
        # Store version
        db['version'] = pymola_version

        # Include references to the shared libraries
        for o, d in objects.items():
            db[d.key] = d.library
        db['library_os'] = os.name

        # Describe variables per category
        for key in ['states', 'der_states', 'alg_states', 'inputs', 'outputs', 'parameters', 'constants']:
            db[key] = [e.to_dict() for e in getattr(model, key)]

        db['delayed_states'] = model.delayed_states

def _load_model(model_folder, model_name, compiler_options):
    shelve_file = os.path.join(model_folder, model_name)

    if compiler_options.get('mtime_check', True):
        # Mtime check
        cache_mtime = os.path.getmtime(shelve_file)
        for folder in [model_folder] + compiler_options.get('library_folders', []):
            for root, dir, files in os.walk(folder, followlinks=True):
                for item in fnmatch.filter(files, "*.mo"):
                    filename = os.path.join(root, item)
                    if os.path.getmtime(filename) > cache_mtime:
                        raise InvalidCacheError("Cache out of date")

    # Create empty model object
    model = CachedModel()

    # Compile shared libraries
    objects = {'dae_residual': ObjectData('dae_residual', ''), 'initial_residual': ObjectData('initial_residual', ''), 'variable_metadata': ObjectData('variable_metadata', '')}

    # Load metadata        
    with shelve.open(shelve_file, 'r') as db:
        if db['version'] != pymola_version:
            raise InvalidCacheError('Cache generated for a different version of pymola')

        if db['library_os'] != os.name:
            raise InvalidCacheError('Cache generated for incompatible OS')

        # Include references to the shared libraries
        for o, d in objects.items():
            f = ca.external(o, db[d.key])

            setattr(model, '_' + o + '_function', f)

        # Load variables per category
        variables_with_metadata = ['states', 'alg_states', 'inputs', 'parameters', 'constants']
        variable_dict = {}
        for key in variables_with_metadata:
            variables = getattr(model, key)
            for i, d in enumerate(db[key]):
                variable = Variable.from_dict(d)
                variables.append(variable)
                variable_dict[variable.symbol.name()] = variable

        model.der_states = [Variable.from_dict(d) for d in db['der_states']]
        model.outputs = [variable_dict[v['name']] for v in db['outputs']]
        model.delayed_states = db['delayed_states']

        # Evaluate variable metadata:
        # We do this in three passes, so that we have constant attributes available through the API,
        # and non-constant expressions as Function calls.

        # 1.  Extract independent parameter values
        metadata = dict(zip(variables_with_metadata, model.variable_metadata_function(ca.veccat(*[np.nan for v in model.parameters]))))
        for key in variables_with_metadata:
            for i, d in enumerate(db[key]):
                variable = variable_dict[d['name']]
                for j, tmp in enumerate(ast.Symbol.ATTRIBUTES):
                    setattr(variable, tmp, metadata[key][i, j])

        # 2.  Plug independent values back into metadata function, to obtain values (such as bounds, or starting values)
        # dependent upon independent parameter values.
        metadata = dict(zip(variables_with_metadata, model.variable_metadata_function(ca.veccat(*[v.value if v.value.is_regular() else np.nan for v in model.parameters]))))
        for key in variables_with_metadata:
            for i, d in enumerate(db[key]):
                variable = variable_dict[d['name']]
                for j, tmp in enumerate(ast.Symbol.ATTRIBUTES):
                    setattr(variable, tmp, metadata[key][i, j])

        # 3.  Fill in any irregular elements with expressions to be evaluated later.
        # Note that an expression is neccessary only if the function value actually depends on the inputs.
        # Otherwise, we would be dealing with a genuine NaN value.
        metadata = dict(zip(variables_with_metadata, model.variable_metadata_function(ca.veccat(*[v.value if v.value.is_regular() else v.symbol for v in model.parameters]))))
        for k, key in enumerate(variables_with_metadata):
            sparsity = model.variable_metadata_function.sparsity_jac(0, k)
            for i, d in enumerate(db[key]):
                variable = variable_dict[d['name']]
                for j, tmp in enumerate(ast.Symbol.ATTRIBUTES):
                    if not getattr(variable, tmp).is_regular():
                        depends_on_parameters = np.any([sparsity.has_nz(i * len(ast.Symbol.ATTRIBUTES) + j, l) for l in range(len(model.parameters))])
                        if depends_on_parameters:
                            setattr(variable, tmp, metadata[key][i, j])
    
    # Done
    return model

def transfer_model(model_folder, model_name, compiler_options={}):
    if compiler_options.get('cache', False):
        try:
            return _load_model(model_folder, model_name, compiler_options)
        except (FileNotFoundError, InvalidCacheError):
            model = _compile_model(model_folder, model_name, compiler_options)
            _save_model(model_folder, model_name, model)
            return model
    else:
        return _compile_model(model_folder, model_name, compiler_options)
        