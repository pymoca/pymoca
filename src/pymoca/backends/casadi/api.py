from collections import namedtuple
from typing import Dict
import distutils.ccompiler
import casadi as ca
import numpy as np
import sys
import os
import fnmatch
import logging
import pickle
import contextlib

from pymoca import parser, tree, ast, __version__
from . import generator
from .model import Model, Variable

logger = logging.getLogger("pymoca")


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

def _compile_model(model_folder: str, model_name: str, compiler_options: Dict[str, str]):
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

    model = generator.generate(tree, model_name, compiler_options)
    if compiler_options.get('check_balanced', True):
        model.check_balanced()

    model.simplify(compiler_options)

    if compiler_options.get('verbose', False):
        model.check_balanced()

    return model

def _codegen_model(model_folder: str, f: ca.Function, library_name: str):
    # Compile shared libraries
    if os.name == 'posix':
        compiler_flags = ['-O2', '-fPIC']
        linker_flags = ['-fPIC']
    else:
        compiler_flags = ['/O2', '/wd4101']  # Shut up unused local variable warnings.
        linker_flags = ['/DLL']

    # Generate C code
    logger.debug("Generating {}".format(library_name))

    cg = ca.CodeGenerator(library_name)
    cg.add(f, True) # Nondifferentiated function
    cg.add(f.forward(1), True) # Jacobian-times-vector product
    cg.add(f.reverse(1), True) # vector-times-Jacobian product
    cg.add(f.reverse(1).forward(1), True) # Hessian-times-vector product
    cg.generate(model_folder + '/')

    compiler = distutils.ccompiler.new_compiler()

    file_name = os.path.relpath(os.path.join(model_folder, library_name + '.c'))
    object_name = compiler.object_filenames([file_name])[0]
    library = os.path.join(model_folder, library_name + compiler.shared_lib_extension)
    try:
        # NOTE: For some reason running in debug mode in PyCharm (2017.1)
        # on Windows causes cl.exe to fail on its own binary name (?!) and
        # the include paths. This does not happen when running directly
        # from cmd.exe / PowerShell or e.g. with debug mode in VS Code.
        compiler.compile([file_name], extra_postargs=compiler_flags)

        # We do not want the "lib" prefix on POSIX systems, so we call
        # link() directly with our desired filename instead of
        # link_shared_lib().
        compiler.link(compiler.SHARED_LIBRARY, [object_name], library, extra_preargs=linker_flags)
    except:
        raise
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(file_name)
            os.remove(object_name)
    return library

def _save_model(model_folder: str, model_name: str, model: Model,
                compiler_options: Dict[str, str], cache=True, codegen=False):
    # Compilation takes precedence over caching, and disables it
    if cache and codegen:
        logger.warning("Both 'cache' and 'codegen' specified. Code generation will take precedence.")
        cache = False

    objects = {'dae_residual': None, 'initial_residual': None, 'variable_metadata': None}
    for o in objects.keys():
        f = getattr(model, o + '_function')

        if codegen:
            objects[o] = _codegen_model(model_folder, f, '{}_{}'.format(model_name, o))
        else:
            objects[o] = f

    # Output metadata
    db_file = os.path.join(model_folder, model_name)
    with open(db_file, 'wb') as f:
        db = {}

        # Store version
        db['version'] = __version__

        # Include references to the shared libraries (codegen) or pickled functions (cache)
        db.update(objects)

        db['library_os'] = os.name

        db['options'] = compiler_options

        # Describe variables per category
        for key in ['states', 'der_states', 'alg_states', 'inputs', 'parameters', 'constants']:
            db[key] = [e.to_dict() for e in getattr(model, key)]

        db['outputs'] = model.outputs

        db['delayed_states'] = model.delayed_states

        pickle.dump(db, f)

def _load_model(model_folder: str, model_name: str, compiler_options: Dict[str, str]) -> CachedModel:
    db_file = os.path.join(model_folder, model_name)

    if compiler_options.get('mtime_check', True):
        # Mtime check
        cache_mtime = os.path.getmtime(db_file)
        for folder in [model_folder] + compiler_options.get('library_folders', []):
            for root, dir, files in os.walk(folder, followlinks=True):
                for item in fnmatch.filter(files, "*.mo"):
                    filename = os.path.join(root, item)
                    if os.path.getmtime(filename) > cache_mtime:
                        raise InvalidCacheError("Cache out of date")

    # Create empty model object
    model = CachedModel()

    # Load metadata
    with open(db_file, 'rb') as f:
        db = pickle.load(f)

        if db['version'] != __version__:
            raise InvalidCacheError('Cache generated for a different version of pymoca')

        if db['library_os'] != os.name:
            raise InvalidCacheError('Cache generated for incompatible OS')

        if db['options'] != compiler_options:
            raise InvalidCacheError('Cache generated for different compiler options')

        # Include references to the shared libraries
        for o in ['dae_residual', 'initial_residual', 'variable_metadata']:
            if isinstance(db[o], str):
                # Path to codegen'd library
                f = ca.external(o, db[o])
            else:
                # Pickled CasADi Function; use as is
                assert isinstance(db[o], ca.Function)
                f = db[o]

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
        model.outputs = db['outputs']
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
        parameter_vector = ca.veccat(*[v.symbol for v in model.parameters])
        metadata = dict(zip(variables_with_metadata, model.variable_metadata_function(ca.veccat(*[v.value if v.value.is_regular() else v.symbol for v in model.parameters]))))
        for k, key in enumerate(variables_with_metadata):
            for i, d in enumerate(db[key]):
                variable = variable_dict[d['name']]
                for j, tmp in enumerate(ast.Symbol.ATTRIBUTES):
                    if not getattr(variable, tmp).is_regular():
                        if (not isinstance(metadata[key][i, j], ca.DM)
                            and ca.depends_on(metadata[key][i, j], parameter_vector)):
                            setattr(variable, tmp, metadata[key][i, j])

    # Done
    return model

def transfer_model(model_folder: str, model_name: str, compiler_options=None):
    if compiler_options is None:
        compiler_options = {}
    cache = compiler_options.get('cache', False)
    codegen = compiler_options.get('codegen', False)

    if cache or codegen:
        # Until CasADi supports pickling MXFunctions, caching implies
        # expanding to SX. We only raise a warning when we have to (re)compile
        # the model though.
        raise_expand_warning = False
        if cache and not compiler_options.get('expand_mx', False):
            compiler_options['expand_mx'] = True
            raise_expand_warning = True

        try:
            return _load_model(model_folder, model_name, compiler_options)
        except (FileNotFoundError, InvalidCacheError):
            if raise_expand_warning:
                logger.warning("Caching implies expanding to SX. Setting 'expand_mx' to True.")
            model = _compile_model(model_folder, model_name, compiler_options)
            _save_model(model_folder, model_name, model, compiler_options, cache, codegen)
            return model
    else:
        return _compile_model(model_folder, model_name, compiler_options)
