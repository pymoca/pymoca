from collections import namedtuple
from typing import Dict
import distutils.ccompiler
import casadi as ca
import numpy as np
import copy
import sys
import os
import fnmatch
import logging
import pickle
import contextlib

from pymoca import parser, tree, ast, __version__
from . import generator
from .model import CASADI_ATTRIBUTES, Model, Variable, DelayArgument

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
        self.delay_states = []
        self.delay_arguments = []

        self._dae_residual_function = None
        self._initial_residual_function = None
        self._variable_metadata_function = None
        self._delay_arguments_function = None

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
    def delay_arguments_function(self):
        return self._delay_arguments_function

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

    model._post_checks()

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

def save_model(model_folder: str, model_name: str, model: Model,
               compiler_options: Dict[str, str]) -> None:
    """
    Saves a CasADi model to disk.

    :param model_folder: Folder where the precompiled CasADi model will be stored.
    :param model_name: Name of the model.
    :param model: Model instance.
    :param compiler_options: Dictionary of compiler options.
    """

    objects = {'dae_residual': None, 'initial_residual': None, 'variable_metadata': None, 'delay_arguments': None}
    for o in objects.keys():
        f = getattr(model, o + '_function')

        if compiler_options.get('codegen', False):
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


        # Caching using CasADi functions will lead to constants seemingly
        # depending on MX variables. Figuring out that they do not is slow,
        # especially when doing it on a lazy function call, as would be the
        # case when reading from cache. So instead, we do the depency check
        # once when saving the model.

        # Metadata dependency checking
        parameter_vector = ca.veccat(*[v.symbol for v in model.parameters])

        for k, key in enumerate(['states', 'alg_states', 'inputs', 'parameters', 'constants']):
            metadata_shape = (len(getattr(model, key)), len(CASADI_ATTRIBUTES))
            m = db[key + "__metadata_dependent"] = np.zeros(metadata_shape, dtype=bool)
            for i, v in enumerate(getattr(model, key)):
                for j, tmp in enumerate(CASADI_ATTRIBUTES):
                    attr = getattr(v, tmp)
                    if isinstance(attr, ca.MX) and ca.depends_on(attr, parameter_vector):
                        m[i, j] = True

        # Delay dependency checking
        if model.delay_states:

            all_symbols = [model.time,
                           *model._symbols(model.states),
                           *model._symbols(model.der_states),
                           *model._symbols(model.alg_states),
                           *model._symbols(model.inputs),
                           *model._symbols(model.constants),
                           *model._symbols(model.parameters)]
            symbol_to_index = {x: i for i, x in enumerate(all_symbols)}

            expressions, durations = zip(*model.delay_arguments)

            duration_dependencies = []
            for dur in durations:
                duration_dependencies.append(
                    [symbol_to_index[var] for var in ca.symvar(dur) if ca.depends_on(dur, var)])
            db['__delay_duration_dependent'] = duration_dependencies

        db['outputs'] = model.outputs

        db['delay_states'] = model.delay_states

        pickle.dump(db, f)

def load_model(model_folder: str, model_name: str, compiler_options: Dict[str, str]) -> CachedModel:
    """
    Loads a precompiled CasADi model into a CachedModel instance.

    :param model_folder: Folder where the precompiled CasADi model is located.
    :param model_name: Name of the model.
    :param compiler_options: Dictionary of compiler options.

    :returns: CachedModel instance.
    """

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

        # Check compiler options. We ignore the library folders, as they have
        # already been checked, and checking them will impede platform
        # portability of the cache.
        exclude_options = ['library_folders']
        old_opts = {k: v for k, v in db['options'].items() if k not in exclude_options}
        new_opts = {k: v for k, v in compiler_options.items() if k not in exclude_options}

        if old_opts != new_opts:
            raise InvalidCacheError('Cache generated for different compiler options')

        # Pickles are platform independent, but dynamic libraries are not
        if compiler_options.get('codegen', False):
            if db['library_os'] != os.name:
                raise InvalidCacheError('Cache generated for incompatible OS')

        # Include references to the shared libraries
        for o in ['dae_residual', 'initial_residual', 'variable_metadata', 'delay_arguments']:
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
        model.delay_states = db['delay_states']

        # Evaluate variable metadata:
        parameter_vector = ca.veccat(*[v.symbol for v in model.parameters])
        metadata = dict(zip(variables_with_metadata, model.variable_metadata_function(parameter_vector)))
        independent_metadata = dict(zip(variables_with_metadata, model.variable_metadata_function(ca.veccat(*[np.nan for v in model.parameters]))))

        for k, key in enumerate(variables_with_metadata):
            m = db[key + "__metadata_dependent"]
            for i, d in enumerate(db[key]):
                variable = variable_dict[d['name']]
                for j, tmp in enumerate(CASADI_ATTRIBUTES):
                    if m[i, j]:
                        setattr(variable, tmp, metadata[key][i, j])
                    else:
                        setattr(variable, tmp, independent_metadata[key][i, j])

        # Evaluate delay arguments:
        if model.delay_states:
            args = [model.time,
                    ca.veccat(*model._symbols(model.states)),
                    ca.veccat(*model._symbols(model.der_states)),
                    ca.veccat(*model._symbols(model.alg_states)),
                    ca.veccat(*model._symbols(model.inputs)),
                    ca.veccat(*model._symbols(model.constants)),
                    ca.veccat(*model._symbols(model.parameters))]
            delay_arguments_raw = model.delay_arguments_function(*args)

            nan_args = [ca.repmat(np.nan, *arg.size()) for arg in args]
            independent_delay_arguments_raw = model.delay_arguments_function(*nan_args)

            delay_expressions_raw = delay_arguments_raw[::2]
            delay_durations_raw = delay_arguments_raw[1::2]
            independent_delay_durations_raw = independent_delay_arguments_raw[1::2]

            assert 1 == len({len(delay_expressions_raw), len(delay_durations_raw),
                len(independent_delay_durations_raw)})

            all_symbols = [model.time,
                           *model._symbols(model.states),
                           *model._symbols(model.der_states),
                           *model._symbols(model.alg_states),
                           *model._symbols(model.inputs),
                           *model._symbols(model.constants),
                           *model._symbols(model.parameters)]

            duration_dependencies = db['__delay_duration_dependent']

            for i, expr in enumerate(delay_expressions_raw):
                if duration_dependencies[i]:
                    dur = delay_durations_raw[i]

                    deps = set(ca.symvar(dur))
                    actual_deps = {all_symbols[j] for j in duration_dependencies[i]}
                    false_deps = deps - actual_deps

                    if false_deps:
                        [dur] = ca.substitute(
                            [dur],
                            list(false_deps),
                            [ca.repmat(np.nan, *d.size()) for d in false_deps])
                else:
                    dur = independent_delay_durations_raw[i]

                model.delay_arguments.append(DelayArgument(expr, dur))

        # Try to coerce parameters into their Python types
        for p in model.parameters:
            for attr in CASADI_ATTRIBUTES:
                v = getattr(p, attr)
                v_mx = ca.MX(v)
                if v_mx.is_constant() and v_mx.is_regular():
                    setattr(p, attr, p.python_type(v))

    # Done
    return model

def transfer_model(model_folder: str, model_name: str, compiler_options=None):
    if compiler_options is None:
        compiler_options = {}
    else:
        compiler_options = copy.copy(compiler_options)

    cache = compiler_options.setdefault('cache', False)
    codegen = compiler_options.setdefault('codegen', False)

    # Compilation takes precedence over caching, and disables it
    if cache and codegen:
        logger.warning("Both 'cache' and 'codegen' specified. Code generation will take precedence.")
        cache = compiler_options['cache'] = False

    if cache or codegen:
        # Until CasADi supports pickling MXFunctions, caching implies
        # expanding to SX. We only raise a warning when we have to (re)compile
        # the model though.
        raise_expand_warning = False
        if cache and not compiler_options.get('expand_mx', False):
            compiler_options['expand_mx'] = True
            raise_expand_warning = True

        try:
            return load_model(model_folder, model_name, compiler_options)
        except (FileNotFoundError, InvalidCacheError):
            if raise_expand_warning:
                logger.warning("Caching implies expanding to SX. Setting 'expand_mx' to True.")
            model = _compile_model(model_folder, model_name, compiler_options)
            save_model(model_folder, model_name, model, compiler_options)
            return model
    else:
        return _compile_model(model_folder, model_name, compiler_options)
