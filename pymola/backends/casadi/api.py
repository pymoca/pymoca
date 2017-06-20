from collections import namedtuple
import casadi as ca
import sys
import os
import fnmatch
import logging
import shelve

from pymola import parser, tree
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


class ObjectData:
    # This is not a named tuple, since we need read/write access to 'library'
    def __init__(self, key, derivatives, library):
        self.key = key
        self.derivatives = derivatives
        self.library = library

def _compile_model(model_folder, model_name, compiler_options):
    # Load folders
    ast = None
    for folder in [model_folder] + compiler_options.get('library_folders', []):
        for root, dir, files in os.walk(folder, followlinks=True):
            for item in fnmatch.filter(files, "*.mo"):
                logger.info("Parsing {}".format(item))

                with open(os.path.join(root, item), 'r') as f:
                    if ast is None:
                        ast = parser.parse(f.read())
                    else:
                        ast.extend(parser.parse(f.read()))

    # Compile
    logger.info("Generating CasADi model")
    
    model = generator.generate(ast, model_name)
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
            
    objects = {'dae_residual': ObjectData('dae_residual', True, ''), 'initial_residual': ObjectData('initial_residual', True, ''), 'variable_metadata': ObjectData('variable_metadata', False, '')}
    for o, d in objects.items():
        f = getattr(model, o + '_function')
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
        cg.generate(model_folder + '/')

        file_name = os.path.join(model_folder, library_name + '.c')

        d.library = os.path.join(model_folder, '{}.{}'.format(library_name, ext))
        cc = os.getenv('CC', 'gcc')
        cflags = os.getenv('CFLAGS', '-O3')
        try:
            os.system("{} {} -shared {} -o {}".format(cc, cflags, file_name, d.library))
        except:
            raise
        finally:
            os.remove(file_name)

    # Output metadata     
    shelve_file = os.path.join(model_folder, model_name)   
    with shelve.open(shelve_file, 'n') as db:
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
        ast = None
        for folder in [model_folder] + compiler_options.get('library_folders', []):
            for root, dir, files in os.walk(folder, followlinks=True):
                for item in fnmatch.filter(files, "*.mo"):
                    filename = os.path.join(root, item)
                    if os.path.getmtime(filename) > cache_mtime:
                        raise OSError("Cache out of date")

    # Create empty model object
    model = CachedModel()

    # Compile shared libraries
    objects = {'dae_residual': ObjectData('dae_residual', True, ''), 'initial_residual': ObjectData('initial_residual', True, ''), 'variable_metadata': ObjectData('variable_metadata', False, '')}

    # Load metadata        
    with shelve.open(shelve_file, 'r') as db:
        if db['library_os'] != os.name:
            raise OSError('Cache generated for incompatible OS')

        # Include references to the shared libraries
        for o, d in objects.items():
            f = ca.external(o, db[d.key])
            print(f.name())
            f.print_dimensions()

            setattr(model, '_' + o + '_function', f)

        # Evaluate variable metadata
        metadata = dict(zip(['states', 'alg_states', 'inputs', 'parameters', 'constants'], model.variable_metadata_function(ca.veccat(*[p.symbol for p in model.parameters]))))

        # Load variables per category
        variable_dict = {}
        for key in metadata.keys():
            variables = getattr(model, key)
            for i, d in enumerate(db[key]):
                variable = Variable.from_dict(d)
                for j, tmp in enumerate(model.VARIABLE_METADATA):
                    setattr(variable, tmp, metadata[key][i, j])
                variables.append(variable)
                variable_dict[variable.symbol.name()] = variable

        model.der_states = [Variable.from_dict(d) for d in db['der_states']]
        model.outputs = [variable_dict[v['name']] for v in db['outputs']]
        model.delayed_states = db['delayed_states']
    
    # Done
    return model

def transfer_model(model_folder, model_name, compiler_options={}):
    if compiler_options.get('cache', True):
        try:
            return _load_model(model_folder, model_name, compiler_options)
        except OSError:
            model = _compile_model(model_folder, model_name, compiler_options)
            _save_model(model_folder, model_name, model)
            return model
    else:
        return _compile_model(model_folder, model_name, compiler_options)


    
