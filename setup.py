#!/usr/bin/env python
"""A python/modelica based simulation environment.

Pymola contains a Python based compiler for the modelica language
and enables interacting with Modelica easily in Python.

"""

from __future__ import print_function
import os
import sys
import subprocess
from pip.req import parse_requirements
import pprint

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext
from setuptools import Command

MAJOR = 0
MINOR = 0
MICRO = 5
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 1 - Planning
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Other
Topic :: Software Development
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Physics
Topic :: Scientific/Engineering :: Visualization
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Topic :: Software Development :: Code Generators
Topic :: Software Development :: Compilers
Topic :: Software Development :: Embedded Systems
"""

#pylint: disable=no-init, too-few-public-methods


import sys
python_version = '.'.join([str(i) for i in sys.version_info[:3]])
python_version_required = '2.7.0'
if python_version < python_version_required:
    sys.exit("Sorry, only Python >= {:s} is supported".format(python_version_required))

class AntlrBuildCommand(Command):
    """Customized setuptools build command."""
    user_options=[]
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        "Run the build command"
        call_antlr4('Modelica.g4')

def call_antlr4(arg):
    "calls antlr4 on grammar file"
    #pylint: disable=unused-argument, unused-variable
    antlr_path = os.path.join(ROOT_DIR, "java", "antlr-4.5.1-complete.jar")
    classpath = ".:{:s}:$CLASSPATH".format(antlr_path)
    generated = os.path.join(ROOT_DIR, 'pymola', 'generated')
    cmd = "java -Xmx500M -cp \"{classpath:s}\" org.antlr.v4.Tool {arg:s}" \
            " -o {generated:s} -Dlanguage=Python2".format(**locals())
    print(cmd)
    proc = subprocess.Popen(cmd.split(), cwd=os.path.join(ROOT_DIR, 'pymola'))
    proc.communicate()
    with open(os.path.join(ROOT_DIR, 'pymola', 'generated', '__init__.py'), 'w') as fid:
        fid.write('')

def git_version():
    "Return the git revision as a string"
    def _minimal_ext_cmd(cmd):
        "construct minimal environment"
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            var = os.environ.get(k)
            if var is not None:
                env[k] = var
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "Unknown"

    return git_revision


def get_version_info():
    """
    Adding the git rev number needs to be done inside write_version_py(),
    otherwise the import of package.version messes up
    the build under Python 3.
    """
    full_version = VERSION
    if os.path.exists('.git'):
        git_revision = git_version()
    elif os.path.exists('pymola/version.py'):
        # must be a source distribution, use existing version file
        try:
            from pymola.version import git_revision as git_revision
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "pymola/version.py and the build directory "
                              "before building.")
    else:
        git_revision = "Unknown"

    if not ISRELEASED:
        full_version += '.git.' + git_revision[:7]

    return full_version, git_revision


def write_version_py(filename='pymola/version.py'):
    """
    Create a version.py file.
    """
    cnt = """
# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    full_version, git_revision = get_version_info()

    with open(filename, 'w') as fid:
        try:
            fid.write(cnt % {'version': VERSION,
                             'full_version': full_version,
                             'git_revision': git_revision,
                             'isrelease': str(ISRELEASED)})
        finally:
            fid.close()


def setup_package():
    """
    Setup the package.
    """
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    # Rewrite the version file everytime
    write_version_py()

    # Install requirements
    with open('requirements.txt', 'r') as req_file:
        install_reqs = req_file.read().split('\n')
    if sys.version_info[0] == 2:
        with open('requirements-py2.txt', 'r') as req_file:
            install_reqs += req_file.read().split('\n')
    elif sys.version_info[0] == 3:
        with open('requirements-py3.txt', 'r') as req_file:
            install_reqs += req_file.read().split('\n')
    # pprint.pprint(install_reqs)

    metadata = dict(
        name='pymola',
        maintainer="James Goppert",
        maintainer_email="james.goppert@gmail.com",
        description=DOCLINES[0],
        long_description="\n".join(DOCLINES[2:]),
        url='https://github.com/jgoppert/pymola',
        author='James Goppert',
        author_email='james.goppert@gmail.com',
        download_url='https://github.com/jgoppert/pymola',
        license='GPLv3+',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        install_requires=install_reqs,
        tests_require=['coverage >= 4.0', 'nose >= 1.3.7'],
        test_suite='nose.collector',
        packages=find_packages(
            # choosing to distribute tests
            # exclude=['*.test']
        ),
        cmdclass={
            'antlr': AntlrBuildCommand,
        }
    )

    full_version = get_version_info()[0]
    metadata['version'] = full_version

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return

if __name__ == '__main__':
    setup_package()
