#!/usr/bin/env python
"""A python/modelica based simulation environment.

Pymola contains a Python based compiler for the modelica language
and enables interacting with Modelica easily in Python.

"""

from __future__ import print_function
from setuptools import setup

import os
import sys
import subprocess
import pprint
import shutil
import fnmatch

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext
from setuptools import Command

from Cython.Build import cythonize
import versioneer

MAJOR = 0
MINOR = 1
MICRO = 5
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 1 - Planning
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
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


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

#pylint: disable=no-init, too-few-public-methods


import sys
python_version = '.'.join([str(i) for i in sys.version_info[:3]])
python_version_required = '3.4.0'
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
    antlr_path = os.path.join(ROOT_DIR, "java", "antlr-4.7-complete.jar")
    classpath = ".:{:s}:$CLASSPATH".format(antlr_path)
    generated = os.path.join(ROOT_DIR, 'pymola', 'generated')
    cmd = "java -Xmx500M -cp \"{classpath:s}\" org.antlr.v4.Tool {arg:s}" \
            " -o {generated:s} -visitor -Dlanguage=Python3".format(**locals())
    print(cmd)
    proc = subprocess.Popen(cmd.split(), cwd=os.path.join(ROOT_DIR, 'pymola'))
    proc.communicate()
    with open(os.path.join(ROOT_DIR, 'pymola', 'generated', '__init__.py'), 'w') as fid:
        fid.write('')
    for root, dir, files in os.walk(generated):
        for item in fnmatch.filter(files, "Modelica*.py"):
            filename = os.path.join(root, item)
            shutil.move(filename, filename.replace('.py', '.pyx'))

def setup_package():
    """
    Setup the package.
    """
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    # Install requirements
    with open('requirements.txt', 'r') as req_file:
        install_reqs = req_file.read().split('\n')
    # pprint.pprint(install_reqs)

    # Disable compiler optimization.  We have to do this, as the default -O3 triggers a bug in clang causing an initialization failure.
    #os.environ['CFLAGS'] = '-O0'
    # Or alternatively, use gcc:
    
    os.environ['CC'] = 'gcc'

    # Without disabling this, it will reach a limit
    # on tracking and disable tracking and then recompile, which is slow
    os.environ['CFLAGS'] = '-fno-var-tracking-assignments'

    metadata = dict(
        version=versioneer.get_version(),
        name='pymola',
        maintainer="James Goppert",
        maintainer_email="james.goppert@gmail.com",
        description=DOCLINES[0],
        long_description="\n".join(DOCLINES[2:]),
        url='https://github.com/jgoppert/pymola',
        author='James Goppert',
        author_email='james.goppert@gmail.com',
        download_url='https://github.com/jgoppert/pymola',
        license='BSD',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        install_requires=install_reqs,
        tests_require=['coverage >= 3.7.1', 'nose >= 1.3.1'],
        test_suite='nose.collector',
        packages=find_packages(
            # choosing to distribute tests
            # exclude=['*.test']
        ),
        cmdclass={
            'antlr': AntlrBuildCommand,
            'versioneer': versioneer.get_cmdclass(),  
        },
        ext_modules=cythonize('pymola/generated/*.pyx',
            compiler_directives={'boundscheck': False,
                'initializedcheck': False, 'language_level': 3})
    )

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return

if __name__ == '__main__':
    setup_package()
