#!/usr/bin/env python
"""A python/modelica based simulation environment.

Pymoca contains a Python based compiler for the modelica language
and enables interacting with Modelica easily in Python.

"""

from __future__ import print_function

import os
import sys
import subprocess

from setuptools import setup, find_packages
from setuptools import Command

import versioneer

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


python_version = '.'.join([str(i) for i in sys.version_info[:3]])
python_version_required = '3.5.0'
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
    generated = os.path.join(ROOT_DIR, 'src', 'pymoca', 'generated')
    cmd = "java -Xmx500M -cp \"{classpath:s}\" org.antlr.v4.Tool {arg:s}" \
            " -o {generated:s} -visitor -Dlanguage=Python3".format(**locals())
    print(cmd)
    proc = subprocess.Popen(cmd.split(), cwd=os.path.join(ROOT_DIR, 'src', 'pymoca'))
    proc.communicate()
    with open(os.path.join(generated, '__init__.py'), 'w') as fid:
        fid.write('')

def setup_package():
    """
    Setup the package.
    """
    with open('requirements.txt', 'r') as req_file:
        install_reqs = req_file.read().split('\n')

    cmdclass_ = {'antlr': AntlrBuildCommand}
    cmdclass_.update(versioneer.get_cmdclass())

    setup(
        version=versioneer.get_version(),
        name='pymoca',
        maintainer="James Goppert",
        maintainer_email="james.goppert@gmail.com",
        description=DOCLINES[0],
        long_description="\n".join(DOCLINES[2:]),
        url='https://github.com/pymoca/pymoca',
        author='James Goppert',
        author_email='james.goppert@gmail.com',
        download_url='https://github.com/pymoca/pymoca',
        license='BSD',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        install_requires=install_reqs,
        tests_require=['coverage >= 3.7.1', 'nose >= 1.3.1'],
        test_suite='nose.collector',
        python_requires='>=3.5',
        packages=find_packages("src"),
        package_dir={"": "src"},
        cmdclass=cmdclass_
    )


if __name__ == '__main__':
    setup_package()
