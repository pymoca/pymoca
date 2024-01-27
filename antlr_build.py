import os
import subprocess

from setuptools import Command


class AntlrBuildCommand(Command):
    """Customized setuptools build command."""

    user_options = []

    def initialize_options(self):
        """initialize options"""
        pass

    def finalize_options(self):
        """finalize options"""
        pass

    def run(self):
        "Run the build command"
        call_antlr4("Modelica.g4")


def call_antlr4(arg):
    "calls antlr4 on grammar file"
    # pylint: disable=unused-argument, unused-variable
    antlr_path = os.path.join(ROOT_DIR, "java", "antlr-4.13.1-complete.jar")
    classpath = os.pathsep.join([".", "{:s}".format(antlr_path), "$CLASSPATH"])
    generated = os.path.join(ROOT_DIR, "src", "pymoca", "generated")
    cmd = (
        'java -Xmx500M -cp "{classpath:s}" org.antlr.v4.Tool {arg:s}'
        " -o {generated:s} -visitor -Dlanguage=Python3".format(**locals())
    )
    print(cmd)
    proc = subprocess.Popen(cmd.split(), cwd=os.path.join(ROOT_DIR, "src", "pymoca"))
    proc.communicate()
    with open(os.path.join(generated, "__init__.py"), "w") as fid:
        fid.write("")


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    call_antlr4("Modelica.g4")
