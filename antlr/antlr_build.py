"""Generate parser code from given grammar file"""

import os
import subprocess

# Root directory of the repo is one level up from this script
ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]


def call_antlr4(arg):
    "calls antlr4 on grammar file"
    # pylint: disable=unused-argument, unused-variable
    antlr_path = os.path.join(ROOT_DIR, "antlr", "antlr-4.13.1-complete.jar")
    classpath = os.pathsep.join([".", f"{antlr_path:s}", "$CLASSPATH"])
    generated = os.path.join(ROOT_DIR, "src", "pymoca", "generated")
    cmd = (
        f'java -Xmx500M -cp "{classpath:s}" org.antlr.v4.Tool {arg:s}'
        f" -o {generated:s} -visitor -Dlanguage=Python3"
    )
    print(cmd)
    with subprocess.Popen(cmd.split(), cwd=os.path.join(ROOT_DIR, "src", "pymoca")) as proc:
        proc.communicate()
    with open(os.path.join(generated, "__init__.py"), "w", encoding="utf-8") as fid:
        fid.write("")


if __name__ == "__main__":
    call_antlr4("Modelica.g4")
