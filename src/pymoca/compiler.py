#!/usr/bin/env python
"""Modelica translator/compiler tool using pymoca"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pprint
import sys
import time
from enum import IntEnum
from pathlib import Path

import pymoca.ast
from pymoca import __version__

log = logging.getLogger("pymoca")
logging.basicConfig(stream=sys.stderr)

# Registry of available translation backends.
TRANSLATOR_TYPES = ("casadi", "sympy")


class Stage(IntEnum):
    """Pipeline stages; value ordering defines execution sequence"""

    PARSE = 0
    INSTANTIATE = 1
    FLATTEN = 2
    TRANSLATE = 3


class MyArgumentParser(argparse.ArgumentParser):
    """Make options file treat each space separated word as a separate argument"""

    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()


def list_modelica_files(paths: list[Path]) -> list[Path]:
    """Find all Modelica files in given paths (can be files and directories)"""
    files = []
    for path in paths:
        if path.is_file() and path.suffix == ".mo":
            files.append(path)
        elif path.is_dir():
            for glob_path in path.glob("**/*.mo"):
                files.append(glob_path)
    return files


def parse_file(path: Path) -> pymoca.ast.Tree | None:
    """Parse a Modelica file and return AST or None on failure"""
    import pymoca.parser

    ast = None
    try:
        log.info("Parsing %s ...", path)
        try:
            ast = pymoca.parser.parse_file(path)
        except pymoca.parser.ModelicaSyntaxError as err:
            pymoca.parser.print_syntax_error(err)
            if log.level in (logging.DEBUG, logging.INFO):
                log.exception('Syntax error in file "%s"', path)
            else:
                log.error('Syntax error in file "%s": %s', path, err)
        else:
            if log.level == logging.DEBUG:
                log.debug(json.dumps(ast.to_json(ast), indent=2))
    # KeyError and AttributeError are problems in ASTListener
    except (KeyError, AttributeError, OSError, UnicodeDecodeError) as err:
        if log.level in (logging.DEBUG, logging.INFO):
            log.exception('Parse error in file "%s"', path)
        else:
            log.error('Parse error in file "%s": %s', path, err)
        return None
    return ast


def parse_all(
    paths: list[Path], ast: pymoca.ast.Tree | None = None
) -> tuple[list[Path], list[Path]]:
    """Parse a list of files and directory trees and add to given AST

    Returns: tuple (list of all .mo files, list of files with parse errors)
    """
    files = list_modelica_files(paths)
    if not files:
        return [], []
    if not ast:
        ast = pymoca.ast.Tree(name="ModelicaTree")
    error_files = []
    for path in files:
        try:
            file_ast = parse_file(path)
        except NotImplementedError as exc:
            log.debug("Skipping %s: %s", path, exc)
            continue
        if file_ast:
            ast.extend(file_ast)
        else:
            error_files.append(path)
    return files, error_files


def translate(
    library_ast: pymoca.ast.Tree,
    model: str,
    translator: str,
    options: dict,
    outdir: Path | None = None,
) -> bool:
    """Generate code for model from library_ast into outdir

    Returns: True on success, False on failure
    """
    if outdir is None:
        outdir = Path(".")

    log.info("Generating model for %s ...", model)
    if translator == "sympy":
        import pymoca.backends.sympy.generator as sympy_gen

        outfile = outdir / (model + ".py")
        try:
            result = sympy_gen.generate(library_ast, model, options)
            with outfile.open("w") as file:
                file.write(result)
        except OSError:
            if log.level == logging.DEBUG:
                log.exception('Error writing "%s"', outfile)
            else:
                log.error('Error writing "%s"', outfile)
            return False
        except KeyError:
            log.exception("Problem translating %s to SymPy", model)
            return False
    elif translator == "casadi":
        import pymoca.backends.casadi.api as casadi_api

        try:
            casadi_model = casadi_api.generate_model(library_ast, model, options)
        except Exception:  # pylint: disable=broad-except
            if log.level == logging.DEBUG:
                log.exception("Problem generating CasADi model %s", model)
            else:
                log.error("Problem generating CasADi model %s", model)
            return False
        try:
            casadi_api.save_model(str(outdir), model, casadi_model, options)
        except OSError:
            if log.level == logging.DEBUG:
                log.exception('Error writing CasADi cache for "%s" to "%s"', model, outdir)
            else:
                log.error('Error writing CasADi cache for "%s" to "%s"', model, outdir)
            return False
    else:
        raise NotImplementedError("Translator for {} not implemented".format(translator))
    return True


def emit_stage_output(model: str, stage_name: str, result, args) -> None:
    """Write pipeline stage result to stdout or a file in outdir"""
    if args.output_format == "repr":
        text, ext = pprint.pformat(result.to_json(result), width=100), "txt"
    else:
        text, ext = json.dumps(result.to_json(result), indent=2), "json"
    if args.outdir is not None:
        outfile = args.outdir / "{}.{}.{}".format(model, stage_name, ext)
        with outfile.open("w") as f:
            f.write(text)
        log.info("Stage output written to %s", outfile)
    else:
        print(text)


def run_model(library_ast: pymoca.ast.Tree, model: str, stage: Stage, args, options: dict) -> int:
    """Run the pipeline up to stage for model

    Returns: error count (0 or 1)
    """
    import pymoca.ast
    import pymoca.parser
    import pymoca.tree

    try:
        if stage == Stage.PARSE:
            result = pymoca.tree.find_name(library_ast, model)
            if not isinstance(result, pymoca.ast.Class):
                log.error("Model %s not found in parsed files", model)
                return 1
            emit_stage_output(model, "parse", result, args)
        elif stage == Stage.INSTANTIATE:
            result = pymoca.tree.instantiate(library_ast, model)
            emit_stage_output(model, "instantiate", result, args)
        elif stage == Stage.FLATTEN:
            result = pymoca.tree.flatten_class(library_ast, model)
            emit_stage_output(model, "flatten", result, args)
        elif stage == Stage.TRANSLATE:
            outdir = args.outdir if args.outdir is not None else Path(".")
            if not translate(library_ast, model, args.translator, options, outdir):
                return 1
    except pymoca.parser.ModelicaSyntaxError as exc:
        import io

        buf = io.StringIO()
        pymoca.parser.print_syntax_error(exc, buf)
        log.error("Syntax error while processing %s:\n%s", model, buf.getvalue())
        return 1
    except pymoca.tree.ModelicaError as exc:
        log.error("Error at %s stage for %s: %s", stage.name.lower(), model, exc)
        if log.level == logging.DEBUG:
            log.exception("Traceback for %s", model)
        return 1
    except NotImplementedError as exc:
        log.error("Not implemented at %s stage for %s: %s", stage.name.lower(), model, exc)
        return 1
    except Exception:  # pylint: disable=broad-except
        if log.level == logging.DEBUG:
            log.exception("Unexpected error at %s stage for %s", stage.name.lower(), model)
        else:
            log.error("Unexpected error at %s stage for %s", stage.name.lower(), model)
        return 1
    return 0


def build_arg_parser() -> MyArgumentParser:
    """Build and return the argument parser."""
    example_help = """
    Examples:

        Parse all files in "MSL-4.0.x" directory tree, printing any errors and time taken:
            pymoca -v test/libraries/MSL-4.0.x

        Instantiate the OpAmp model and dump its instance tree as JSON:
            pymoca -v -p test/libraries/MSL-4.0.x \\
                -m Modelica.Electrical.Analog.Basic.OpAmp --stage instantiate

        Flatten "Spring" model and write to a file:
            pymoca -p test/models -m Spring --stage flatten -o out/

        Generate SymPy code for "Spring" model:
            pymoca -p test/models -m Spring -t sympy -D verbose=true -o out/

        Read some options above from a file:
            pymoca -p test/models -m Spring @args.txt
            where args.txt contains:
            -t sympy
            -o out/
    """
    argp = MyArgumentParser(
        description="Translate Modelica code to specified output code",
        epilog=example_help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@",
    )
    argp.add_argument(
        "PATHNAME",
        type=Path,
        nargs="*",
        help="Modelica files and directory trees, all of which are parsed",
    )
    argp.add_argument(
        "-p",
        "--path",
        action="append",
        default=[],
        help='"{}" separated path list to add to MODELICAPATH'.format(os.pathsep),
    )
    argp.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="print extra info; -vv is even more verbose",
    )
    argp.add_argument(
        "--version", action="version", version=__version__, help="print pymoca version"
    )
    genargs = argp.add_argument_group(
        "pipeline arguments", "control which stage to run and what to output"
    )
    genargs.add_argument(
        "-m",
        "--model",
        action="append",
        help="model to process (e.g. Package1.Package2.ModelName)",
    )
    genargs.add_argument(
        "-t",
        "--translator",
        choices=TRANSLATOR_TYPES,
        help="translate Modelica to this output type (implies --stage translate)",
    )
    genargs.add_argument(
        "--stage",
        choices=("parse", "instantiate", "flatten", "translate"),
        default=None,
        help="stop at and emit output from this pipeline stage "
        "(default: translate if -t given, flatten if -m given, otherwise parse, "
        "which emits output only when --stage parse is given explicitly)",
    )
    genargs.add_argument(
        "--format",
        dest="output_format",
        choices=("json", "repr"),
        default="json",
        help="serialization format for intermediate stage output (default: json)",
    )
    genargs.add_argument(
        "-D",
        "--define",
        action="append",
        help="translator option in the form NAME=VALUE with no spaces or quoted",
    )
    genargs.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help="directory for stage output files; if omitted, intermediate stages print to stdout",
    )
    return argp


def validate_args(argp: MyArgumentParser, args, modelicapath_env: str) -> None:
    """Check for invalid argument combinations; calls argp.error() (sys.exit 2) on failure"""
    if not (args.PATHNAME or args.path or modelicapath_env):
        argp.error("PATHNAME or -p/--path or environment variable MODELICAPATH required")
    if (args.path or modelicapath_env) and not (args.model or args.PATHNAME):
        argp.error("Nothing to do. Specify at least -m/--model")
    if args.translator and not args.model:
        argp.error("-t/--translator requires -m/--model")
    if args.stage == "translate" and not args.translator:
        argp.error("--stage translate requires -t/--translator")
    if args.translator == "casadi":
        if args.stage and args.stage != "translate":
            argp.error("CasADi backend only supports --stage translate")
    if not args.translator:
        if args.define:
            log.warning("Ignoring -D (options only used with -t/--translator)")
        if args.model and not args.stage:
            log.info("No translator specified (-t), flattening model only")


def build_define_options(args) -> tuple[dict, int]:
    """Parse -D NAME=VALUE args into a translator options dict.

    Returns: (options dict, error count)
    """
    options: dict = {}
    errors = 0
    if not args.define:
        return options, errors
    for opt in args.define:
        parts = opt.split("=")
        if len(parts) == 2:
            key, raw_value = parts
            value_lower = raw_value.lower()
            if value_lower == "true":
                value: str | bool = True
            elif value_lower == "false":
                value = False
            else:
                value = raw_value
            options[key] = value
        else:
            log.error('Invalid option syntax (need -D NAME=VALUE): "%s"', opt)
            errors += 1
    return options, errors


def infer_stage(args) -> Stage:
    """Determine pipeline stop stage from explicit --stage or implicit flags"""
    if args.stage:
        return Stage[args.stage.upper()]
    if args.translator:
        return Stage.TRANSLATE
    if args.model:
        return Stage.FLATTEN
    return Stage.PARSE


def _run_pipeline(args, modelica_path: list[Path], stage: Stage, options: dict) -> int:
    """Parse files then run each model through the new pipeline up to stage"""
    import pymoca.parser

    library_ast = pymoca.parser.modelicapath_to_tree(dirs=modelica_path)
    modelica_files, error_files = parse_all(args.PATHNAME, library_ast)

    errors = 0
    if not modelica_files and not modelica_path:
        log.error("No Modelica files in given PATHNAMEs")
        errors += 1
    else:
        errors += len(error_files)

    if not errors:
        if args.model:
            for model in args.model:
                errors += run_model(library_ast, model, stage, args, options)
        elif args.stage == "parse":
            emit_stage_output(library_ast.name or "root", "parse", library_ast, args)

    if error_files:
        log.info("%d of %d files had parse errors", len(error_files), len(modelica_files))

    return errors


def main(argv: list[str] | None = None) -> int:
    """Parse command line options and run the pipeline.

    :param argv: command line arguments, not including program name (pass "-h" for help);
        None reads sys.argv, as when invoked via the installed ``pymoca`` script
    :return: number of errors
    """
    argp = build_arg_parser()
    args = argp.parse_args(argv)

    if args.verbose == 0:
        log.setLevel(logging.WARNING)
    elif args.verbose == 1:
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.DEBUG)

    import pymoca.parser

    modelicapath_env = os.getenv(pymoca.parser.MODELICAPATH_ENV_VAR, default="")
    validate_args(argp, args, modelicapath_env)  # may call sys.exit(2)

    modelica_path: list[Path] = []
    errors = 0
    try:
        modelica_path = pymoca.parser.resolve_modelicapath(args.path)
    except pymoca.parser.ModelicaPathError as err:
        log.error("%s", err)
        errors += 1

    options, opt_errors = build_define_options(args)
    errors += opt_errors

    if args.outdir is not None and not args.outdir.is_dir():
        if args.outdir.is_file():
            log.error('Invalid output directory: "%s"', args.outdir)
            errors += 1
        else:
            try:
                args.outdir.mkdir()
            except OSError:
                log.error('Unable to create output directory: "%s"', args.outdir)
                errors += 1

    for path in args.PATHNAME:
        if not path.exists():
            log.error('File or directory does not exist: "%s"', path)
            errors += 1

    if errors:
        return errors

    stage = infer_stage(args)
    tic = time.perf_counter()

    errors = _run_pipeline(args, modelica_path, stage, options)

    toc = time.perf_counter()
    log.info("Finished in {:0.4f} seconds".format(toc - tic))
    return errors


def console_main() -> None:
    """Console script entry point; exit status is the error count clamped to 255"""
    err = main()
    logging.shutdown()
    sys.exit(min(err, 255))


if __name__ == "__main__":
    console_main()
