#!/usr/bin/env python
"""Modelica translator/compiler tool using pymoca

This duplicates some things in casadi.api, but is useful for other backends
TODO: Perhaps refactor the parts common with casadi backend
"""
from __future__ import generators

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pymoca.ast
import pymoca.tree
from pymoca import __version__

# Import of backends delayed until needed

log = logging.getLogger("pymoca")
logging.basicConfig(stream=sys.stderr)

# In lieu of a pluggable backend api, we'll use this for now.
BACKEND_TARGET_OPTIONS = ("casadi", "sympy")


def list_modelica_files(paths: List[Path]) -> List[Path]:
    """Find all Modelica files in given paths (can be files and directories)

    :param paths: List of Paths to search
    :return: List of valid Modelica file Paths found
    """
    # Make flat list of .mo files
    files = []
    for path in paths:
        if path.is_file() and path.suffix == ".mo":
            files.append(path)
        elif path.is_dir():
            for glob_path in path.glob("**/*.mo"):
                files.append(glob_path)
    return files


def parse_file(path: Path) -> Union[pymoca.ast.Tree, None]:
    """Parse a Modelica file and return AST or None on failure

    :param path: single Path containing Modelica code
    :return: parsed ast.Tree or None on error
    """
    # Per comment in backends.casadi.api.py, antlr4 is slow to load, so delay to here
    # pylint: disable=redefined-outer-name
    import pymoca.parser  # pylint: disable=imports

    ast = None
    try:
        log.info("Parsing %s ...", path)
        with path.open(encoding="utf-8") as file:
            ast = pymoca.parser.parse(file.read())
        if ast is None:
            log.error('Syntax error in file "%s"', path)
        elif log.level == logging.DEBUG:
            log.debug(json.dumps(ast.to_json(ast), indent=2))
    # KeyError and AttributeError are problems in ASTListener
    except (KeyError, AttributeError, OSError):
        if log.level in (logging.DEBUG, logging.INFO):
            log.exception('Parse error in file "%s"', path)
        else:
            log.error('Parse error in file "%s"', path)
        return None
    return ast


def parse_all(paths: List[Path], ast: pymoca.ast.Tree = None) -> Tuple[List[Path], List[Path]]:
    """Parse a list of files and directory trees and add to given AST

    :param paths: List of files and diretory trees to parse
    :param ast: Optional ast.Tree to add parsed AST to
    :return: tuple (list of all .mo files, list of files with parse errors)
    """
    files = list_modelica_files(paths)
    if not files:
        return [], []
    if not ast:
        ast = pymoca.ast.Tree(name="ModelicaTree")
    # Parse all .mo files
    error_files = []
    for path in files:
        file_ast = parse_file(path)
        if file_ast:
            ast.extend(file_ast)
        else:
            error_files.append(path)
    return files, error_files


def flatten_class(library_ast: pymoca.ast.Tree, class_: str) -> pymoca.ast.Tree:
    """Flatten given class and return AST

    :param library_ast: Previously parsed AST containing the above class
    :param class_: Class to flatten, e.g. 'Package1.Package2.Model'
    :return: flattened pymoca.ast.Tree
    """
    log.info("Flattening %s ...", class_)
    component_ref = pymoca.ast.ComponentRef.from_string(class_)
    flat_tree = pymoca.tree.flatten(library_ast, component_ref)
    if log.level == logging.DEBUG:
        log.debug(json.dumps(flat_tree.to_json(flat_tree), indent=2))
    return flat_tree


def translate(
    library_ast: pymoca.ast.Tree,
    model: str,
    translator: str,
    options: dict,
    outdir: Optional[Path] = None,
) -> bool:
    """Given parsed Modelica AST, generate code for model into given directory

    :param library_ast: Previously parsed AST containing the above model class
    :param model: Modelica Class to generate code for
    :param translator: target translator to use (e.g. 'sympy' or 'casadi')
    :param options: dict of options to pass to translator
    :param outdir: directory to put results in
    :return: True on success, False on failure
    """
    if outdir is None:
        outdir = Path(".")

    log.info("Generating model for %s ...", model)
    # Currenly only support sympy; envision others being added in future
    if translator == "sympy":
        import pymoca.backends.sympy.generator as sympy_gen

        try:
            result = sympy_gen.generate(library_ast, model, options)
            outfile = outdir.joinpath(model + ".py")
            with outfile.open("w") as file:
                file.write(result)
        except OSError:
            if log.level is logging.DEBUG:
                log.exception('Error writing "%s"', outfile)
            else:
                log.error('Error writing "%s"', outfile)
            return False
        except KeyError:
            log.exception("Problem translating %s to SymPy", model)
            return False
    else:
        raise NotImplementedError("Translator for {} not implemented".format(translator))
    return True


def main(argv: List[str]) -> int:
    """Parse command line options and do the work

    :param argv: list of command line arguments, but not including program name
    :return: number of usage errors (not parse errors)
    """
    # TODO: Add better usage documentation in docstring
    argp = argparse.ArgumentParser(description="Translate Modelica files")
    argp.add_argument(
        "PATH",
        type=Path,
        nargs="+",
        help="Modelica files and directory trees, all of which are parsed",
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
    genargs = argp.add_argument_group("translation arguments", "without these, just parse only")
    genargs.add_argument(
        "-m",
        "--model",
        action="append",
        help="model to translate (e.g. Package1.Package2.ModelName); "
        "if no target specified, then flatten only",
    )
    genargs.add_argument(
        "-t", "--target", choices=BACKEND_TARGET_OPTIONS, help="code target to use"
    )
    genargs.add_argument(
        "-O",
        "--option",
        action="append",
        help="target generator option in the form NAME=VALUE with no spaces or quoted",
    )
    genargs.add_argument(
        "-o", "--outdir", type=Path, default=".", help="directory to contain generated model code"
    )

    args = argp.parse_args(argv)
    errors = 0  # For additional argument checks beyond what argparse does

    if args.verbose == 0:
        log.setLevel(logging.WARNING)
    elif args.verbose == 1:
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.DEBUG)

    # Check for invalid option combinations (argp.error will exit)
    if args.target and not args.model:
        argp.error("-t/--target requires -m/--model")
    if not args.target:
        if args.option:
            log.warning("Ignoring -O (options only used with -t/--target)")
        if args.model:
            log.info("No target specified (-t option), flattening model only")

    # Check that paths exist
    if not args.outdir.is_dir():
        log.error('Invalid output directory: "%s"', args.outdir)
        errors += 1
    for path in args.PATH:
        if not path.exists():
            log.error('File or directory does not exist: "%s"', path)
            errors += 1

    # Build target generator options dict from args
    options = {}
    if args.option:
        for opt in args.option:
            optsplit = opt.split("=")
            if len(optsplit) == 2:
                # Convert True/False values, otherwise string value as-is
                value_lower = optsplit[1].lower()
                if value_lower == "true":
                    optsplit[1] = True
                elif value_lower == "false":
                    optsplit[1] = False
                options[optsplit[0]] = optsplit[1]
            else:
                log.error('Invalid option syntax (need -O NAME=VALUE): "%s"', opt)
                errors += 1

    if errors:
        return errors

    tic = time.perf_counter()
    error_files = []  # type: List[Path]
    modelica_files = []  # type: List[Path]
    if args.target == "sympy" or not args.target:
        library_ast = pymoca.ast.Tree(name="ModelicaTree")
        modelica_files, error_files = parse_all(args.PATH, library_ast)
        if not modelica_files:
            errors += 1
            log.error("No Modelica files in given PATHs")
        elif error_files:
            errors += len(error_files)
        if not errors and args.model:
            for model in args.model:
                if args.target:
                    translate(library_ast, model, "sympy", options, args.outdir)
                elif args.model:
                    try:
                        _ = flatten_class(library_ast, model)
                    # tree.flatten_class can throw Exception in several places
                    except Exception:  # pylint: disable=broad-except
                        if log.level is logging.DEBUG:
                            log.exception("Error flattening %s", model)
                        else:
                            log.error("Error flattening %s", model)
                        errors += 1

    elif args.target == "casadi":
        import pymoca.backends.casadi.api as casadi_api

        modelica_files = list_modelica_files(args.PATH)
        if not modelica_files:
            errors += 1
            log.error("No Modelica files in given PATHs")
        else:
            for model in args.model:
                # Infer model directory to pass to casadi.api
                model_dir = None
                for path in modelica_files:
                    if path.stem == model:
                        if model_dir:
                            # More than one found (ambiguous)
                            log.error("More than one Modelica file found for %s", model)
                            errors += 1
                            model_dir = None
                            break
                        model_dir = path.parent
                if not model_dir:
                    log.error("No unique Modelica file corresponding to model %s", model)
                else:
                    log.info("Generating model for %s ...", model)
                    try:
                        _ = casadi_api.transfer_model(str(model_dir), model, options)
                    # TODO: Figure out more specific Exceptions for CasADi transfer_model
                    except Exception:  # pylint: disable=broad-except
                        if log.level is logging.DEBUG:
                            log.exception("Problem generating CasADi model %s", model)
                        else:
                            log.error("Problem generating CasADi model %s", model)
                        errors += 1

    else:
        # Should never get here because argparse should have caught above
        assert args.target in BACKEND_TARGET_OPTIONS
    toc = time.perf_counter()

    goodbye_message = "Finished in {:0.4f} seconds".format(toc - tic)
    if error_files:
        goodbye_message = " ".join(
            [
                goodbye_message,
                "with {} of {} files with parse errors.".format(
                    len(error_files), len(modelica_files)
                ),
            ]
        )
    log.info(goodbye_message)
    return errors


if __name__ == "__main__":
    err = main(sys.argv[1:])
    logging.shutdown()
    sys.exit(err)
