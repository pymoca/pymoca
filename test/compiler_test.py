#!/usr/bin/env python
"""
Test compiler tool
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest

import tools.compiler

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(FILE_DIR, "models")
GENERATED_DIR = os.path.join(FILE_DIR, "generated")
SPRING_MODEL = os.path.join(MODEL_DIR, "Spring.mo")
AIRCRAFT_MODEL = os.path.join(MODEL_DIR, "Aircraft.mo")
BOUNCING_BALL_XML = os.path.join(MODEL_DIR, "bouncing-ball.xml")


class CompilerTest(unittest.TestCase):
    """Compiler tool test. Leave testing of pymoca details to other tests.

    When argparse catches errors in command arguments it calls sys.exit(2).
    If arguments pass argparse, number of errors is expected to be returned from compiler.
    """

    def run_compiler(self, args, check_errors=True):
        "Run compiler with all arguments given as a string"
        exitval = tools.compiler.main(args.split())
        if check_errors:
            self.assertFalse(exitval, msg="tools.compiler " + args)
        return exitval

    def run_compiler_add_model_dir(self, options, model="Spring", filename="Spring.mo"):
        "Run compiler on filename in tests directory for Modelica class given by model"
        filename_path = os.path.join(MODEL_DIR, filename)
        args = options + " -m" + model + " " + filename_path
        return self.run_compiler(args)

    def test_argparse_checks_good(self):
        "Stuff that argparse should handle ok and exit"
        arg_examples = [
            "-h",
            "--version",
        ]
        # argparse does a sys.exit(0) with these
        for args in arg_examples:
            with self.assertRaises(SystemExit) as context:
                self.run_compiler(args, check_errors=False)
            self.assertEqual(context.exception.code, 0)

    def test_argparse_checks_bad(self):
        "Stuff that argparse should catch and exit"
        arg_examples = [
            "",  # Must give at least a PATH
            "-m",
            "-t",
            "-O",
            "-o",  # These expect an argument
            "-t sausage",  # Invalid target
        ]
        # argparse does a sys.exit(2) with these
        for args in arg_examples:
            with self.assertRaises(SystemExit) as context:
                self.run_compiler(args, check_errors=False)
            self.assertEqual(context.exception.code, 2)

    def test_bad_argument_combinations(self):
        "Bad option combinations caught outside argparse"
        # --target requires --model
        # This one uses argparse.error which does a sys.exit(2)
        with self.assertRaises(SystemExit) as context:
            self.run_compiler("-tsympy " + SPRING_MODEL, check_errors=False)
        self.assertEqual(context.exception.code, 2)

        # These check multiple errors before exiting
        # List of tuples of (args, number_of_expected_errors)
        bad_options = [
            # 1) Output file doesn't exist
            # 2) Modelica file doesn't exist
            ("-o bacon sausage", 2),
            # 1) No .mo files given (assume none in GENERATED_DIR)
            (GENERATED_DIR, 1),
            # 1) Modelica file does not exist
            ("sausage", 1),
            # 1) Bad target option syntax
            # 2) Give a file instead of a directory for output Path
            # 3) Give a Modelica file that does not exist
            (" ".join(("-t casadi -m Spring -O eggs -o", SPRING_MODEL, "spam")), 3),
        ]
        for args, expected_errors in bad_options:
            errors = self.run_compiler(args, check_errors=False)
            self.assertEqual(errors, expected_errors)

    def test_parse_only(self):
        "If only a Modelica file is given, then compiler tool stops after parse"
        # Parse one file
        self.run_compiler(SPRING_MODEL)
        # Parse two files
        self.run_compiler(" ".join([SPRING_MODEL, AIRCRAFT_MODEL]))
        # Parse all files in a directory and a given file
        self.run_compiler(" ".join([MODEL_DIR, SPRING_MODEL]))

    def test_flatten_only(self):
        "If model is given and target is not, then compiler tool stops after flatten"
        # Parse and flatten
        self.run_compiler("-v -m Spring " + SPRING_MODEL)
        self.run_compiler("-v -m Aircraft " + AIRCRAFT_MODEL)
        self.run_compiler("-v -m Spring -m Aircraft " + " ".join([SPRING_MODEL, AIRCRAFT_MODEL]))

    def test_casadi_options_good(self):
        "CasADi generation expected to run ok"
        # Run examples on default Spring model, give other options here
        arg_examples = [
            "-vv -Ospam=eggs",  # Flatten only, -O ignored
            "-v -tcasadi",  # -v = logging.INFO
            "-vv --target=casadi",  # -vv = logging.DEBUG
            "-t=casadi -Ocache=False -Ocodegen=False -Ocheck_balanced=True",
        ]
        for args in arg_examples:
            self.run_compiler_add_model_dir(args)

    def test_casadi_options_bad(self):
        "CasADi generation expected to fail"
        # List of tuples of (args, number_of_expected_errors)
        bad_options = [
            # 1) No .mo files given (assume none in GENERATED_DIR)
            ("-tcasadi -mspam " + GENERATED_DIR, 1),
            # 1) Given Modelica file does not exist
            ("-tcasadi -mspam eggs", 1),
            # 1) Can't infer Modelica file from given model (Spring also in Package subdir)
            ("-tcasadi -mSpring " + MODEL_DIR, 1),
        ]
        for args, expected_errors in bad_options:
            errors = self.run_compiler(args, check_errors=False)
            self.assertEqual(errors, expected_errors)

    def test_sympy_options_good(self):
        "SymPy options that should produce good output"
        # Run examples on default Spring model
        arg_examples = [
            "-t sympy -o " + GENERATED_DIR + " -vv",
        ]
        for args in arg_examples:
            self.run_compiler_add_model_dir(args)

    def test_sympy_options_bad(self):
        "SymPy generation expected to fail"
        # List of tuples of (args, number_of_expected_errors)
        bad_options = [
            # 1) No .mo files found (assume none in GENERATED_DIR)
            ("-tsympy -mspam " + GENERATED_DIR, 1),
            # 1) Given Modelica file does not exist
            ("-tsympy -mspam eggs", 1),
            # 1) Parse error
            ("-tsympy -mSpring " + BOUNCING_BALL_XML, 1),
        ]
        for args, expected_errors in bad_options:
            errors = self.run_compiler(args, check_errors=False)
            self.assertEqual(errors, expected_errors)


if __name__ == "__main__":
    unittest.main()
