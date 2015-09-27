"""
Unit testing.
"""
import unittest
from .. import gen_sympy
import os
import subprocess

TEST_DIR = os.path.dirname(os.path.realpath(__file__))

class Test(unittest.TestCase):
    "Testing"

    def test_bouncing_ball(self):
        "Test if bouncing ball simulates"
        #pylint: disable=no-self-use
        gen_sympy.main(argv=[os.path.join(TEST_DIR, "BouncingBall.mo")])
        script = os.path.join(TEST_DIR, 'BouncingBall.mo.py')
        cmd = 'python {script:s}'.format(**locals())
        subprocess.Popen(cmd.split())

