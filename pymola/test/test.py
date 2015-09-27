"""
Unit testing.
"""
import unittest
from .. import gen_sympy
from .. import gen_sympy2
import os
import subprocess

TEST_DIR = os.path.dirname(os.path.realpath(__file__))

class Test(unittest.TestCase):
    "Testing"

    def test_bouncing_ball(self):
        "Test if bouncing ball simulates"
        #pylint: disable=no-self-use
        src = os.path.join(TEST_DIR, 'BouncingBall.mo')
        out = os.path.join(TEST_DIR, 'BouncingBall.pymola.py')
        gen_sympy.main(argv=[src, out])
        cmd = 'python {out:s}'.format(**locals())
        proc = subprocess.Popen(cmd.split())
        proc.communicate()

    def test_bouncing_ball2(self):
        "Test if bouncing ball simulates"
        #pylint: disable=no-self-use
        src = os.path.join(TEST_DIR, 'BouncingBall.mo')
        out = os.path.join(TEST_DIR, 'BouncingBall2.pymola.py')
        gen_sympy2.main(argv=[src, out])
        # cmd = 'python {out:s}'.format(**locals())
        # proc = subprocess.Popen(cmd.split())
        # proc.communicate()


