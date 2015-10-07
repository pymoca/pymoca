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
        out = os.path.join(TEST_DIR, 'BouncingBall_pymola.py')
        gen_sympy.main(argv=[src, out])
        cmd = 'python {out:s}'.format(**locals())
        proc = subprocess.Popen(cmd.split())
        proc.communicate()

    def test_bouncing_ball2(self):
        "Test if bouncing ball simulates with visitor version"
        #pylint: disable=no-self-use
        src = os.path.join(TEST_DIR, 'BouncingBall.mo')
        out = os.path.join(TEST_DIR, 'BouncingBall2_pymola.py')
        gen_sympy2.main(argv=[src, out])
        # cmd = 'python {out:s}'.format(**locals())
        # proc = subprocess.Popen(cmd.split())
        # proc.communicate()

    def test_estimator(self):
        "Test if estimator simulates"
        #pylint: disable=no-self-use
        src = os.path.join(TEST_DIR, 'Estimator.mo')
        out = os.path.join(TEST_DIR, 'Estimator_pymola.py')
        gen_sympy.main(argv=[src, out])
        cmd = 'python {out:s}'.format(**locals())
        proc = subprocess.Popen(cmd.split())
        proc.communicate()


