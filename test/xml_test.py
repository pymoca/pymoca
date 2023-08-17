#!/usr/bin/env python
"""
Test XML backend
"""
import os
import sys
import time
import unittest

# get matplotlib from analysis, since logic for plotting
# without display already handled there
import pymoca.parser as mo_parser
from pymoca.backends.xml import analysis, generator, sim_scipy
from pymoca.backends.xml import parser as xml_parser
from pymoca.backends.xml.analysis import plt


TEST_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(TEST_DIR, "models")
GENERATED_DIR = os.path.join(TEST_DIR, "generated")


class XmlTest(unittest.TestCase):
    """
    Xml tests
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @staticmethod
    def flush():
        sys.stdout.flush()
        sys.stdout.flush()
        time.sleep(0.01)

    @unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Fails with InvocationError")
    def test_noise(self):
        # compile to ModelicaXML
        with open(os.path.join(MODEL_DIR, "Noise.mo"), "r") as f:
            txt = f.read()
        ast_tree = mo_parser.parse(txt)
        model_xml = generator.generate(ast_tree, "Noise")

        # save xml model to disk
        with open(os.path.join(GENERATED_DIR, "Noise.xml"), "w") as f:
            f.write(model_xml)

        # load xml model
        model = xml_parser.parse(model_xml, verbose=False)
        print(model)

        # convert to ode
        model_ode = model.to_ode()  # type: model.HybridOde
        print(model_ode)

        # simulate
        data = sim_scipy.sim(model_ode, {"tf": 1, "dt": 0.001, "verbose": True})

        # plot
        analysis.plot(data, fields=["x", "m"])
        plt.draw()
        plt.pause(0.1)
        plt.close()

    @unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Fails with InvocationError")
    def test_simple_circuit(self):
        # compile to ModelicaXML
        with open(os.path.join(MODEL_DIR, "SimpleCircuit.mo"), "r") as f:
            txt = f.read()
        ast_tree = mo_parser.parse(txt)
        model_xml = generator.generate(ast_tree, "SimpleCircuit")

        # save xml model to disk
        with open(os.path.join(GENERATED_DIR, "SimpleCircuit.xml"), "w") as f:
            f.write(model_xml)

        # load xml model
        model = xml_parser.parse(model_xml, verbose=False)
        print(model)

        # convert to ode
        model_ode = model.to_ode()  # type: model.HybridOde
        print(model_ode)

        # simulate
        data = sim_scipy.sim(model_ode, {"tf": 1, "dt": 0.001, "verbose": True})

        # plot
        analysis.plot(data, fields=["x", "c", "m"])
        plt.draw()
        plt.pause(0.1)
        plt.close()

    @unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "Fails with InvocationError")
    def test_bouncing_ball(self):
        # generate
        with open(os.path.join(MODEL_DIR, "BouncingBall.mo"), "r") as f:
            txt = f.read()
        ast_tree = mo_parser.parse(txt)
        generator.generate(ast_tree, "BouncingBall")

        # parse
        example_file = os.path.join(MODEL_DIR, "bouncing-ball.xml")
        model = xml_parser.parse_file(example_file, verbose=False)
        print(model)

        # convert to ode
        model_ode = model.to_ode()  # type: model.HybridOde
        model_ode.prop["x"]["start"] = 1
        print(model_ode)

        # simulate
        data = sim_scipy.sim(model_ode, {"tf": 3.5, "dt": 0.01, "verbose": True})

        # plot
        analysis.plot(data, linewidth=0.5, marker=".", markersize=0.5)
        plt.draw()
        plt.pause(0.1)
        plt.close()

        # simulate in soft real-time
        do_realtime = False
        if do_realtime:
            print("\nsoft-realtime simulation")
            time_start = time.time()

            def realtime_callback(t, x, y, m, p, c):
                t_real = time.time() - time_start
                lag = t_real - t
                if abs(lag) > 0.1:
                    print("real: {:10f} > sim: {:10f}, lag: {:10f}".format(t_real, t, lag))
                elif lag < 0:
                    time.sleep(-lag)

            data = sim_scipy.sim(
                model_ode, {"tf": 3.5, "dt": 0.01, "verbose": True}, user_callback=realtime_callback
            )

        # plt.gca().set_ylim(-2, 2)
        self.flush()
