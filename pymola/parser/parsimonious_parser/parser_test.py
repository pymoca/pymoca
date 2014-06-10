import unittest
from .parser import modelica_parser
from .visitors import ModelicaPrinter


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_empty_class(self):
        ast = modelica_parser.parse(empty_class_src)
        print ModelicaPrinter().visit(ast)

    # @unittest.skip('not working yet')
    def test_hello_world(self):
        ast = modelica_parser.parse(hello_world_src)
        print ModelicaPrinter().visit(ast)

empty_class_src = """
class test "empty"
end test;
"""

hello_world_src = """
model helloworld "A differential equation"
real a;
real b;
equation
algorithm
end helloworld;
"""
# model HelloWorld "Adifferrentialequation"
# equation
# end HelloWorld;
# Real x(start=1);
# equation
# der(x) = -x;
#
