import unittest
import json
from grako.exceptions import FailedParse

from ..parser import ModelicaParser
from ..semantics import ModelicaSemantics

basic_test_code = '''
model test
    import Analog=Modelica.Electrical.Analog;
    Real a;
equation
algorithm
end test;
'''


class Test(unittest.TestCase):

    def setUp(self):
        self.parser = ModelicaParser()
        pass

    def basic_test(self):
        rule_name = 'stored_definition'
        semantics = ModelicaSemantics('Modelica.ebnf')
        try:
            ast = self.parser.parse(
                basic_test_code,
                rule_name=rule_name,
                semantics=semantics,
                trace=False)
            print(json.dumps(ast, indent=2))
        except FailedParse as e:
            print(e)
            try:
                ast = self.parser.parse(
                    basic_test_code,
                    rule_name=rule_name,
                    semantics=semantics,
                    trace=True)
            except:
                pass
