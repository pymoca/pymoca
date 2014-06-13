import unittest
import json
import os

from grako.exceptions import FailedParse, FailedSemantics
from grako.buffering import Buffer

from .. import parser
from .. import semantics
from .. import model

path = os.path.dirname(os.path.realpath(__file__))


class Test(unittest.TestCase):

    def setUp(self):
        self.parser = parser.ModelicaParser()
        pass

    def bouncing_ball_test(self):
        COMMENTS_RE = r'/\*(?:|\n)*?\*/|//[^\n]*?\n'
        rule_name = 'stored_definition'
        sem = semantics.ModelicaSemantics('Modelica.ebnf')
        f = open(os.path.join(path, 'BouncingBall.mo'))
        buffer = Buffer(f.read(), comments_re=COMMENTS_RE,
                        trace=True)
        try:
            ast = self.parser.parse(
                buffer,
                rule_name=rule_name,
                semantics=sem,
                trace=False)
            print(json.dumps(ast, indent=2))
        except FailedSemantics as e:
            print(e)
        except FailedParse as e:
            print(e)
