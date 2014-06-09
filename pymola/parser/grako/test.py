import unittest
import pprint

from .parser import ModelicaParser


class Test(unittest.TestCase):

    def setUp(self):
        self.parser = ModelicaParser()
        pass

    def test_basic(self):
        res = self.parser.parse('''
            class test "hello world \a \b \f \r"
                flow a;
            end test;

            class test2
            end test2;
        ''', rule_name='stored_definition')
        pprint.pprint(res)
