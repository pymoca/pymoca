import unittest
from ..parser import ModelicaParser


class Test(unittest.TestCase):

    def setUp(self):
        self.parser = ModelicaParser()

    def class_test(self):
        self.parser.parse('''
        class hello1 "hello" + " bye"
        end hello1;
        class hello2 "hello world" + " example"
            flow discrete input Real a=1, b=2;
        public
            Real c=3;
            Real d=3;
        equation
        algorithm
        end hello2;
        ''')
