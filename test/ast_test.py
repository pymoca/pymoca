import unittest

from pymoca import ast


class TestASTManipulation(unittest.TestCase):
    def setUp(self):
        self.ast = ast.Tree()

    def test_add_class(self):
        c = ast.Class(name='TestClass')
        self.ast.add_class(c)

        self.assertEqual(c.parent, self.ast)
        self.assertIn(c.name, self.ast.classes)

        self.ast.remove_class(c)
        self.assertNotIn(c.name, self.ast.classes)

    def test_add_symbol(self):
        s = ast.Symbol(
            name='TestSymbol',
            type=ast.ComponentRef.from_tuple('Real')
        )
        self.ast.add_symbol(s)

        self.assertIn(s.name, self.ast.symbols)

        self.ast.remove_symbol(s)
        self.assertNotIn(s.name, self.ast.symbols)

    def test_add_equation(self):
        e = ast.Equation(
            left=ast.ComponentRef.from_tuple('a'),
            right=ast.ComponentRef.from_tuple('b')
        )
        self.ast.add_equation(e)

        self.assertIn(e, self.ast.equations)

        self.ast.remove_equation(e)
        self.assertNotIn(e, self.ast.equations)
