import inspect
import unittest

from pymoca import ast


class TestASTManipulation(unittest.TestCase):
    def setUp(self):
        self.ast = ast.Tree()

    def test_add_class(self):
        c = ast.Class(name="TestClass")
        self.ast.add_class(c)

        self.assertEqual(c.parent, self.ast)
        self.assertIn(c.name, self.ast.classes)

        self.ast.remove_class(c)
        self.assertNotIn(c.name, self.ast.classes)

    def test_add_symbol(self):
        s = ast.Symbol(name="TestSymbol", type=ast.ComponentRef.from_tuple("Real"))
        self.ast.add_symbol(s)

        self.assertIn(s.name, self.ast.symbols)

        self.ast.remove_symbol(s)
        self.assertNotIn(s.name, self.ast.symbols)

    def test_add_equation(self):
        e = ast.Equation(
            left=ast.ComponentRef.from_tuple("a"), right=ast.ComponentRef.from_tuple("b")
        )
        self.ast.add_equation(e)

        self.assertIn(e, self.ast.equations)

        self.ast.remove_equation(e)
        self.assertNotIn(e, self.ast.equations)

        self.ast.add_initial_equation(e)
        self.assertIn(e, self.ast.initial_equations)

        self.ast.remove_initial_equation(e)
        self.assertNotIn(e, self.ast.initial_equations)


class TestASTReprAndStr(unittest.TestCase):
    def test_all_repr_and_str_len(self):
        for attr_string in dir(ast):
            attr = getattr(ast, attr_string)
            if inspect.isclass(attr) and issubclass(attr, ast.Node):
                class_instance = attr()
                self.assertNotEqual(len(repr(class_instance)), 0)
                if isinstance(class_instance, ast.ComponentRef):
                    self.assertEqual(len(str(class_instance)), 0)
                else:
                    self.assertNotEqual(len(str(class_instance)), 0)

    def test_component_ref(self):
        cref = ast.ComponentRef.from_string("A0.B1.C2")

        cref_tuple = cref.to_tuple()
        self.assertEqual(cref_tuple[0], "A0")
        self.assertEqual(cref_tuple[1], "B1")
        self.assertEqual(cref_tuple[2], "C2")

        cref_d3 = ast.ComponentRef.from_string("D3")
        self.assertEqual(str(cref_d3), "D3")

        cref_cat = cref.from_tuple(cref_tuple + cref_d3.to_tuple())
        self.assertEqual(str(cref_cat), "A0.B1.C2.D3")
        self.assertEqual(repr(cref_cat), "'A0'['B1'['C2'['D3']]]")
