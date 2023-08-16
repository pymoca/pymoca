import unittest

from pymoca.backends.casadi.alias_relation import AliasRelation


class TestAliasRelation(unittest.TestCase):
    def test_double_alias(self):
        alias_relation = AliasRelation()

        alias_relation.add("a", "b")
        alias_relation.add("b", "c")
        alias_relation.add("c", "a")

        self.assertSetEqual(
            alias_relation.canonical_variables,
            {
                "a",
            },
        )
        self.assertSetEqual(alias_relation.aliases("a"), {"a", "b", "c"})
