from collections import OrderedDict


# Code snippet from RTC-Tools, copyright Stichting Deltares, originally under the terms of the GPL
# version 3.  Relicensed with permission.
class AliasRelation:
    def __init__(self):
        self._aliases = {}
        self._canonical_variables_map = OrderedDict()
        self._canonical_variables = set()

    def add(self, a, b):
        # Get the canonical names and signs
        canonical_a, sign_a = self.canonical_signed(a)
        canonical_b, sign_b = self.canonical_signed(b)

        # Determine if signs need to be inverted when merging the aliased sets
        opposite_signs = (sign_a + sign_b) == 0

        # Construct aliases (a set of equivalent variables)
        aliases = self.aliases(canonical_a)
        inverted_aliases = self.aliases("-" + canonical_a)
        for v in self.aliases(canonical_b):
            if opposite_signs:
                v = self.__toggle_sign(v)
            aliases.add(v)
            inverted_aliases.add(self.__toggle_sign(v))

        for v in aliases:
            self._aliases[self.__toggle_sign(v)] = inverted_aliases
            self._aliases[v] = aliases

        # Update _canonical_variables with new canonical var and remove old ones
        self._canonical_variables.add(canonical_a)
        self._canonical_variables.discard(canonical_b)

        for v in aliases:
            self._canonical_variables_map[v] = (canonical_a, sign_a)
            self._canonical_variables_map[self.__toggle_sign(v)] = (canonical_a, -sign_a)

    def __toggle_sign(self, v):
        if self.__is_negative(v):
            return v[1:]
        else:
            return '-' + v

    @staticmethod
    def __is_negative(v):
        return True if v[0] == '-' else False

    def aliases(self, a):
        if a in self._aliases:
            return self._aliases[a]
        else:
            return {a}

    def canonical_signed(self, a):
        if a in self._canonical_variables_map:
            return self._canonical_variables_map[a]
        else:
            if self.__is_negative(a):
                return a[1:], -1
            else:
                return a, 1

    @property
    def canonical_variables(self):
        return self._canonical_variables.keys()

    def __iter__(self):
        # Note that order is not guaranteed, because we are looping over a set.
        for canonical_variable in self._canonical_variables:
            aliases = self.aliases(canonical_variable).copy()
            aliases.discard(canonical_variable)
            yield canonical_variable, aliases
