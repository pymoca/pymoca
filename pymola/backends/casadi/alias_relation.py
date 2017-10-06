from collections import OrderedDict, MutableSet

# From https://code.activestate.com/recipes/576694/
class OrderedSet(MutableSet):

    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def __getitem__(self, index):
        # Method added by JB
        if isinstance(index, slice):
            start, stop, stride = index.indices(len(self))
            return [self.__getitem__(i) for i in range(start, stop, stride)]
        else:
            end = self.end
            curr = end[2]
            i = 0
            while curr is not end:
                if i == index:
                    return curr[0]
                curr = curr[2]
                i += 1
            raise IndexError('set index {} out of range with length {}'.format(index, len(self)))

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)
# End snippet


# Code snippet from RTC-Tools, copyright Stichting Deltares, originally under the terms of the GPL
# version 3.  Relicensed with permission.
class AliasRelation:
    def __init__(self):
        self._aliases = {}
        self._canonical_variables = OrderedSet()

    def add(self, a, b):
        aliases = self.aliases(a)
        for v in self.aliases(b):
            aliases.add(v)
        for v in aliases:
            self._aliases[v] = aliases
        self._canonical_variables.add(aliases[0])
        for v in aliases[1:]:
            try:
                self._canonical_variables.remove(v)
            except KeyError:
                pass

    def flatten(self):
        for v in self.canonical_variables:
            vt = self.__toggle(v)
            for al in self.aliases(v):
                self.add(vt, self.__toggle(al))

        for v in self.canonical_variables:
            if v[0] == '-':
                del self._aliases[v]
                self._canonical_variables.remove(v)

    def __toggle(self, v):
        if v[0] == '-':
            return v[1:]
        else:
            return '-' + v

    def aliases(self, a):
        return self._aliases.get(a, OrderedSet([a]))

    def canonical_signed(self, a):
        if a in self._aliases:
            return self.aliases(a)[0], 1
        else:
            if a[0] == '-':
                b = a[1:]
            else:
                b = '-' + a
            if b in self._aliases:
                return self.aliases(b)[0], -1
            else:
                return self.aliases(a)[0], 1

    @property
    def canonical_variables(self):
        return self._canonical_variables

    def __iter__(self):
        return ((canonical_variable, self.aliases(canonical_variable)[1:]) for canonical_variable in self._canonical_variables)
