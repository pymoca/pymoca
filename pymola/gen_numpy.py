from __future__ import print_function, absolute_import, division, print_function, unicode_literals
from . import tree

import os
import sys
import copy

import numpy as np


class NumpyGenerator(tree.TreeListener):

    def exitArray(self, tree):
        self.src[tree] = [self.src[e] for e in tree.values]

    def exitPrimary(self, tree):
        self.src[tree] = float(tree.value)
