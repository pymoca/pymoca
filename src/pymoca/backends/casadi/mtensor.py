"""
The functions and classes defined here are only supposed to be used internally
within Pymoca to work around the <= 2D only shortcoming of CasADi. In the
eventual generated model, everything should should be ca.MX.
"""

import casadi as ca

import numpy as np


def _new_mx(name, *shape):
    if len(shape) == 1 and not np.isscalar(shape[0]):
        shape = shape[0]

    assert len(shape) <= 2

    obj = ca.MX.sym(name, *shape)
    obj._modelica_shape = tuple(shape)
    return obj


class _MTensor:
    def __init__(self, name, *shape):
        self._shape = tuple(shape)
        self._mx = ca.MX.sym(name, np.prod(shape))

    @property
    def shape(self):
        return self._shape

    def __getattr__(self, attr):
        return getattr(self._mx, attr)

    def __getitem__(self, k):
        assert len(k) == len(self._shape)
        return self._mx[np.ravel_multi_index(tuple(k,), self._shape)]
