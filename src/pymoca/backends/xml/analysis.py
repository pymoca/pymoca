"""
Analysis routines for simulation output.
"""
import os
from typing import Dict, List

import matplotlib as mpl

import numpy as np

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402, I100


def plot(data: Dict[str, np.array], fields: List[str] = None, *args, **kwargs):
    """
    Plot simulation data.
    :data: A dictionary of arrays.
    :fields: A list of variables you want to plot (e.g. ['x', y', 'c'])
    """
    if plt is None:
        return

    if fields is None:
        fields = ['x', 'y', 'm', 'c']
    labels = []
    lines = []
    for field in fields:
        if min(data[field].shape) > 0:
            f_lines = plt.plot(data['t'], data[field], *args, **kwargs)
            lines.extend(f_lines)
            labels.extend(data['labels'][field])
    plt.legend(lines, labels, ncol=2, loc='best')
    plt.xlabel('t, sec')
    plt.grid()
