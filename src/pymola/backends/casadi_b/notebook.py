from io import StringIO
import sys
from IPython.display import Latex, Image
import casadi as ca
from casadi.tools.graph import dotgraph
import re


def latex_print(expr, sub=False):
    if expr.shape[0] == 0:
        return r'\begin{bmatrix}\end{bmatrix}'
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    expr.print_sparse()
    sys.stdout = old_stdout
    x = mystdout.getvalue().split('\n')

    shape_split = x[0].split(':')[1].split(',')[0].strip().split('-')
    m = int(shape_split[0])
    n = int(shape_split[2])
    e = {}
    b = {
        'cos': r'\cos',
        'sin': r'\sin',
        'tan': r'\tan',
        'power': r'\^'
    }
    for l in ['alpha', 'beta', 'gamma', 'Gamma',
            'delta', 'Delta', 'epsilon', 'zeta', 'eta', 'theta',
            'Theta', 'iota', 'kappa', 'lambda', 'Lambda', 'mu',
            'nu', 'xi', 'pi', 'rho', 'sigma', 'tau', 'upsilon',
            'phi', 'Phi', 'chi', 'psi', 'Psi', 'omega', 'Omega']:
        b[l] = '\\' + l
    s = '|'.join(sorted(b.keys(), key=len, reverse=True))
    b_pattern = re.compile(s)
    t = {}
    for i in range(1, len(x)):
        line = x[i].strip()
        if line[0] == "@":
            split = line.split("=")
            term = split[1].replace(',', '')
            term = b_pattern.sub(lambda x: b[x.group()], term)
            e[split[0]] = term
        elif line[0] == "(":
            split = line.split("->")
            indices = split[0].strip().replace('(', '').replace(')', '').split(',')
            i = int(indices[0].strip())
            j = int(indices[1].strip())
            t[(i, j)] = str(split[1])
    s = r"\begin{bmatrix} "
    for i in range(m):
        row = []
        for j in range(n):
            term = "0";
            if (i, j) in t.keys():
                term = t[(i, j)]
            term = b_pattern.sub(lambda x: b[x.group()], term)
            if sub:
                for k in sorted(e.keys(), key=len, reverse=True):
                    term = term.replace(k, e[k])
            row += [r"{:s}".format(term)]
        s += " & ".join(row) + r" \\"
    s += r" \end{bmatrix}"
    if not sub:
        s += r"\begin{align}"
        for k in e.keys():
            s += r"{:s}: && {:s} \\".format(k, e[k])
        s += r"\end{align}"
    return s

def latex(x, sub=False):
    return Latex(latex_print(x, sub))

def graph(*args, **kwargs):
    pydot = dotgraph(*args, **kwargs)
    return Image(pydot.create_png())

#  vim: set et fenc=utf-8 ff=unix sts=0 sw=4 ts=4 : 
