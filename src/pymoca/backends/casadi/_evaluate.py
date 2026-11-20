"""Numeric evaluation of flat Modelica expressions, for values needed at generation time."""

import math
import operator
from typing import Any, Callable, List, Optional

import numpy as np

from pymoca import ast

_BINARY = {
    "+": operator.add,
    "-": operator.sub,
    "/": operator.truediv,
    "^": operator.pow,
    ".+": operator.add,
    ".-": operator.sub,
    ".*": operator.mul,
    "./": operator.truediv,
    ".^": operator.pow,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "<>": operator.ne,
    "!=": operator.ne,
    "div": lambda a, b: math.trunc(a / b),
    "mod": lambda a, b: a - math.floor(a / b) * b,
    "rem": lambda a, b: a - math.trunc(a / b) * b,
}

_UNARY = {
    "abs": abs,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "floor": math.floor,
    "ceil": math.ceil,
    "integer": math.floor,
    "sign": lambda x: (x > 0) - (x < 0),
    "noEvent": lambda x: x,
}


class EvaluationError(ValueError):
    """An expression cannot be evaluated to a number or string at generation time."""


def _scalar(value):
    """Return a numpy scalar or 0-d array as the matching Python scalar."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(value, np.generic):
        return value.item()
    return value


class Evaluator:
    """Evaluate expressions with names looked up through *lookup*.

    Calls to functions that are not builtin go to *call*, which returns the outputs.
    """

    def __init__(
        self,
        lookup: Callable[[str], Any],
        call: Optional[Callable[[str, List[Any]], List[Any]]] = None,
    ):
        self.lookup = lookup
        self.call = call

    def __call__(self, node) -> Any:
        if isinstance(node, (bool, int, float, str, np.ndarray)) or node is None:
            return node
        method = getattr(self, "_eval_" + type(node).__name__, None)
        if method is None:
            raise EvaluationError(f"Cannot evaluate {node!r}")
        return method(node)

    def _eval_Primary(self, node: ast.Primary):
        if node.value is None:
            raise EvaluationError("Missing value")
        return node.value

    def _eval_Array(self, node: ast.Array):
        values = [self(v) for v in node.values]
        if any(isinstance(v, str) for v in values):
            return values
        return np.array(values)

    def _eval_Slice(self, node: ast.Slice):
        start, step, stop = self(node.start), self(node.step), self(node.stop)
        return np.arange(start, stop + step / 2, step)

    def _eval_IfExpression(self, node: ast.IfExpression):
        for condition, expression in zip(node.conditions, node.expressions):
            if self(condition):
                return self(expression)
        return self(node.expressions[-1])

    def _eval_ComponentRef(self, node: ast.ComponentRef):
        # A flat name has no children, unlike a dotted reference to an enumeration literal
        name = ".".join(node.to_tuple()) if node.child else node.name
        value = self.lookup(name)
        indices = [i for dims in node.indices for i in dims if i is not None]
        if indices:
            value = np.asarray(value)[tuple(self._index(i) for i in indices)]
        return _scalar(value)

    def _index(self, index):
        if isinstance(index, ast.Slice) and isinstance(index.start, ast.Primary):
            if index.start.value is None:
                return slice(None)
        value = self(index)
        if isinstance(value, np.ndarray):
            return value.astype(int) - 1
        return int(value) - 1

    def _eval_Expression(self, node: ast.Expression):
        op = str(node.operator)
        operands = node.operands
        if op == "size":
            value = np.shape(self(operands[0]))
            return np.array(value) if len(operands) == 1 else value[int(self(operands[1])) - 1]
        if op == "and":
            return bool(self(operands[0])) and bool(self(operands[1]))
        if op == "or":
            return bool(self(operands[0])) or bool(self(operands[1]))
        args = [self(o) for o in operands]
        if op == "not":
            return not args[0]
        if op in ("-", ".-") and len(args) == 1:
            return -args[0]
        if op in ("+", ".+") and len(args) == 1:
            return args[0]
        if op == "*":
            a, b = args
            both_arrays = np.ndim(a) > 0 and np.ndim(b) > 0
            return _scalar(np.dot(a, b)) if both_arrays else a * b
        if op in _BINARY and len(args) == 2:
            return _scalar(_BINARY[op](args[0], args[1]))
        if op in _UNARY and len(args) == 1:
            return _UNARY[op](args[0])
        if op in ("min", "max"):
            reduce = min if op == "min" else max
            return _scalar(reduce(np.ravel(args[0]))) if len(args) == 1 else reduce(*args)
        if op == "sum":
            return _scalar(np.sum(args[0]))
        if op == "product":
            return _scalar(np.prod(args[0]))
        if op in ("zeros", "ones"):
            return (np.zeros if op == "zeros" else np.ones)([int(a) for a in args])
        if op == "fill":
            return np.full([int(a) for a in args[1:]], args[0])
        if op in ("transpose", "vector", "scalar"):
            return {"transpose": np.transpose, "vector": np.ravel, "scalar": _scalar}[op](args[0])
        if self.call is None:
            raise EvaluationError(f"Cannot evaluate a call to {op}")
        outputs = self.call(op, args)
        return outputs[0]
