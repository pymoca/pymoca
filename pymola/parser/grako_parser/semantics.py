from sets import Set
from grako.exceptions import FailedSemantics

keywords = Set([
    'algorithm', 'and', 'annotation', 'assert',
    'block', 'break', 'class', 'connect', 'connector',
    'constant', 'constrainedby', 'der', 'discrete',
    'each', 'else', 'elseif', 'elsewhen',
    'encapsulated', 'end', 'enumeration', 'equation',
    'expandable', 'extends', 'external', 'false',
    'final', 'flow', 'for', 'function', 'if', 'import',
    'impure', 'in', 'initial', 'inner', 'input',
    'loop', 'model', 'not', 'operator', 'or',
    'outer', 'output', 'package', 'parameter',
    'partial', 'protected', 'public', 'pure',
    'record', 'redeclare', 'return', 'stream', 'then',
    'true', 'type', 'when', 'while', 'within'])


class ModelicaSemantics(object):

    def __init__(self, name):
        self.__name = name

    def IDENT(self, ast):
        if ast in keywords:
            raise FailedSemantics(
                '{:s} is a keywaord'.format(str(ast)))
        return ast
