#!/usr/bin/env python

import os
import sys

import ply.lex as lex
import ply.yacc as yacc
from ply.lex import TOKEN


class Parser:
    """
    Base class for a lexer/parser that has the rules defined as methods
    """
    tokens = ()
    precedence = ()

    def __init__(self, **kw):
        self.debug = kw.get('debug', 0)
        self.classes = {}
        try:
            modname = os.path.split(
                os.path.splitext(__file__)[0])[1] \
                + "_" + self.__class__.__name__
        except:
            modname = "parser"+"_"+self.__class__.__name__
        self.debugfile = modname + ".dbg"
        self.tabmodule = modname + "_" + "parsetab"
        # print self.debugfile, self.tabmodule

        # Build the lexer and parser
        lex.lex(module=self, debug=self.debug)
        yacc.yacc(module=self,
                  debug=self.debug,
                  debugfile=self.debugfile,
                  tabmodule=self.tabmodule)

    def run(self):
        while 1:
            try:
                s = raw_input('modelica > ')
            except EOFError:
                break
            if not s:
                continue
            yacc.parse(s)
            print self.classes

    def dump(self, obj, nested_level=0, output=sys.stdout):
        spacing = '   '
        if type(obj) == dict:
            print >> output, '%s{' % ((nested_level) * spacing)
            for k, v in obj.items():
                if hasattr(v, '__iter__'):
                    print >> output, '%s%s:' % \
                        ((nested_level + 1) * spacing, k)
                    self.dump(v, nested_level + 1, output)
                else:
                    print >> output, '%s%s: %s' % \
                        ((nested_level + 1) * spacing, k, v)
            print >> output, '%s}' % (nested_level * spacing)
        elif type(obj) == list:
            print >> output, '%s[' % ((nested_level) * spacing)
            for v in obj:
                if hasattr(v, '__iter__'):
                    self.dump(v, nested_level + 1, output)
                else:
                    print >> output, '%s%s' % ((nested_level + 1) * spacing, v)
            print >> output, '%s]' % ((nested_level) * spacing)
        else:
            print >> output, '%s%s' % (nested_level * spacing, obj)

    def parse(self, s):
        res = yacc.parse(s)
        self.dump(res)


class ModelicaParser(Parser):

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Tokens
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ----------------------------------------------------------
    # Reserved Keywords
    # ----------------------------------------------------------
    keywords = (
        'algorithm', 'and', 'annotation', 'assert', 'block',
        'break', 'class', 'connect', 'connector', 'constant',
        'constrainedby', 'der', 'discrete', 'each', 'else',
        'elseif', 'elsewhen', 'encapsulated', 'end', 'enumeration',
        'equation', 'expandable', 'extends', 'external', 'false',
        'final', 'flow', 'for', 'function', 'if', 'import', 'impure',
        'in', 'initial', 'inner', 'input', 'loop', 'model', 'not',
        'operator', 'or', 'outer', 'output', 'package', 'parameter',
        'partial', 'protected', 'public', 'pure', 'record',
        'redeclare', 'return', 'stream', 'then',
        'true', 'type', 'when', 'while', 'within')

    reserved = {key: key.upper() for key in keywords}

    # ----------------------------------------------------------
    # Symbols
    # ----------------------------------------------------------
    symbols = {
        '\;': 'SEMI',
        '\=': 'EQUALS',
        '\+': 'PLUS',
        '\,': 'COMMA',
        '\(': 'LPAREN',
        '\)': 'RPAREN',
    }
    # create token handling for symbols
    for key in symbols.keys():
        exec("t_{:s} = '{:s}'".format(symbols[key], key))

    # ----------------------------------------------------------
    # Token Processing
    # ----------------------------------------------------------
    tokens = [
        'IDENT',
        'UNSIGNED_NUMBER',
        'STRING'
    ] + reserved.values() + symbols.values()

    t_ignore = ' \t'

    re_dict = {
        'digit': r'[0-9]',
        'nondigit': r'[_a-zA-Z]',
        's_escape': r'([\\][\']|[\\]["])',
        's_char': r'[^\\"]',
    }
    re_process_list = [
        ('q_char', r'{nondigit}|{digit}|\!|\#'),
        ('q_ident', r"'({q_char}|{s_escape})+'"),
        ('ident', r'({nondigit} ({digit}|{nondigit})*) | {q_ident}'),
        ('string', r'\"({s_char}|{s_escape})*\"'),
        ('unsigned_integer', r'{digit}+'),
        ('unsigned_number', r'{unsigned_integer}([\.]{unsigned_integer}?)?'),
    ]
    for key, val in re_process_list:
        re_dict[key] = val.format(**re_dict)

    @TOKEN(re_dict['ident'])
    def t_IDENT(self, t):
        t.type = self.reserved.get(t.value, 'IDENT')
        return t

    @TOKEN(re_dict['unsigned_number'])
    def t_UNSIGNED_NUMBER(self, t):
        return t

    @TOKEN(re_dict['string'])
    def t_STRING(self, t):
        t.value = t.value[1:-1]  # remove double quotes
        return t

    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")
        return None

    def t_error(self, t):
        print "Illegal character '%s'" % t.value[0]
        t.lexer.skip(1)
        return None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parsing Rules
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # EBNF to BNF
    # {E} -> X = empty | X E
    # [E] -> X = empty | E
    # (E) -> X = E

    precedence = ()

    start = 'stored_definition'

    # ----------------------------------------------------------
    # Parsing Helper Functions
    # ----------------------------------------------------------

    @staticmethod
    def list_extend(p, i1=0, i2=2):
        p[0] = p[i1]
        if p[0] is None:
            p[0] = []
        if len(p) > 2:
            p[0].extend([p[i2]])

    @staticmethod
    def store_as_list(p):
        p[0] = p[1:]

    @staticmethod
    def print_p(self, p):
        for i in range(len(p)):
            print(i, p[i])

    # ----------------------------------------------------------
    # Parsing Helper Rules
    # ----------------------------------------------------------

    def p_empty(self, p):
        'empty : '
        pass

    def p_error(self, p):
        if p is None:
            print("Syntax error at root definition.")
        else:
            print("Syntax error at line {:d}, {:s}".
                  format(p.lineno, p.value))

    # ----------------------------------------------------------
    # B.2.1 Stored Definition - Within
    # ----------------------------------------------------------

    # stored_definition:
    # [ within [ name ] ";" ]
    # { [ final ] class_definition ";" }

    def p_stored_definition(self, p):
        'stored_definition : within_opt class_definitions'
        p[0] = {'within': p[1], 'classes': p[2]}

    def p_within_opt(self, p):
        '''within_opt : WITHIN name_opt SEMI
            | empty'''
        p[0] = {'name': p[1]}

    def p_name_opt(self, p):
        '''name_opt : name
            | empty'''
        p[0] = False if p[1] is None else True

    def p_class_definitions(self, p):
        '''class_definitions : class_definitions final_opt class_definition SEMI
            | empty'''
        if len(p) > 2:  # not empty
            class_defs = p[1]
            final_opt = p[2]
            class_def = p[3]
            if class_defs is None:
                class_defs = {}
            if class_def is not None:
                class_def['final'] = final_opt
                class_defs[class_def['name']] = class_def
            p[0] = class_defs

    def p_final_opt(self, p):
        '''final_opt : FINAL
            | empty'''
        p[0] = False if p[1] is None else True

    # ----------------------------------------------------------
    # B.2.2 Class Definition
    # ----------------------------------------------------------

    # class_definition :
    # [ encapsulated ] class_prefixes
    # class_specifier

    def p_class_definition(self, p):
        'class_definition : encapsulated_opt '\
            'class_prefixes class_specifier'
        encapsulated = p[1]
        prefixes = p[2]
        specifier = p[3]
        p[0] = {
            'encapsulated': encapsulated,
            'name': specifier['name'],
            'comment': specifier['comment'],
            'composition': specifier['composition'],
        }
        for key in prefixes.keys():
            p[0][key] = prefixes[key]

    # class_prefixes :
    # [ partial ]
    # ( class | model | [ operator ] record | block |
    # [ expandable ] connector | type |
    # package | [ ( pure | impure ) ] [ operator ]
    # function | operator )

    def p_class_prefixes(self, p):
        'class_prefixes : partial_opt class_type_opt'
        p[0] = {'partial': p[1], 'type': p[2]}

    def p_partial_opt(self, p):
        '''partial_opt : PARTIAL
            | empty'''
        p[0] = False if p[1] is None else True

    def p_class_type_opt(self, p):
        '''class_type_opt : CLASS
            | MODEL
            | RECORD
            | OPERATOR RECORD
            | BLOCK
            | CONNECTOR
            | EXPANDABLE CONNECTOR
            | TYPE
            | PACKAGE
            | PURE FUNCTION
            | IMPURE FUNCTION
            | PURE OPERATOR FUNCTION
            | IMPURE OPERATOR FUNCTION
            | OPERATOR'''
        p[0] = ''.join(p[1:len(p)])

    # class_specifier :
    # IDENT string_comment composition end IDENT
    # | IDENT "=" base_prefix name [ array_subscripts ]
    # [ class_modification ] comment
    # | IDENT "=" enumeration "(" ( [enum_list] | ":" ) ")" comment
    # | IDENT "=" der "(" name "," IDENT { "," IDENT } ")" comment
    # | extends IDENT [ class_modification ] string_comment composition
    # end IDENT

    # TODO
    def p_class_specifier(self, p):
        'class_specifier : IDENT string_comment_opt '\
            'composition END IDENT'
        name = p[1]
        comment = p[2]
        composition = p[3]
        name_end = p[5]
        if name != name_end:
            raise IOError(
                "Syntax error at line {:d}, "
                "class names don't match at beginning and end: "
                "{:s} {:s}".
                format(p.lineno(6), name, name_end))
        p[0] = {
            'name': name,
            'comment': comment,
            'composition': composition,
        }

    # base_prefix :
    # type_prefix

    # def p_base_prefix(self, p):
    #     'base_prefix : type_prefix'

    # enum_list
    # : enumeration_literal { "," enumeration_literal}
    # enumeration_literal : IDENT comment

    # composition :
    # element_list
    # { public element_list |
    # protected element_list |
    # equation_section |
    # algorithm_section
    # }
    # [ external [ language_specification ]
    # [ external_function_call ] [ annotation ] ";" ]
    # [ annotation ";" ]

    def p_composition(self, p):
        '''composition : element_list composition_list'''
        self.list_extend(p)

    def p_composition_or(self, p):
        '''composition_or : PUBLIC element_list
            | PROTECTED element_list
            | equation_section
            | algorithm_section'''
        self.store_as_list(p)

    def p_composition_list(self, p):
        '''composition_list : composition_list composition_or
            | empty'''
        self.list_extend(p)

    # language_specification :
    # STRING

    # def p_language_specification(self, p):
    #     'language_specification : STRING'
    #     self.store_as_list(p)

    # external_function_call :
    # [ component_reference "=" ]
    # IDENT "(" [ expression_list ] ")"

    # element_list :
    # { element ";" }

    def p_element_list(self, p):
        '''element_list : element_list element SEMI
            | empty'''
        self.store_as_list(p)

    # element :
    # import_clause |
    # extends_clause |
    # [ redeclare ]
    # [ final ]
    # [ inner ] [ outer ]
    # ( ( class_definition | component_clause) |
    # replaceable ( class_definition | component_clause)247
    # [constraining_clause comment])

    def p_element(self, p):
        'element : component_clause'
        p[0] = p[1]

    # import_clause :
    # import ( IDENT "=" name | name ["." ( "*" |
    # "{" import_list "}" ) ] ) comment
    # import_list :
    # IDENT [ "," import_list ]

    # def p_external_function_call(self, p):
    #     'external_function_call : component_reference_opt '\
    #         ' IDENT "(" expression_list ")"'
    #     self.store_as_list(p)

    # ----------------------------------------------------------
    # B.2.3 Extends
    # ----------------------------------------------------------

    # extends_clause :
    # extends name [ class_modification ] [annotation]

    # constraining_clause :
    # constrainedby name [ class_modification ]

    # ----------------------------------------------------------
    # B.2.4 Component Clause
    # ----------------------------------------------------------

    # component_clause:
    # type_prefix type_specifier [ array_subscripts ] component_list

    # type_prefix :
    # [ flow | stream ]
    # [ discrete | parameter | constant ] [ input | output ]
    def p_type_prefix(self, p):
        'type_prefix : type_prefix_1 type_prefix_2 type_prefix_3'
        p[0] = [p[1], p[2], p[3]]

    def p_type_prefix1(self, p):
        '''type_prefix_1 : FLOW
            | STREAM
            | empty'''
        p[0] = p[1]

    def p_type_prefix2(self, p):
        '''type_prefix_2 : DISCRETE
            | PARAMETER
            | CONSTANT
            | empty'''
        p[0] = p[1]

    def p_type_prefix3(self, p):
        '''type_prefix_3 : INPUT
            | OUTPUT
            | empty'''
        p[0] = p[1]

    # type_specifier :
    # name

    def p_type_specifier(self, p):
        'type_specifier : name'
        p[0] = p[1]

    # component_list :
    # component_declaration { "," component_declaration }

    # component_declaration :
    # declaration [ condition_attribute ] comment

    # condition_attribute:
    # if expression

    # declaration :
    # IDENT [ array_subscripts ] [ modification ]

    # ----------------------------------------------------------
    # B.2.5 Modification
    # ----------------------------------------------------------

    # modification :
    # class_modification [ "=" expression ]
    # | "=" expression
    # | ":=" expression

    # class_modification :
    # "(" [ argument_list ] ")"

    # def p_class_modification(self, p):
    #     'class_modification : LPAREN argument_list_opt RPAREN'
    #     p[0] = p[2]

    # argument_list :
    # argument { "," argument }

    # def p_arguemnt_list(self, p):
    #     'argument_list : argument arguement_list_part'
    #     p[0] = p[1:]
    # def p_argument_list_part(self, p):
    #     '''argument_list_part : argument_list_part COMMA argument
    #         | empty'''
    #     p[0] = p[1]

    # argument :
    # element_modification_or_replaceable
    # | element_redeclaration

    # TODO
    # def p_argument(self, p):
    #     'argument : IDENT'

    # element_modification_or_replaceable:
    # [ each ] [ final ] ( element_modification | element_replaceable)

    # element_modification :
    # name [ modification ] string_comment

    # element_redeclaration :
    # redeclare [ each ] [ final ]
    # ( ( short_class_definition | component_clause1) | element_replaceable )

    # element_replaceable:
    # replaceable ( short_class_definition | component_clause1)
    # [constraining_clause]

    # component_clause1 :
    # type_prefix type_specifier component_declaration1

    # component_declaration1 :
    # declaration comment

    # short_class_definition :
    # class_prefixes IDENT "="
    # ( base_prefix name [ array_subscripts ]
    # [ class_modification ] comment |
    # enumeration "(" ( [enum_list] | ":" ) ")" comment )

    # ----------------------------------------------------------
    # B.2.6 Equations
    # ----------------------------------------------------------

    # equation_section :
    # [ initial ] equation { equation ";" }

    # TODO
    def p_equation_section(self, p):
        'equation_section : EQUATION'
        self.store_as_list(p)

    # algorithm_section :
    # [ initial ] algorithm { statement ";" }

    # TODO
    def p_algorithm_section(self, p):
        'algorithm_section : ALGORITHM'
        self.store_as_list(p)

    # equation :
    # ( simple_expression "=" expression
    # | if_equation
    # | for_equation
    # | connect_clause
    # | when_equation
    # | name function_call_args )
    # comment

    # statement :
    # ( component_reference ( ":=" expression | function_call_args )
    # | "(" output_expression_list ")" ":="
    # component_reference function_call_args
    # | break
    # | return
    # | if_statement
    # | for_statement
    # | while_statement
    # | when_statement )
    # comment

    # if_equation :
    # if expression then
    # { equation ";" }
    # { elseif expression then
    # { equation ";" }
    # }
    # [ else
    # { equation ";" }
    # ]
    # end if

    # if_statement :
    # if expression then
    # { statement ";" }
    # { elseif expression then
    # { statement ";" }
    # }
    # [ else
    # { statement ";" }
    # ]
    # end if

    # for_equation :
    # for for_indices loop
    # { equation ";" }
    # end for

    # for_statement :
    # for for_indices loop
    # { statement ";" }
    # end for

    # for_indices :
    # for_index {"," for_index}

    # for_index:
    # IDENT [ in expression ]

    # while_statement :
    # while expression loop
    # { statement ";" }
    # end while

    # when_equation :
    # when expression then
    # { equation ";" }
    # { elsewhen expression then
    # { equation ";" } }
    # end when

    # when_statement :
    # when expression then
    # { statement ";" }
    # { elsewhen expression then
    # { statement ";" } }
    # end when

    # connect_clause :
    # connect "(" component_reference "," component_reference ")"

    def p_component_clause(self, p):
        'component_clause : type_prefix type_specifier '\
            'array_subscripts_opt component_list'
        p[0] = {
            'prefix': p[1],
            'specifier': p[2],
            'array_subscripts': p[3],
            'component_list': p[4],
        }

    # TODO
    def p_array_subscripts_opt(self, p):
        '''array_subscripts_opt : empty'''
        p[0] = p[1]

    def p_component_list(self, p):
        '''component_list : component
            | component_list COMMA component
            | empty
        '''
        if len(p) > 2:
            if p[1] is None:
                p[0] = []
            else:
                p[0] = p[1]
            p[0].extend([p[3]])
        elif p[1] is not None:
            p[0] = [p[1]]

    def p_component(self, p):
        '''component : IDENT EQUALS UNSIGNED_NUMBER'''
        p[0] = [p[1], p[2], p[3]]

    def p_encapsulated_opt(self, p):
        '''encapsulated_opt : ENCAPSULATED
            | empty'''
        p[0] = False if p[1] is None else True

    # ----------------------------------------------------------
    # B.2.7 Expressions
    # ----------------------------------------------------------

    # expression :
    # simple_expression
    # | if expression then expression { elseif expression then expression }
    # else expression

    # simple_expression :
    # logical_expression [ ":" logical_expression [ ":" logical_expression ] ]

    # logical_expression :
    # logical_term { or logical_term }

    # logical_term :
    # logical_factor { and logical_factor }

    # logical_factor :
    # [ not ] relation

    # relation :
    # arithmetic_expression [ rel_op arithmetic_expression ]

    # rel_op :
    # "<" | "<=" | ">" | ">=" | "==" | "<>"

    # arithmetic_expression :
    # [ add_op ] term { add_op term }

    # add_op :
    # "+" | "-" | ".+" | ".-"

    # term :
    # factor { mul_op factor }

    # mul_op :
    # "*" | "/" | ".*" | "./"

    # factor :
    # primary [ ("^" | ".^") primary ]

    # primary :
    # UNSIGNED_NUMBER
    # | STRING
    # | false
    # | true
    # | ( name | der | initial ) function_call_args
    # | component_reference
    # | "(" output_expression_list ")"
    # | "[" expression_list { ";" expression_list } "]"
    # | "{" function_arguments "}"
    # | end

    # name :
    # [ "." ] IDENT { "." IDENT }

    # TODO
    def p_name(self, p):
        '''name : IDENT'''
        self.store_as_list(p)

    # TODO
    # def p_names(self, p):
    #     '''names : names '.' IDENT
    #         | empty'''
    #     self.store_as_list(p)

    # component_reference :
    # [ "." ] IDENT [ array_subscripts ] { "." IDENT [ array_subscripts ] }

    # def p_component_reference_opt(self, p):
    #     '''component_refefrence_opt : component_reference EQUALS
    #         | empty'''
    #     self.store_as_list(p)

    # TODO
    # def p_component_reference(self, p):
    #     '''component_reference : IDENT'''
    #     self.store_as_list(p)

    # function_call_args :
    # "(" [ function_arguments ] ")"

    # function_arguments :
    # function_argument [ "," function_arguments | for for_indices ]
    # | named_arguments

    # named_arguments: named_argument [ "," named_arguments ]

    # named_argument: IDENT "=" function_argument

    # function_argument :
    # function name "(" [ named_arguments ] ")" | expression
    # output_expression_list:
    # [ expression ] { "," [ expression ] }

    # expression_list :
    # expression { "," expression }

    # array_subscripts :
    # "[" subscript { "," subscript } "]"

    # subscript :
    # ":" | expression

    # comment :
    # string_comment [ annotation ]

    # string_comment :
    # [ STRING { "+" STRING } ]

    def p_string_comment_part(self, p):
        '''string_comment_part : string_comment_part PLUS STRING
            | empty'''
        if (len(p) > 2):
            if p[1] is None:
                p[0] = p[3]
            else:
                p[0] = ''.join([p[1], p[3]])

    def p_string_comment(self, p):
        'string_comment : STRING string_comment_part'
        p[0] = ''.join([p[1], p[2]])

    def p_string_comment_opt(self, p):
        '''string_comment_opt : string_comment
            | empty'''
        p[0] = p[1]

    # annotation :
    # annotation class_modification
    # def p_annotation(self, p):
    #     'annotation : STRING'
    #     p[0] = p[1]

if __name__ == '__main__':
    parser = ModelicaParser()
    parser.parse('''
    class hello1 "hello" + " bye"
    end hello1;
    class hello2 "hello world" + " example"
        flow discrete input Real a=1, b=2;
    public
        Real c=3;
        Real d=3;
    equation
    algorithm
    end hello2;
    ''')
