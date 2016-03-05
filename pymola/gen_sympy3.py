#!/usr/bin/env python
"""
Modelica compiler.
"""
from __future__ import print_function
import sys
import antlr4
import antlr4.Parser
import argparse
import jinja2
from collections import OrderedDict
from .generated.ModelicaLexer import ModelicaLexer
from .generated.ModelicaParser import ModelicaParser
from .generated.ModelicaListener import ModelicaListener

#pylint: disable=invalid-name, no-self-use, missing-docstring, unused-variable, protected-access
#pylint: disable=too-many-public-methods

class SympyPrinter(ModelicaListener):

    def __init__(self, parser, trace):
        """
        Constructor
        """
        self._val_dict = OrderedDict()
        self.result = None
        self._parser = parser
        self._trace = trace
        self.indent = "            "

    @staticmethod
    def print_indented(ldr, s):
        if s is not None:
            for line in str(s).split('\n'):
                print(ldr, line)

    def setValue(self, ctx, val):
        """
        Sets tree values.
        """
        self._val_dict[ctx] = val

    def getValue(self, ctx):
        """
        Gets tree values.
        """
        return self._val_dict[ctx]

    def enterEveryRule(self, ctx):
        self.setValue(ctx, None)
        if self._trace:
            ldr = " "*(ctx.depth()-1)
            rule = self._parser.ruleNames[ctx.getRuleIndex()]

            print(ldr, rule + "{")
            in_str = ctx.getText()
            if in_str > 0:
                print(ldr, "==============================")
                print(ldr, "INPUT\n")
                self.print_indented(ldr, ctx.getText())
                print(ldr, "==============================\n")

    def visitTerminal(self, node):
        pass

    def visitErrorNode(self, node):
        pass

    def exitEveryRule(self, ctx):
        rule = self._parser.ruleNames[ctx.getRuleIndex()]
        if self._trace:
            ldr = " "*ctx.depth()
            lines = None
            try:
                lines = self.getValue(ctx)
            except KeyError as e:
                pass

            if lines is not None:
                print(ldr, "==============================")
                print(ldr, "OUTPUT\n")
                self.print_indented(ldr, lines)
                print(ldr, "==============================\n")

            print(ldr + '} //' + rule + '\n')
        # if self.getValue(ctx) is None:
            # raise RuntimeError(
                    # "no value set for {:s}:\ninput:\n{:s}".format(
                        # rule, ctx.getText()))

#=========================================================
#  B.2.1 Stored Definition - Within
#=========================================================

# B.2.1.1 ------------------------------------------------
# stored_definition :
#     ('within' name? ';')?
#     ('final'? class_definition ';')*
#     ;
    def exitStored_definition(self, ctx):
        # TODO within/ final
        result = ''
        for cls in ctx.class_definition():
            result += self.getValue(cls)
        self.result = result

#=========================================================
#  B.2.2 Class Definition
#=========================================================

# B.2.2.1 ------------------------------------------------
# class_definition :
#     'encapsulated'? class_prefixes
#     class_specifier
#     ;
    def exitClass_definition(self, ctx):
        # TODO encapsulated/ class_prefixes
        self.setValue(ctx, self.getValue(ctx.class_specifier()))

# B.2.2.2 ------------------------------------------------
# class_prefixes : 
#     'partial'?
#     (
#         'class'
#         | 'model'
#         | 'operator'? 'record'
#         | 'block'
#         | 'expandable'? 'connector'
#         | 'type'
#         | 'package'
#         | ('pure' | 'impure')? 'operator'? 'function'
#         | 'operator'
#     )
#     ;
    def exitClass_prefixes(self, ctx):
        self.setValue(ctx, [c.getText() for c in ctx.getChildren()])

# B.2.2.3 ------------------------------------------------
# class_specifier :
#     IDENT string_comment composition 'end' IDENT                    # class_spec_comp
#     | IDENT '=' base_prefix name array_subscripts?
#         class_modification? comment                                 # class_spec_base
#     | IDENT '=' 'enumeration' '(' (enum_list? | ':') ')' comment    # class_spec_enum
#     | IDENT '=' 'der' '(' name ',' IDENT (',' IDENT )* ')' comment  # class_spec_der
#     | 'extends' IDENT class_modification? string_comment            # class_spec_extends
#         composition 'end' IDENT
#     ;

    def exitClass_spec_comp(self, ctx):
        if str(ctx.IDENT()[0]) != str(ctx.IDENT()[1]):
            raise SyntaxError('class names must match {:s}, {:s}'.format(ctx.IDENT()[0], ctx.IDENT()[1]))
        name = ctx.IDENT()[0].getText()
        comment = self.getValue(ctx.string_comment())
        composition = self.getValue(ctx.composition())
        result = """
class {name:s} :
    \"\"\"
    {comment:s}
    \"\"\"

    def __init__(self):
        self.data = {composition:s}
        pass

""".format(**locals())
        self.setValue(ctx, result)

    def exitClass_spec_base(self, ctx):
        # TODO
        raise NotImplementedError("")

    def exitClass_spec_enum(self, ctx):
        # TODO
        raise NotImplementedError("")

    def exitClass_spec_der(self, ctx):
        # TODO
        raise NotImplementedError("")

    def exitClass_spec_extends(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.2.4 ------------------------------------------------
# base_prefix :
#     type_prefix
#     ;
    def exitBase_prefix(self, ctx):
        # TODO
        raise NotImplementedError("")


# B.2.2.5 ------------------------------------------------
# enum_list :
#     enumeration_literal (',' enumeration_literal)*
#     ;
    def exitEnum_list(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.2.6 ------------------------------------------------
# enumeration_literal :
#     IDENT comment
#     ;
    def exitEnumeration_literal(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.2.7 ------------------------------------------------
# composition :
#     element_list
#     (
#         'public' epub=element_list
#         | 'protected' epro=element_list
#         | equation_section
#         | algorithm_section
#     )*
#     ( 'external' language_specification?
#         external_function_call?
#         ext_annotation=annotation? ':')?
#     (comp_annotation=annotation ';')?
#     ;
    def exitComposition(self, ctx):
        elist = self.getValue(ctx.element_list()[0])
        elist_public = []
        elist_protected = []
        eq_section = []
        alg_section = []
        language_spec = None
        ext_func_call = None
        ext_annotation = None
        comp_annotation = None
        if ctx.epub is not None:
            elist_public = [self.getValue(e) for e in ctx.epub()]
        if ctx.epro is not None:
            elist_protected = [self.getValue(e) for e in ctx.epro()]
        eq_section = [self.getValue(e) for e in ctx.equation_section()]
        alg_section = [self.getValue(e) for e in ctx.algorithm_section()]
        if ctx.language_specification() is not None:
            language_spec = self.getValue(ctx.language_specification()),
        if ctx.external_function_call() is not None:
            ext_func_call = self.getValue(ctx.external_function_call()),
        if ctx.ext_annotation is not None:
            ext_annotation =  self.getValue(ctx.ext_annotation()),
        if ctx.comp_annotation is not None:
            comp_annotation =  self.getValue(ctx.comp())
        self.setValue(ctx, {
            'elist': elist,
            'elist_public': elist_public,
            'elist_protected': elist_protected,
            'eq_section': eq_section,
            'alg_section': alg_section,
            'language_spec': language_spec,
            'ext_func_call': ext_func_call,
            'ext_annotation': ext_annotation,
            'comp_annotation': comp_annotation})

# B.2.2.8 ------------------------------------------------
# language_specification :
#     STRING
#     ;
    def exitLanguage_specification(self, ctx):
        self.setValue(ctx, ctx.STRING().getText()[1:-1])

# B.2.2.9 ------------------------------------------------
# external_function_call :
#     (component_reference '=')?
#     IDENT '(' expression_list? ')'
#     ;
    def exitExternal_functioni_call(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.2.10 -----------------------------------------------
# element_list : 
#     (element ';')*
#     ;
    def exitElement_list(self, ctx):
        self.setValue(ctx, [self.getValue(e) for e in ctx.element()])

# B.2.2.11 -----------------------------------------------
# element :
#     import_clause
#     | extends_clause
#     | redeclare='redeclare'? final='final'?
#         inner='inner'? outer='outer'?
#         ((classdef=class_definition | comp=component_clause)
#          | 'replaceable' (rclassdef=class_definition | rcomp=component_clause)
#             (constraining_clause comment)?
#         );
    def exitElement(self, ctx):
        if ctx.import_clause() is not None:
            self.setValue(ctx, self.getValue(ctx.imports_clause()))
        elif ctx.extends_clause() is not None:
            self.setValue(ctx, self.getValue(ctx.extends_clause()))
        else:
            classdef = None
            comp = None
            rclassdef = None
            rcomp = None
            exitConstraining_clause = None
            comment = None
            constraining_clause = None
            if ctx.rclassdef != None:
                rclassdef = self.getValue(ctx.rclass)
            if ctx.rcomp != None:
                rcomp = self.getValue(ctx.rcomp)
            if ctx.classdef != None:
                classdef = self.getValue(ctx.classdef)
            if ctx.comp != None:
                classdef = self.getValue(ctx.comp)
            if ctx.constraining_clause() != None:
                constraining_clause = self.getValue(ctx.constraining_clause())
            if ctx.comment() != None:
                comment = self.getValue(ctx.comment())
            self.setValue(ctx, {
                'redeclare': ctx.redeclare != None,
                'final': ctx.final != None,
                'inner': ctx.inner != None,
                'outer': ctx.outer != None,
                'classdef': classdef,
                'rclassdef': rclassdef,
                'rcomp': rcomp,
                'comp': comp,
                'constraining_clause': constraining_clause,
                'comment': comment
            })

# B.2.2.12 -----------------------------------------------
# import_clause :
#     'import' ( IDENT '=' name
#         | name ('.' ( '*' | '{' import_list '}' ) )? ) comment
#     ;
    def exitImport_clause(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.2.13 -----------------------------------------------
# import_list : 
#     IDENT (',' import_list)*
#     ;
    def exitImport_list(self, ctx):
        # TODO
        raise NotImplementedError("")

#=========================================================
# B.2.3 Extends
#=========================================================

# B.2.3.1 ------------------------------------------------
# extends_clause :
#     'extends' name class_modification? annotation?
#     ;
    def exitExtends_clause(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.3.2 ------------------------------------------------
# constraining_clause:
#     'constrainedby' name class_modification?
#     ;
    def exitConstraining_clause(self, ctx):
        # TODO
        raise NotImplementedError("")

#=========================================================
# B.2.4 Component Clause
#=========================================================

# B.2.4.1 ------------------------------------------------
# component_clause :
#     type_prefix type_specifier array_subscripts? component_list
#     ;
    def exitComponent_clause(self, ctx):
        if ctx.array_subscripts() is not None:
            array_subscripts = self.getValue(ctx.array_subscripts())
        else:
            array_subscripts = None
        self.setValue(ctx, {
            'type_prefix': self.getValue(ctx.type_prefix()),
            'type_specifier': self.getValue(ctx.type_specifier()),
            'array_subscripts': array_subscripts,
            'component_list': self.getValue(ctx.component_list())
            })

# B.2.4.2 ------------------------------------------------
# type_prefix :
#     ('flow' | 'stream')?
#     ('discrete' | 'parameter' | 'constant')?
#     ('input' | 'output')?
#     ;
    def exitType_prefix(self, ctx):
        self.setValue(ctx, [c.getText() for c in ctx.getChildren()])

# B.2.4.3 ------------------------------------------------
# type_specifier:
#     name
#     ;
    def exitType_specifier(self, ctx):
        self.setValue(ctx, self.getValue(ctx.name()))

# B.2.4.4 ------------------------------------------------
# component_list:
#     component_declaration ( ',' component_declaration)*
#     ;
    def exitComponent_list(self, ctx):
        self.setValue(ctx,
            [self.getValue(c) for c in ctx.component_declaration()])

# B.2.4.5 ------------------------------------------------
# component_declaration :
#     declaration condition_attribute? comment
#     ;
    def exitComponent_declaration(self, ctx):
        # TODO condition_attribute, comment
        self.setValue(ctx, self.getValue(ctx.declaration()))

# B.2.4.6 ------------------------------------------------
# condition_attribute :
#     'if' expression
#     ;
    def exitCondition_attribute(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.4.7 ------------------------------------------------
# declaration :
#     IDENT array_subscripts? modification?
#     ;
    def exitDeclaration(self, ctx):
        # TODO array subscripts and modification
        self.setValue(ctx, '{:s}'.format(ctx.IDENT().getText()))

#=========================================================
# B.2.5 Modification
#=========================================================

# B.2.5.1 ------------------------------------------------
# modification :
#     class_modification ('=' expression)?    # modification_class
#     | '=' expression                        # modification_assignment
#     | ':=' expression                       # modification_assignment2
#     ;

# class_modification ('=' expression)?
    def exitModification_class(self, ctx):
        # TODO
        raise NotImplementedError("")

# '=' expression
    def exitModification_assignment(self, ctx):
        # TODO
        raise NotImplementedError("")

# ':=' expression
    def exitModification_assignment2(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.5.2 ------------------------------------------------
# class_modification :
#     '(' argument_list? ')'
#     ;
    def exitClass_modification(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.5.3 ------------------------------------------------
# argument_list :
#     argument (',' argument)*
#     ;
    def exitArgument_list(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.5.4 ------------------------------------------------
# argument :
#     element_modification_or_replaceable
#     | element_redeclaration
#     ;
    def exitArgument(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.5.5 ------------------------------------------------
# element_modification_or_replaceable:
#     'each'?
#     'final'?
#     (element_modification | element_replaceable)
#     ;
    def exitElement_modification_or_replaceable(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.5.6 ------------------------------------------------
# element_modification :
#     name modification? string_comment
#     ;
    def exitElement_modification(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.5.7 ------------------------------------------------
# element_redeclaration :
#     'redeclare'
#     'each'?
#     'final'?
#     ( (short_class_definition | component_clause1)
#       | element_replaceable)
#     ;
    def exitElement_redeclaration(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.5.8 ------------------------------------------------
# element_replaceable:
#     'replaceable'
#     (short_class_definition | component_clause1)
#     constraining_clause?
#     ;
    def exitElement_replaceable(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.5.9 ------------------------------------------------
# component_clause1 :
#     type_prefix type_specifier component_declaration1
#     ;
    def exitComponent_clause1(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.5.10 -----------------------------------------------
# component_declaration1 :
#     declaration comment
#     ;
    def exitComponent_declaration1(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.5.11 -----------------------------------------------
# short_class_definition :
#     class_prefixes IDENT '='
#     ( base_prefix name array_subscripts?
#         class_modification? comment
#         | 'enumeration' '(' (enum_list? | ':') ')' comment)
#     ;
    def exitShort_class_definition(self, ctx):
        # TODO
        raise NotImplementedError("")

#=========================================================
# B.2.6 Equations
#=========================================================
# 
# B.2.6.1 ------------------------------------------------
# equation_section :
#     init='initial'? 'equation' (equation ';')*
#     ;
    def exitEquation_section(self, ctx):
        eqs = [self.getValue(e) for e in ctx.equation()]
        self.setValue(ctx, {
            'init': ctx.init != None,
            'eqs': eqs})

# B.2.6.2 ------------------------------------------------
# algorithm_section :
#     'initial'? 'algorithm' (statement ';')*
#     ;
    def exitAlgorithm_section(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.6.3 ------------------------------------------------
# equation_options :
#     simple_expression '=' expression    # equation_simple
#     | if_equation                       # equation_if
#     | for_equation                      # equation_for
#     | connect_clause                    # equation_connect_clause
#     | when_equation                     # equation_when
#     | name function_call_args           # equation_function
#     ;

# simple_expression '=' expression
    def exitEquation_simple(self, ctx):
        self.setValue(ctx,
            '{:s} - {:s}'.format(
                self.getValue(ctx.simple_expression()),
                self.getValue(ctx.expression())))

# if_equation
    def exitEquation_if(self, ctx):
        # TODO
        raise NotImplementedError("")

# for_equation
    def exitEquation_for(self, ctx):
        # TODO
        raise NotImplementedError("")

# connect_clause
    def exitEquation_connect_clause(self, ctx):
        # TODO
        raise NotImplementedError("")

# when_equation
    def exitEquation_when(self, ctx):
        # TODO
        raise NotImplementedError("")

# name function_call_args
    def exitEquation_function(self, ctx):
        self.setValue(ctx, self.getValue(ctx.function_call_args()))
        # TODO
        raise NotImplementedError("")

# B.2.6.4 ------------------------------------------------
# equation :
#     equation_options
#     comment
#     ;
    def exitEquation(self, ctx):
        # TODO comment
        self.setValue(ctx, self.getValue(ctx.equation_options()))

# B.2.6.5 ------------------------------------------------
# statement_options :
#     component_reference (':=' expression | function_call_args)  # statement_component_reference
#     | '(' output_expression_list ')' ':=' 
#         component_reference function_call_args                  # statement_component_function
#     | 'break'           # statement_break
#     | 'return'          # statement_return
#     | if_statement      # statement_if
#     | for_statement     # statement_for
#     | while_statement   # statement_while
#     | when_statement    # statement_when
#     ;

# component_reference (':=' expression | function_call_aargs)
    def exitStatement_component_reference(self, ctx):
        # TODO
        raise NotImplementedError("")

# '(' output_expression_list ')' ':=' component_reference function_call_args 
    def exitStatement_component_function(self, ctx):
        # TODO
        raise NotImplementedError("")

# 'break'
    def exitStatement_break(self, ctx):
        # TODO
        raise NotImplementedError("")

# 'return'
    def exitStatement_return(self, ctx):
        # TODO
        raise NotImplementedError("")

# if_statement
    def exitStatement_if(self, ctx):
        # TODO
        raise NotImplementedError("")

# for_statement
    def exitStatement_for(self, ctx):
        # TODO
        raise NotImplementedError("")

# while_statement
    def exitStatement_while(self, ctx):
        # TODO
        raise NotImplementedError("")

# when_statement
    def exitStatement_when(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.6.6 ------------------------------------------------
# statement :
#     statement_options
#     comment
#     ;
    def exitStatement(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.6.7 ------------------------------------------------
# if_equation :
#     'if' expression 'then'
#         (equation ';')*
#     ('elseif' expression 'then'
#         (equation ';')*
#     )*
#     ('else'
#         (equation ';')*
#     )?
#     'end' 'if'
#     ;
    def exitIf_equation(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.6.8 ------------------------------------------------
# if_statement :
#     'if' expression 'then'
#         (statement ';')*
#     ('elseif' expression 'then'
#         (statement ';')*
#     )*
#     ('else'
#         (statement ';')*
#     )?
#     'end' 'if'
#     ;
    def exitIf_statement(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.6.9 ------------------------------------------------
# for_equation :
#     'for' for_indices 'loop'
#         (equation ';')*
#     'end' 'for'
#     ;
    def exitFor_equation(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.6.10 -----------------------------------------------
# for_statement :
#     'for' for_indices 'loop'
#         (statement ';')*
#     'end' 'for'
#     ;
    def exitFor_statement(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.6.11 -----------------------------------------------
# for_indices :
#     for_index (',' for_index)*
#     ;
    def exitFor_indices(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.6.12 -----------------------------------------------
# for_index :
#     IDENT ('in' expression)?
#     ;
    def exitFor_index(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.6.13 -----------------------------------------------
# while_statement:
#     'while' expression 'loop'
#         (statement ';')*
#     'end' 'while'
#     ;
    def exitWhile_statement(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.6.14 -----------------------------------------------
# when_equation:
#     'when' expression 'then'
#         (equation ';')*
#     ('elsewhen' expression 'then'
#         (equation ';')*
#     )*
#     'end' 'when'
#     ;
    def exitWhen_equation(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.6.15 -----------------------------------------------
# when_statement:
#     'when' expression 'then'
#         (statement ';')*
#     ('elsewhen' expression 'then'
#         (statement ';')*
#     )*
#     'end' 'when'
#     ;
    def exitWhen_statement(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.6.16 -----------------------------------------------
# connect_clause :
#     'connect' '(' component_reference ',' component_reference ')'
#     ;
    def exitConnect_clause(self, ctx):
        # TODO
        raise NotImplementedError("")

#=========================================================
# B.2.7 Expressions
#=========================================================

# B.2.7.1 ------------------------------------------------
# expression :
#     simple_expression                           # expression_simple
#     | 'if' expression 'then' expression         
#     ( 'elseif' expression 'then' expression)*
#     'else' expression                           # expression_if
#     ;

# simple_expression
    def exitExpression_simple(self, ctx):
        self.setValue(ctx, self.getValue(ctx.simple_expression()))

# 'if' expression 'then' expression         
#     ( 'elseif' expression 'then' expression)*
#     'else' expression
    def exitExpression_if(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.7.2 ------------------------------------------------
# simple_expression :
#     expr (':' expr
#         (':' expr)?)?
#     ;
    def exitSimple_expression(self, ctx):
        # TODO rest of expressions
        exprs = [self.getValue(e) for e in ctx.expr()]
        self.setValue(ctx, exprs)


# B.2.7.3 ------------------------------------------------
# expr :
#     '-' expr                                                # expr_neg
#     | primary op=('^' | '.^') primary                       # expr_exp 
#     | expr op=('*' | '/' | '.*' | './') expr                # expr_mul
#     | expr  op=('+' | '-' | '.+' | '.-') expr               # expr_add
#     | expr  op=('<' | '<=' | '>' | '>=' | '==' | '<>') expr # expr_rel
#     | 'not' expr                                            # expr_not    
#     | expr  'and' expr                                      # expr_and
#     | expr  'or' expr                                       # expr_or
#     | primary                                               # expr_primary
#     ;

# '-' expr
    def exitExpr_neg(self, ctx):
        self.setValue(ctx, '-{:s}'.format(
            self.getValue(ctx.expr())))

# primary op=('^' | '.^') primary
    def exitExpr_exp(self, ctx):
        p = ctx.primary()
        self.setValue(ctx, '{:s} {:s} {:s}'.format(
            self.getValue(p[0]),
            op.getText(),
            self.getValue(p[1])))

# expr op=('*' | '/' | '.*' | './') expr
    def exitExpr_mul(self, ctx):
        e = ctx.expr()
        self.setValue(ctx, '{:s} {:s} {:s}'.format(
            self.getValue(e[0]),
            op.getText(),
            self.getValue(e[1])))

# expr  op=('+' | '-' | '.+' | '.-') expr
    def exitExpr_add(self, ctx):
        e = ctx.expr()
        self.setValue(ctx, '{:s} {:s} {:s}'.format(
            self.getValue(e[0]),
            op.getText(),
            self.getValue(e[1])))

# expr  op=('<' | '<=' | '>' | '>=' | '==' | '<>') expr
    def exitExpr_rel(self, ctx):
        e = ctx.expr()
        self.setValue(ctx, '{:s} {:s} {:s}'.format(
            self.getValue(e[0]),
            op.getText(),
            self.getValue(e[1])))

# 'not' expr
    def exitExpr_not(self, ctx):
        self.setValue(ctx, '!{:s}'.format(
            ctx.expr()))

# expr  'and' expr
    def exitExpr_or(self, ctx):
        e = ctx.expr()
        self.setValue(ctx, '{:s} and {:s}'.format(
            self.getValue(e[0]),
            op.getText(),
            self.getValue(e[1])))

# expr  'or' expr
    def exitExpr_or(self, ctx):
        e = ctx.expr()
        self.setValue(ctx, '{:s} or {:s}'.format(
            self.getValue(e[0]),
            op.getText(),
            self.getValue(e[1])))

# primary
    def exitExpr_primary(self, ctx):
        self.setValue(ctx, self.getValue(ctx.primary()))

# B.2.7.4 ------------------------------------------------
# primary :
#     UNSIGNED_NUMBER                                     # primary_unsigned_number
#     | STRING                                            # primary_string
#     | 'false'                                           # primary_false
#     | 'true'                                            # primary_true
#     | name function_call_args                           # primary_function
#     | 'der' function_call_args                          # primary_derivative
#     | 'initial' function_call_args                      # primary_initial
#     | component_reference                               # primary_component_reference
#     | '(' output_expression_list ')'                    # primary_output_expression_list
#     | '[' expression_list (';' expression_list)* ']'    # primary_expression_list
#     | '{' function_arguments '}'                        # primary_function_arguments
#     | 'end'                                             # primary_end
#     ;

# UNSIGNED_NUMBER
    def exitPrimary_unsigned_number(self, ctx):
        self.setValue(ctx, ctx.getText())

# STRING
    def exitPrimary_string(self, ctx):
        self.setValue(ctx, ctx.getText()[1:-1])

# 'false'
    def exitPrimary_false(self, ctx):
        self.setValue(ctx, 'False')

# 'true'
    def exitPrimary_true(self, ctx):
        self.setValue(ctx, 'True')

# name function_call_args
    def exitPrimary_funtion(self, ctx):
        # TODO
        raise NotImplementedError("")

# 'der' function_call_args
    def exitPrimary_derivative(self, ctx):
        name = ctx.function_call_args().function_arguments().function_argument()[0].getText()
        self.setValue(ctx, '{:s}.diff(self.t)'.format(name))

# 'initial' function_call_args
    def exitPrimary_initial(self, ctx):
        # TODO
        raise NotImplementedError("")

# component_reference
    def exitPrimary_component_reference(self, ctx):
        # TODO
        self.setValue(ctx,
                self.getValue(ctx.component_reference()))

# '[' expression_list (';' expression_list)* ']'
    def exitPrimary_output_expression_list(self, ctx):
        # TODO
        raise NotImplementedError("")

# '{' function_arguments '}'
    def exitPrimary_function_arguments(self, ctx):
        # TODO
        raise NotImplementedError("")

# 'end'
    def exitPrimary_end(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.7.5 ------------------------------------------------
# name :
#     '.'? IDENT ('.' IDENT)*
#     ;
    def exitName(self, ctx):
        self.setValue(ctx, ctx.getText())

# B.2.7.6 ------------------------------------------------
# component_reference :
#     '.'? IDENT array_subscripts? ('.' IDENT array_subscripts?)*
#     ;
    def exitComponent_reference(self, ctx):
        # TODO array_subscripts and other IDENTs
        self.setValue(ctx, [c.getText() for c in ctx.IDENT()])

# B.2.7.7 ------------------------------------------------
# function_call_args :
#     '(' function_arguments? ')'
#     ;
    def exitFunction_call_args(self, ctx):
        args = ctx.function_arguments()
        if  args is not None:
            self.setValue(ctx, self.getValue(args))
        else:
            self.setValue(ctx, "")

# B.2.7.8 ------------------------------------------------
# function_arguments :
#     function_argument (',' function_argument | 'for' for_indices)*
#     | named_arguments
#     ;
    def exitFunction_arguments(self, ctx):
        # TODO for_indices and named_arguments
        self.setValue(ctx,
            [self.getValue(arg) for arg in ctx.function_argument()])

# B.2.7.9 ------------------------------------------------
# named_arguments : named_argument (',' named_argument)*
#     ;
    def exitNamed_arguments(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.7.10 -----------------------------------------------
# named_argument : IDENT '=' function_argument
#     ;
    def exitNamed_argument(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.7.11 -----------------------------------------------
# function_argument :
#     'function' name '(' named_arguments? ')'    # argument_function
#     | expression                                # argument_expression
#     ;
    def exitFunction_argument(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.7.12 -----------------------------------------------
# output_expression_list :
#     expression? (',' expression)*
#     ;
    def exitOutput_expression_list(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.7.13 -----------------------------------------------
# expression_list :
#     expression (',' expression)*
#     ;
    def exitExpression_list(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.7.14 -----------------------------------------------
# array_subscripts :
#     '[' subscript (',' subscript)* ']'
#     ;
    def exitArray_subscripts(self, ctx):
        # TODO
        raise NotImplementedError("")

# B.2.7.15 -----------------------------------------------
# subscript :
#     ':' | expression
#     ;
    def exitSubscript(ctx):
        # TODO
        raise NotImplementedError("")

# B.2.7.16 -----------------------------------------------
# comment :
#     string_comment annotation?
#     ;
    def exitComment(self, ctx):
        self.setValue(ctx, ctx.getText())

# B.2.7.17 -----------------------------------------------
# string_comment :
#     (STRING ('+' STRING)*)?
#     ;
    def exitString_comment(self, ctx):
        result = ''
        for s in ctx.STRING():
            # remove quotes
            result += s.getText()[1:-1]
        self.setValue(ctx, result)

# B.2.7.18 -----------------------------------------------
# annotation :
#     'annotation' class_modification
#     ;
    def exitAnnotation(self, ctx):
        # TODO
        raise NotImplementedError("")

#=========================================================
# Generator
#=========================================================

def generate(modelica_model, trace=False):
    "The modelica model"
    input_stream = antlr4.InputStream(modelica_model)
    lexer = ModelicaLexer(input_stream)
    stream = antlr4.CommonTokenStream(lexer)
    parser = ModelicaParser(stream)
    tree = parser.stored_definition()
    # print(tree.toStringTree(recog=parser))
    sympyPrinter = SympyPrinter(parser, trace)
    walker = antlr4.ParseTreeWalker()
    walker.walk(sympyPrinter, tree)
    return sympyPrinter.result

#=========================================================
# Commande Line Interface
#=========================================================

def main(argv):
    #pylint: disable=unused-argument
    "The main function"
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('out')
    parser.add_argument('-t', '--trace', action='store_true')
    parser.set_defaults(trace=False)
    args = parser.parse_args(argv)
    with open(args.src, 'r') as f:
        modelica_model = f.read()
    sympy_model = generate(modelica_model, trace=args.trace)

    with open(args.out, 'w') as f:
        f.write(sympy_model)

if __name__ == '__main__':
    main(sys.argv[1:])

# vi:ts=4 sw=4 et nowrap:
