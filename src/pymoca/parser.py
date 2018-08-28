#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import antlr4
import antlr4.Parser
from typing import Dict
from collections import deque, OrderedDict
import copy

from . import ast
# noinspection PyUnresolvedReferences,PyUnresolvedReferences
from .generated.ModelicaLexer import ModelicaLexer
# noinspection PyUnresolvedReferences,PyUnresolvedReferences
from .generated.ModelicaListener import ModelicaListener
# noinspection PyUnresolvedReferences,PyUnresolvedReferences
from .generated.ModelicaParser import ModelicaParser


# TODO
#  - Named function arguments (note that either all have to be named, or none)
#  - Make sure slice indices (eventually) evaluate to integers

class ModelicaFile:
    def __init__(self, **kwargs):
        self.within = []  # type: List[ComponentRef]
        self.classes = OrderedDict()  # type: OrderedDict[str, Class]
        super().__init__(**kwargs)


# noinspection PyPep8Naming
class ASTListener(ModelicaListener):
    def __init__(self):
        self.file_node = None  # type: ModelicaFile
        self.ast = {}  # type: Dict[ast.Node]
        self.ast_result = None  # type: ast.Node
        self.class_nodes = deque([ast.Class()])  # type: deque[ast.Class]
        self.comp_clause = None  # type: ast.ComponentClause
        self.eq_sect = None  # type: ast.EquationSection
        self.alg_sect = None  # type: ast.AlgorithmSection
        self.symbol_node = None  # type: ast.Symbol
        self.eq_comment = None  # type: str
        self.sym_count = 0  # type: int
        self.in_extends_clause = False

    @property
    def class_node(self):
        return self.class_nodes[-1]

    # FILE ===========================================================

    def enterStored_definition(self, ctx: ModelicaParser.Stored_definitionContext):

        file_node = ModelicaFile()
        self.ast[ctx] = file_node
        self.file_node = file_node

    def exitStored_definition(self, ctx: ModelicaParser.Stored_definitionContext):
        within = []
        if ctx.component_reference() is not None:
            within = [self.ast[ctx.component_reference()]]
        self.file_node.within = within

        for class_node in [self.ast[e] for e in ctx.stored_definition_class()]:
            self.ast[ctx].classes[class_node.name] = class_node
        self.ast_result = self.ast[ctx]

    # CLASS ===========================================================

    def exitStored_definition_class(self, ctx: ModelicaParser.Stored_definition_classContext):
        class_node = self.ast[ctx.class_definition()]
        class_node.final = ctx.FINAL() is not None
        self.ast[ctx] = class_node

    def enterClass_definition(self, ctx: ModelicaParser.Class_definitionContext):
        class_node = ast.Class()
        class_node.encapsulated = ctx.ENCAPSULATED() is not None
        class_node.partial = ctx.class_prefixes().PARTIAL() is not None
        class_node.type = ctx.class_prefixes().class_type().getText()

        self.class_nodes.append(class_node)

        self.ast[ctx] = class_node

    def exitClass_definition(self, ctx: ModelicaParser.Class_definitionContext):
        class_node = self.class_nodes.pop()
        self.class_node.classes[class_node.name] = class_node

    def exitShort_class_definition(self, ctx):
        self.ast[ctx] = ast.ShortClassDefinition(name=ctx.IDENT().getText(),
                                                 type=ctx.class_prefixes().class_type().getText(),
                                                 component=self.ast[ctx.component_reference()])

    def exitClass_spec_comp(self, ctx: ModelicaParser.Class_spec_compContext):
        class_node = self.class_node
        class_node.name = ctx.IDENT()[0].getText()
        class_node.comment = self.ast[ctx.string_comment()]

    def exitClass_spec_base(self, ctx: ModelicaParser.Class_spec_baseContext):
        class_node = self.class_node
        class_node.name = ctx.IDENT().getText()
        class_node.comment = self.ast[ctx.comment()]

        if ctx.class_modification() is not None:
            class_modification = self.ast[ctx.class_modification()]
        else:
            class_modification = ast.ClassModification()
        extends_clause = ast.ExtendsClause(component=self.ast[ctx.component_reference()],
                                          class_modification=class_modification)
        class_node.extends.append(extends_clause)

    def exitComposition(self, ctx: ModelicaParser.CompositionContext):
        for clause in self.ast[ctx.epriv]:
            if isinstance(clause, ast.ComponentClause):
                for symbol in clause.symbol_list:
                    symbol.visibility = ast.Visibility.PRIVATE
            elif isinstance(clause, ast.ExtendsClause):
                clause.visibility = ast.Visibility.PRIVATE

        if ctx.epub is not None:
            for clause in self.ast[ctx.epub]:
                if isinstance(clause, ast.ComponentClause):
                    for symbol in clause.symbol_list:
                        symbol.visibility = ast.Visibility.PUBLIC
                elif isinstance(clause, ast.ExtendsClause):
                    clause.visibility = ast.Visibility.PUBLIC

        if ctx.epro is not None:
            for clause in self.ast[ctx.epro]:
                if isinstance(clause, ast.ComponentClause):
                    for symbol in clause.symbol_list:
                        symbol.visibility = ast.Visibility.PROTECTED
                elif isinstance(clause, ast.ExtendsClause):
                    clause.visibility = ast.Visibility.PROTECTED

        for eqlist in [self.ast[e] for e in ctx.equation_section()]:
            if eqlist is not None:
                if eqlist.initial:
                    self.class_node.initial_equations += eqlist.equations
                else:
                    self.class_node.equations += eqlist.equations

        for alglist in [self.ast[e] for e in ctx.algorithm_section()]:
            if alglist is not None:
                if alglist.initial:
                    self.class_node.initial_statements += alglist.statements
                else:
                    self.class_node.statements += alglist.statements

        if ctx.comp_annotation is not None:
            self.class_node.annotation = self.ast[ctx.comp_annotation]

    def exitArgument(self, ctx: ModelicaParser.ArgumentContext):
        argument = ast.ClassModificationArgument()
        if ctx.element_modification_or_replaceable() is not None:
            argument.value = self.ast[ctx.element_modification_or_replaceable()]
            argument.redeclare = False
        else:
            argument.value = self.ast[ctx.element_redeclaration()]
            argument.redeclare = True
        self.ast[ctx] = argument

    def exitArgument_list(self, ctx: ModelicaParser.Argument_listContext):
        self.ast[ctx] = [self.ast[a] for a in ctx.argument()]

    def exitClass_modification(self, ctx: ModelicaParser.Class_modificationContext):
        arguments = []
        if ctx.argument_list() is not None:
            arguments = self.ast[ctx.argument_list()]
        self.ast[ctx] = ast.ClassModification(arguments=arguments)

    def enterEquation_section(self, ctx: ModelicaParser.Equation_sectionContext):
        eq_sect = ast.EquationSection(
            initial=ctx.INITIAL() is not None
        )
        self.ast[ctx] = eq_sect
        self.eq_sect = eq_sect

    def exitEquation_section(self, ctx: ModelicaParser.Equation_sectionContext):
        eq_sect = self.ast[ctx]
        if eq_sect.initial:
            eq_sect.equations.extend(self.ast[ctx.equation_block()])
        else:
            eq_sect.equations.extend(self.ast[ctx.equation_block()])

    def exitEquation_block(self, ctx: ModelicaParser.Equation_blockContext):
        self.ast[ctx] = [self.ast[e] for e in ctx.equation()]

    def exitStatement_block(self, ctx):
        self.ast[ctx] = [self.ast[e] for e in ctx.statement()]

    def enterAlgorithm_section(self, ctx: ModelicaParser.Algorithm_sectionContext):
        alg_sect = ast.AlgorithmSection(
            initial=ctx.INITIAL() is not None
        )
        self.ast[ctx] = alg_sect
        self.alg_sect = alg_sect

    def exitAlgorithm_section(self, ctx: ModelicaParser.Algorithm_sectionContext):
        alg_sect = self.ast[ctx]
        if alg_sect.initial:
            alg_sect.statements.extend(self.ast[ctx.statement_block()])
        else:
            alg_sect.statements.extend(self.ast[ctx.statement_block()])

    # EQUATION ===========================================================

    def enterEquation(self, ctx: ModelicaParser.EquationContext):
        pass

    def exitEquation(self, ctx):
        self.ast[ctx] = self.ast[ctx.equation_options()]
        try:
            self.ast[ctx].comment = self.ast[ctx.comment()]
        except AttributeError:
            pass

    def exitEquation_simple(self, ctx: ModelicaParser.Equation_simpleContext):
        self.ast[ctx] = ast.Equation(
            left=self.ast[ctx.simple_expression()],
            right=self.ast[ctx.expression()])

    def exitEquation_if(self, ctx: ModelicaParser.Equation_ifContext):
        self.ast[ctx] = self.ast[ctx.if_equation()]

    def exitEquation_for(self, ctx: ModelicaParser.Equation_forContext):
        self.ast[ctx] = self.ast[ctx.for_equation()]

    def exitEquation_connect_clause(self, ctx: ModelicaParser.Equation_connect_clauseContext):
        self.ast[ctx] = self.ast[ctx.connect_clause()]

    def exitArgument_expression(self, ctx: ModelicaParser.Argument_expressionContext):
        self.ast[ctx] = self.ast[ctx.expression()]

    def exitIf_equation(self, ctx: ModelicaParser.If_equationContext):
        blocks = [self.ast[b] for b in ctx.blocks]
        conditions = [self.ast[c] for c in ctx.conditions]
        if len(conditions) == len(blocks) - 1:
            conditions.append(True)
        self.ast[ctx] = ast.IfEquation(
            conditions=conditions,
            blocks=blocks)

    def exitWhen_equation(self, ctx: ModelicaParser.When_equationContext):
        blocks = [self.ast[b] for b in ctx.blocks]
        conditions = [self.ast[c] for c in ctx.conditions]
        if len(conditions) == len(blocks) - 1:
            conditions.append(True)
        self.ast[ctx] = ast.WhenEquation(
            conditions=conditions,
            blocks=blocks)

    def exitFor_equation(self, ctx: ModelicaParser.For_equationContext):
        self.ast[ctx] = ast.ForEquation(
            indices=self.ast[ctx.for_indices()],
            equations =self.ast[ctx.equation_block()])

    def exitConnect_clause(self, ctx: ModelicaParser.Connect_clauseContext):
        self.ast[ctx] = ast.ConnectClause(
            left=self.ast[ctx.component_reference()[0]],
            right=self.ast[ctx.component_reference()[1]])

    # STATEMENT ==========================================================

    # TODO:
    # - Missing statement types:
    #   + Function calls (inside current function).
    #   + Break
    #   + Return
    #   + While
    #   + When? (also in equation missing)

    def enterStatement(self, ctx: ModelicaParser.StatementContext):
        pass

    def exitStatement(self, ctx: ModelicaParser.StatementContext):
        self.ast[ctx] = self.ast[ctx.statement_options()]
        try:
            self.ast[ctx].comment = self.ast[ctx.comment()]
        except AttributeError:
            pass

    def exitStatement_component_reference(self, ctx: ModelicaParser.Statement_component_referenceContext):
        self.ast[ctx] = ast.AssignmentStatement(
            left=[self.ast[ctx.component_reference()]],
            right=self.ast[ctx.expression()])

    def exitStatement_component_function(self, ctx: ModelicaParser.Statement_component_functionContext):
        all_comp_refs = [self.ast[x] for x in ctx.component_reference()]

        right = ast.Expression(
            operator=all_comp_refs[-1],
            operands=[self.ast[x.expression()]
                      for x in ctx.function_call_args().function_arguments().function_argument()]
        )

        self.ast[ctx] = ast.AssignmentStatement(
            left=all_comp_refs[:-1],
            right=right)

    def exitStatement_if(self, ctx: ModelicaParser.Statement_ifContext):
        self.ast[ctx] = self.ast[ctx.if_statement()]

    def exitStatement_for(self, ctx: ModelicaParser.Statement_forContext):
        self.ast[ctx] = self.ast[ctx.for_statement()]

    def exitStatement_when(self, ctx: ModelicaParser.Equation_whenContext):
        self.ast[ctx] = self.ast[ctx.when_equation()]

    def exitIf_statement(self, ctx: ModelicaParser.If_statementContext):
        blocks = [self.ast[b] for b in ctx.blocks]
        conditions = [self.ast[c] for c in ctx.conditions]
        if len(conditions) == len(blocks) - 1:
            conditions.append(True)
        self.ast[ctx] = ast.IfStatement(
            conditions=conditions,
            blocks=blocks)

    def exitWhen_statement(self, ctx: ModelicaParser.When_statementContext):
        blocks = [self.ast[b] for b in ctx.blocks]
        conditions = [self.ast[c] for c in ctx.conditions]
        if len(conditions) == len(blocks) - 1:
            conditions.append(True)
        self.ast[ctx] = ast.WhenStatement(
            conditions=conditions,
            blocks=blocks)

    def exitFor_statement(self, ctx: ModelicaParser.For_statementContext):
        self.ast[ctx] = ast.ForStatement(
            indices=self.ast[ctx.for_indices()],
            statements=self.ast[ctx.statement_block()])

    # EXPRESSIONS ===========================================================

    def exitSimple_expression(self, ctx: ModelicaParser.Simple_expressionContext):
        if len(ctx.expr()) > 1:
            if len(ctx.expr()) > 2:
                step = self.ast[ctx.expr()[2]]
            else:
                step = ast.Primary(value=1)
            self.ast[ctx] = ast.Slice(start=self.ast[ctx.expr()[0]], stop=self.ast[ctx.expr()[1]], step=step)
        else:
            self.ast[ctx] = self.ast[ctx.expr()[0]]

    def exitExpression_simple(self, ctx: ModelicaParser.Expression_simpleContext):
        self.ast[ctx] = self.ast[ctx.simple_expression()]

    def exitExpression_if(self, ctx: ModelicaParser.Expression_ifContext):
        all_expr = [self.ast[s] for s in ctx.expression()]
        # Note that an else block is guaranteed to exist.
        conditions = all_expr[:-1:2]
        expressions = all_expr[1::2] + all_expr[-1:]

        self.ast[ctx] = ast.IfExpression(
            conditions=conditions,
            expressions=expressions)

    def exitExpr_primary(self, ctx: ModelicaParser.Expr_primaryContext):
        self.ast[ctx] = self.ast[ctx.primary()]

    def exitExpr_add(self, ctx: ModelicaParser.Expr_addContext):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,
            operands=[self.ast[e] for e in ctx.expr()]
        )

    def exitExpr_exp(self, ctx: ModelicaParser.Expr_expContext):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,
            operands=[self.ast[e] for e in ctx.primary()]
        )

    def exitExpr_mul(self, ctx: ModelicaParser.Expr_mulContext):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,
            operands=[self.ast[e] for e in ctx.expr()]
        )

    def exitExpr_rel(self, ctx: ModelicaParser.Expr_relContext):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,
            operands=[self.ast[e] for e in ctx.expr()]
        )

    def exitExpr_neg(self, ctx: ModelicaParser.Expr_negContext):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,
            operands=[self.ast[ctx.expr()]]
        )

    def exitSubscript(self, ctx: ModelicaParser.SubscriptContext):
        if ctx.expression() is not None:
            self.ast[ctx] = self.ast[ctx.expression()]
        else:
            self.ast[ctx] = ast.Slice()

    def exitArray_subscripts(self, ctx: ModelicaParser.Array_subscriptsContext):
        self.ast[ctx] = [self.ast[s] for s in ctx.subscript()]

    def exitFor_index(self, ctx):
        self.ast[ctx] = ast.ForIndex(name=ctx.IDENT().getText(), expression=self.ast[ctx.expression()])

    def exitFor_indices(self, ctx: ModelicaParser.For_indicesContext):
        self.ast[ctx] = [self.ast[s] for s in ctx.for_index()]

    # PRIMARY ===========================================================

    def exitPrimary_unsigned_number(self, ctx: ModelicaParser.Primary_unsigned_numberContext):
        self.ast[ctx] = ast.Primary(value=float(ctx.getText()))

    def exitPrimary_string(self, ctx: ModelicaParser.Primary_stringContext):
        val = ctx.getText()
        assert val.startswith('"') and val.endswith('"')
        self.ast[ctx] = ast.Primary(value=val[1:-1])

    def exitPrimary_false(self, ctx: ModelicaParser.Primary_falseContext):
        self.ast[ctx] = ast.Primary(value=False)

    def exitPrimary_true(self, ctx: ModelicaParser.Primary_trueContext):
        self.ast[ctx] = ast.Primary(value=True)

    def exitPrimary_function(self, ctx: ModelicaParser.Primary_functionContext):
        # TODO: Could possible be cleaner if we let the expression in the ast bubble up.
        #       E.g. self.ast[x] below, instead of self.ast[x.expression].
        self.ast[ctx] = ast.Expression(
            operator=self.ast[ctx.component_reference()],
            operands=[self.ast[x.expression()]
                      for x in ctx.function_call_args().function_arguments().function_argument()]
        )

    def exitPrimary_derivative(self, ctx: ModelicaParser.Primary_derivativeContext):
        self.ast[ctx] = ast.Expression(
            operator='der',
            operands=[self.ast[x.expression()]
                      for x in ctx.function_call_args().function_arguments().function_argument()]
        )
        # TODO 'state' is not a standard prefix;  disable this for now as it does not work
        # when differentiating states defined in superclasses.
        # if 'state' not in self.class_node.symbols[comp_name].prefixes:
        #    self.class_node.symbols[comp_name].prefixes += ['state']

    def exitType_specifier_element(self, ctx: ModelicaParser.Type_specifier_elementContext):
        self.ast[ctx] = ast.ComponentRef(
            name=ctx.IDENT().getText(),
            indices=[[None]],
            child=[]
        )

    def exitType_specifier(self, ctx: ModelicaParser.Type_specifierContext):
        for element in reversed([self.ast[ctx] for ctx in ctx.type_specifier_element()]):
            if ctx in self.ast:
                element.child = [self.ast[ctx]]
            self.ast[ctx] = element

    def exitComponent_reference_element(self, ctx: ModelicaParser.Component_reference_elementContext):
        if ctx.array_subscripts() is not None:
            indices = [[self.ast[x] for x in ctx.array_subscripts().subscript()]]
        else:
            indices = [[None]]
        self.ast[ctx] = ast.ComponentRef(
            name=ctx.IDENT().getText(),
            indices=indices,
            child=[]
        )

    def exitComponent_reference(self, ctx: ModelicaParser.Component_referenceContext):
        for element in reversed([self.ast[ctx] for ctx in ctx.component_reference_element()]):
            if ctx in self.ast:
                element.child = [self.ast[ctx]]
            self.ast[ctx] = element

    def exitPrimary_component_reference(self, ctx: ModelicaParser.Primary_component_referenceContext):
        self.ast[ctx] = self.ast[ctx.component_reference()]

    def exitPrimary_output_expression_list(self, ctx: ModelicaParser.Primary_output_expression_listContext):
        self.ast[ctx] = [self.ast[x] for x in ctx.output_expression_list().expression()]
        # Collapse lists containing a single expression
        if len(self.ast[ctx]) == 1:
            self.ast[ctx] = self.ast[ctx][0]

    def exitPrimary_function_arguments(self, ctx: ModelicaParser.Primary_function_argumentsContext):
        # TODO: This does not support for generators yet.
        #       Only expressions are supported, e.g. {1.0, 2.0, 3.0}.
        v = [self.ast[x.expression()] for x in ctx.function_arguments().function_argument()]
        self.ast[ctx] = ast.Array(values=v)

    def exitEquation_function(self, ctx: ModelicaParser.Equation_functionContext):
        self.ast[ctx] = ast.Function(
            name=ctx.name().getText(),
            arguments=[self.ast[x.expression()]
                for x in ctx.function_call_args().function_arguments().function_argument()]
        )

    def exitEquation_when(self, ctx: ModelicaParser.Equation_whenContext):
        self.ast[ctx] = self.ast[ctx.when_equation()]

    # COMPONENTS ===========================================================

    def exitElement_list(self, ctx: ModelicaParser.Element_listContext):
        self.ast[ctx] = [self.ast[e] for e in ctx.element()]

    def exitElement(self, ctx: ModelicaParser.ElementContext):
        self.ast[ctx] = self.ast[ctx.getChild(ctx.getAltNumber())]

    def exitImport_list(self, ctx: ModelicaParser.Import_listContext):
        self.ast[ctx] = [ctx.IDENT()] + self.ast[ctx.import_list()]

    def exitImport_clause(self, ctx: ModelicaParser.Import_clauseContext):
        component = self.ast[ctx.component_reference()]
        if ctx.IDENT() is not None:
            self.ast[ctx] = ast.ImportAsClause(component=component, name=ctx.IDENT().getText())
        else:
            symbols = self.ast[ctx.import_list()]
            self.ast[ctx] = ast.ImportFromClause(component=component, symbols=symbols)
        self.class_node.imports += [self.ast[ctx]]

    def enterExtends_clause(self, ctx: ModelicaParser.Extends_clauseContext):
        self.in_extends_clause = True

    def exitExtends_clause(self, ctx: ModelicaParser.Extends_clauseContext):
        if ctx.class_modification() is not None:
            class_modification = self.ast[ctx.class_modification()]
        else:
            class_modification = ast.ClassModification()
        self.ast[ctx] = ast.ExtendsClause(component=self.ast[ctx.component_reference()],
                                          class_modification=class_modification)
        self.class_node.extends += [self.ast[ctx]]

        self.in_extends_clause = False

    def exitRegular_element(self, ctx: ModelicaParser.Regular_elementContext):
        if ctx.comp_elem is not None:
            self.ast[ctx] = self.ast[ctx.comp_elem]
        else:
            self.ast[ctx] = self.ast[ctx.class_elem]

    def exitReplaceable_element(self, ctx: ModelicaParser.Replaceable_elementContext):
        if ctx.comp_elem is not None:
            self.ast[ctx] = self.ast[ctx.comp_elem]
        else:
            self.ast[ctx] = self.ast[ctx.class_elem]

    def enterComponent_clause(self, ctx: ModelicaParser.Component_clauseContext):
        prefixes = ctx.type_prefix().getText().split(' ')
        if prefixes[0] == '':
            prefixes = []
        self.ast[ctx] = ast.ComponentClause(
            prefixes=prefixes,
        )
        self.comp_clause = self.ast[ctx]

    def enterComponent_clause1(self, ctx: ModelicaParser.Component_clause1Context):
        prefixes = ctx.type_prefix().getText().split(' ')
        if prefixes[0] == '':
            prefixes = []
        self.ast[ctx] = ast.ComponentClause(
            prefixes=prefixes,
        )
        self.comp_clause = self.ast[ctx]

    def exitComponent_clause(self, ctx: ModelicaParser.Component_clauseContext):
        clause = self.ast[ctx]
        # The component clause and all its symbols share the same type.
        # However, the type will only be turned into a component reference
        # somewhere between the enterDeclaration and exitDeclaration functions
        # of the symbols. Therefore, we need to keep the component clause's
        # type, and all its symbols' types, pointing at the same empty
        # (ComponentRef) object until we can fill it.
        clause.type.__dict__.update(self.ast[ctx.type_specifier()].__dict__)
        if ctx.array_subscripts() is not None:
            clause.dimensions = self.ast[ctx.array_subscripts()]
            for sym in self.comp_clause.symbol_list:
                s = self.class_node.symbols[sym.name]
                s.dimensions = clause.dimensions

        # We make sure that all references to the objects are unique per
        # symbol making copies. Note that if there is only one symbol in the
        # component clause, it is already unique.
        for sym in self.comp_clause.symbol_list[1:]:
            s = self.class_node.symbols[sym.name]
            s.dimensions = list(s.dimensions)
            s.prefixes = list(s.prefixes)
            s.type = copy.deepcopy(clause.type)

    def exitComponent_clause1(self, ctx: ModelicaParser.Component_clause1Context):
        clause = self.ast[ctx]
        clause.type.__dict__.update(self.ast[ctx.type_specifier()].__dict__)

        for sym in self.comp_clause.symbol_list[1:]:
            s = self.class_node.symbols[sym.name]
            s.dimensions = list(s.dimensions)
            s.prefixes = list(s.prefixes)
            s.type = copy.deepcopy(clause.type)

    def enterComponent_declaration(self, ctx: ModelicaParser.Component_declarationContext):
        sym = ast.Symbol(order=self.sym_count)
        self.sym_count += 1
        self.ast[ctx] = sym
        self.symbol_node = sym
        self.comp_clause.symbol_list += [sym]

    def enterComponent_declaration1(self, ctx: ModelicaParser.Component_declaration1Context):
        sym = ast.Symbol(order=self.sym_count)
        self.sym_count += 1
        self.ast[ctx] = sym
        self.symbol_node = sym
        self.comp_clause.symbol_list += [sym]

    def enterElement_modification(self, ctx: ModelicaParser.Element_modificationContext):
        if self.symbol_node is not None:
            self.ast[ctx] = self.symbol_node
        else:
            sym = ast.Symbol(order=self.sym_count)
            self.sym_count += 1
            self.ast[ctx] = sym
            self.symbol_node = sym

    def exitComponent_declaration(self, ctx: ModelicaParser.Component_declarationContext):
        self.ast[ctx].comment = self.ast[ctx.comment()]
        self.symbol_node = None

    def exitComponent_declaration1(self, ctx: ModelicaParser.Component_declaration1Context):
        self.ast[ctx].comment = self.ast[ctx.comment()]
        self.symbol_node = None

    def enterDeclaration(self, ctx: ModelicaParser.DeclarationContext):
        sym = self.symbol_node
        dimensions = None
        if self.comp_clause.dimensions is not None:
            dimensions = self.comp_clause.dimensions
        sym.name = ctx.IDENT().getText()
        sym.dimensions = dimensions
        sym.prefixes = self.comp_clause.prefixes
        sym.type = self.comp_clause.type

        # Declarations can also occur in extends clauses, in which case we do not have to add it to the class's symbols.
        if not self.in_extends_clause:
            if sym.name in self.class_node.symbols:
                raise IOError(sym.name, 'already defined')
            self.class_node.symbols[sym.name] = sym

    def exitDeclaration(self, ctx: ModelicaParser.DeclarationContext):
        sym = self.symbol_node
        if ctx.array_subscripts() is not None:
            sym.dimensions = self.ast[ctx.array_subscripts()]
        if ctx.modification() is not None:
            for mod in self.ast[ctx.modification()]:
                if isinstance(mod, ast.ClassModification):
                    sym.class_modification = mod
                else:
                    # Assignment of value, which we turn into a modification here.
                    vmod_arg = ast.ClassModificationArgument()
                    vmod_arg.value = ast.ElementModification()
                    vmod_arg.value.component = ast.ComponentRef(name="value")
                    vmod_arg.value.modifications = [mod]

                    if sym.class_modification is None:
                        sym_mod = ast.ClassModification()
                        sym_mod.arguments.append(vmod_arg)
                        sym.class_modification = sym_mod
                    else:
                        sym.class_modification.arguments.append(vmod_arg)

    def exitElement_modification(self, ctx: ModelicaParser.Element_modificationContext):
        component = self.ast[ctx.component_reference()]
        if ctx.modification() is not None:
            modifications = self.ast[ctx.modification()]
        else:
            modifications = []

        self.ast[ctx] = ast.ElementModification(component=component, modifications=modifications)

    def exitModification_class(self, ctx: ModelicaParser.Modification_classContext):
        self.ast[ctx] = [self.ast[ctx.class_modification()]]
        if ctx.expression() is not None:
            self.ast[ctx] += [self.ast[ctx.expression()]]

    def exitModification_assignment(self, ctx: ModelicaParser.Modification_assignmentContext):
        self.ast[ctx] = [self.ast[ctx.expression()]]

    def exitModification_assignment2(self, ctx: ModelicaParser.Modification_assignment2Context):
        self.ast[ctx] = [self.ast[ctx.expression()]]

    def exitElement_replaceable(self, ctx: ModelicaParser.Element_replaceableContext):
        if ctx.component_clause1() is not None:
            self.ast[ctx] = self.ast[ctx.component_clause1()]
        else:
            self.ast[ctx] = self.ast[ctx.short_class_definition()]

    def exitElement_modification_or_replaceable(self, ctx: ModelicaParser.Element_modification_or_replaceableContext):
        if ctx.element_modification() is not None:
            self.ast[ctx] = self.ast[ctx.element_modification()]
        else:
            self.ast[ctx] = self.ast[ctx.element_replaceable()]

    def exitElement_redeclaration(self, ctx: ModelicaParser.Element_redeclarationContext):
        if ctx.component_clause1() is not None:
            self.ast[ctx] = self.ast[ctx.component_clause1()]
        else:
            self.ast[ctx] = self.ast[ctx.short_class_definition()]

    # COMMENTS ==============================================================

    def exitComment(self, ctx: ModelicaParser.CommentContext):
        self.ast[ctx] = self.ast[ctx.string_comment()]

    def exitString_comment(self, ctx: ModelicaParser.String_commentContext):
        self.ast[ctx] = ctx.getText()[1:-1]

    # ANNOTATIONS ==========================================================

    def exitAnnotation(self, ctx: ModelicaParser.AnnotationContext):
        self.ast[ctx] = self.ast[ctx.class_modification()]


# UTILITY FUNCTIONS ========================================================
def file_to_tree(f: ModelicaFile) -> ast.Tree:
    # TODO: We can only insert where classes exist. For example, if we have a
    # within statement, we have to check if the nodes of the within statement
    # are actually in the tree, and if not raise an exception.
    root = ast.Tree()
    insert_node = root
    if f.within:
        for p in f.within[0].to_tuple():
            package = ast.Class(name=p, type="package")
            insert_node.classes[p] = package
            insert_node = package

    insert_node.classes.update(f.classes)

    root.update_parent_refs()

    return root

def parse(text):
    input_stream = antlr4.InputStream(text)
    lexer = ModelicaLexer(input_stream)
    stream = antlr4.CommonTokenStream(lexer)
    parser = ModelicaParser(stream)
    # parser.buildParseTrees = False
    parse_tree = parser.stored_definition()
    ast_listener = ASTListener()
    parse_walker = antlr4.ParseTreeWalker()
    parse_walker.walk(ast_listener, parse_tree)
    modelica_file = ast_listener.ast_result
    return file_to_tree(modelica_file)
