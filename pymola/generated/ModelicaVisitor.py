# Generated from Modelica.g4 by ANTLR 4.5.1
from antlr4 import *

# This class defines a complete generic visitor for a parse tree produced by ModelicaParser.

class ModelicaVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by ModelicaParser#stored_definition.
    def visitStored_definition(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#stored_definition_class.
    def visitStored_definition_class(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_definition.
    def visitClass_definition(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_prefixes.
    def visitClass_prefixes(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_type.
    def visitClass_type(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_spec_comp.
    def visitClass_spec_comp(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_spec_base.
    def visitClass_spec_base(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_spec_enum.
    def visitClass_spec_enum(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_spec_der.
    def visitClass_spec_der(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_spec_extends.
    def visitClass_spec_extends(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#base_prefix.
    def visitBase_prefix(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#enum_list.
    def visitEnum_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#enumeration_literal.
    def visitEnumeration_literal(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#composition.
    def visitComposition(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#language_specification.
    def visitLanguage_specification(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#external_function_call.
    def visitExternal_function_call(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#element_list.
    def visitElement_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#element.
    def visitElement(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#regular_element.
    def visitRegular_element(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#replaceable_element.
    def visitReplaceable_element(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#import_clause.
    def visitImport_clause(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#import_list.
    def visitImport_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#extends_clause.
    def visitExtends_clause(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#constraining_clause.
    def visitConstraining_clause(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#component_clause.
    def visitComponent_clause(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#type_prefix.
    def visitType_prefix(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#type_specifier.
    def visitType_specifier(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#component_list.
    def visitComponent_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#component_declaration.
    def visitComponent_declaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#condition_attribute.
    def visitCondition_attribute(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#declaration.
    def visitDeclaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#modification_class.
    def visitModification_class(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#modification_assignment.
    def visitModification_assignment(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#modification_assignment2.
    def visitModification_assignment2(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_modification.
    def visitClass_modification(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#argument_list.
    def visitArgument_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#argument.
    def visitArgument(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#element_modification_or_replaceable.
    def visitElement_modification_or_replaceable(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#element_modification.
    def visitElement_modification(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#element_redeclaration.
    def visitElement_redeclaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#element_replaceable.
    def visitElement_replaceable(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#component_clause1.
    def visitComponent_clause1(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#component_declaration1.
    def visitComponent_declaration1(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#short_class_definition.
    def visitShort_class_definition(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_section.
    def visitEquation_section(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#algorithm_section.
    def visitAlgorithm_section(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_simple.
    def visitEquation_simple(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_if.
    def visitEquation_if(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_for.
    def visitEquation_for(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_connect_clause.
    def visitEquation_connect_clause(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_when.
    def visitEquation_when(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_function.
    def visitEquation_function(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation.
    def visitEquation(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_component_reference.
    def visitStatement_component_reference(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_component_function.
    def visitStatement_component_function(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_break.
    def visitStatement_break(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_return.
    def visitStatement_return(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_if.
    def visitStatement_if(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_for.
    def visitStatement_for(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_while.
    def visitStatement_while(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_when.
    def visitStatement_when(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement.
    def visitStatement(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#if_equation.
    def visitIf_equation(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#if_statement.
    def visitIf_statement(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#for_equation.
    def visitFor_equation(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#for_statement.
    def visitFor_statement(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#for_indices.
    def visitFor_indices(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#for_index.
    def visitFor_index(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#while_statement.
    def visitWhile_statement(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#when_equation.
    def visitWhen_equation(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#when_statement.
    def visitWhen_statement(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#connect_clause.
    def visitConnect_clause(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expression_simple.
    def visitExpression_simple(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expression_if.
    def visitExpression_if(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#simple_expression.
    def visitSimple_expression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_or.
    def visitExpr_or(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_primary.
    def visitExpr_primary(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_and.
    def visitExpr_and(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_rel.
    def visitExpr_rel(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_not.
    def visitExpr_not(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_neg.
    def visitExpr_neg(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_add.
    def visitExpr_add(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_mul.
    def visitExpr_mul(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_exp.
    def visitExpr_exp(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_unsigned_number.
    def visitPrimary_unsigned_number(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_string.
    def visitPrimary_string(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_false.
    def visitPrimary_false(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_true.
    def visitPrimary_true(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_function.
    def visitPrimary_function(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_derivative.
    def visitPrimary_derivative(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_initial.
    def visitPrimary_initial(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_component_reference.
    def visitPrimary_component_reference(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_output_expression_list.
    def visitPrimary_output_expression_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_expression_list.
    def visitPrimary_expression_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_function_arguments.
    def visitPrimary_function_arguments(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_end.
    def visitPrimary_end(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#name.
    def visitName(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#component_reference.
    def visitComponent_reference(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#function_call_args.
    def visitFunction_call_args(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#function_arguments.
    def visitFunction_arguments(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#named_arguments.
    def visitNamed_arguments(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#named_argument.
    def visitNamed_argument(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#argument_function.
    def visitArgument_function(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#argument_expression.
    def visitArgument_expression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#output_expression_list.
    def visitOutput_expression_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expression_list.
    def visitExpression_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#array_subscripts.
    def visitArray_subscripts(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#subscript.
    def visitSubscript(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#comment.
    def visitComment(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#string_comment.
    def visitString_comment(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#annotation.
    def visitAnnotation(self, ctx):
        return self.visitChildren(ctx)


