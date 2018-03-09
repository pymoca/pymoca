# Generated from Modelica.g4 by ANTLR 4.7
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .ModelicaParser import ModelicaParser
else:
    from ModelicaParser import ModelicaParser

# This class defines a complete generic visitor for a parse tree produced by ModelicaParser.

class ModelicaVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by ModelicaParser#stored_definition.
    def visitStored_definition(self, ctx:ModelicaParser.Stored_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#stored_definition_class.
    def visitStored_definition_class(self, ctx:ModelicaParser.Stored_definition_classContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_definition.
    def visitClass_definition(self, ctx:ModelicaParser.Class_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_prefixes.
    def visitClass_prefixes(self, ctx:ModelicaParser.Class_prefixesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_type.
    def visitClass_type(self, ctx:ModelicaParser.Class_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_spec_comp.
    def visitClass_spec_comp(self, ctx:ModelicaParser.Class_spec_compContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_spec_base.
    def visitClass_spec_base(self, ctx:ModelicaParser.Class_spec_baseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_spec_enum.
    def visitClass_spec_enum(self, ctx:ModelicaParser.Class_spec_enumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_spec_der.
    def visitClass_spec_der(self, ctx:ModelicaParser.Class_spec_derContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_spec_extends.
    def visitClass_spec_extends(self, ctx:ModelicaParser.Class_spec_extendsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#base_prefix.
    def visitBase_prefix(self, ctx:ModelicaParser.Base_prefixContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#enum_list.
    def visitEnum_list(self, ctx:ModelicaParser.Enum_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#enumeration_literal.
    def visitEnumeration_literal(self, ctx:ModelicaParser.Enumeration_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#composition.
    def visitComposition(self, ctx:ModelicaParser.CompositionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#language_specification.
    def visitLanguage_specification(self, ctx:ModelicaParser.Language_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#external_function_call.
    def visitExternal_function_call(self, ctx:ModelicaParser.External_function_callContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#element_list.
    def visitElement_list(self, ctx:ModelicaParser.Element_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#element.
    def visitElement(self, ctx:ModelicaParser.ElementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#regular_element.
    def visitRegular_element(self, ctx:ModelicaParser.Regular_elementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#replaceable_element.
    def visitReplaceable_element(self, ctx:ModelicaParser.Replaceable_elementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#import_clause.
    def visitImport_clause(self, ctx:ModelicaParser.Import_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#import_list.
    def visitImport_list(self, ctx:ModelicaParser.Import_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#extends_clause.
    def visitExtends_clause(self, ctx:ModelicaParser.Extends_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#constraining_clause.
    def visitConstraining_clause(self, ctx:ModelicaParser.Constraining_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#component_clause.
    def visitComponent_clause(self, ctx:ModelicaParser.Component_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#type_prefix.
    def visitType_prefix(self, ctx:ModelicaParser.Type_prefixContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#type_specifier_element.
    def visitType_specifier_element(self, ctx:ModelicaParser.Type_specifier_elementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#type_specifier.
    def visitType_specifier(self, ctx:ModelicaParser.Type_specifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#component_list.
    def visitComponent_list(self, ctx:ModelicaParser.Component_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#component_declaration.
    def visitComponent_declaration(self, ctx:ModelicaParser.Component_declarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#condition_attribute.
    def visitCondition_attribute(self, ctx:ModelicaParser.Condition_attributeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#declaration.
    def visitDeclaration(self, ctx:ModelicaParser.DeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#modification_class.
    def visitModification_class(self, ctx:ModelicaParser.Modification_classContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#modification_assignment.
    def visitModification_assignment(self, ctx:ModelicaParser.Modification_assignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#modification_assignment2.
    def visitModification_assignment2(self, ctx:ModelicaParser.Modification_assignment2Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#class_modification.
    def visitClass_modification(self, ctx:ModelicaParser.Class_modificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#argument_list.
    def visitArgument_list(self, ctx:ModelicaParser.Argument_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#argument.
    def visitArgument(self, ctx:ModelicaParser.ArgumentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#element_modification_or_replaceable.
    def visitElement_modification_or_replaceable(self, ctx:ModelicaParser.Element_modification_or_replaceableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#element_modification.
    def visitElement_modification(self, ctx:ModelicaParser.Element_modificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#element_redeclaration.
    def visitElement_redeclaration(self, ctx:ModelicaParser.Element_redeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#element_replaceable.
    def visitElement_replaceable(self, ctx:ModelicaParser.Element_replaceableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#component_clause1.
    def visitComponent_clause1(self, ctx:ModelicaParser.Component_clause1Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#component_declaration1.
    def visitComponent_declaration1(self, ctx:ModelicaParser.Component_declaration1Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#short_class_definition.
    def visitShort_class_definition(self, ctx:ModelicaParser.Short_class_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_block.
    def visitEquation_block(self, ctx:ModelicaParser.Equation_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_section.
    def visitEquation_section(self, ctx:ModelicaParser.Equation_sectionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_block.
    def visitStatement_block(self, ctx:ModelicaParser.Statement_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#algorithm_section.
    def visitAlgorithm_section(self, ctx:ModelicaParser.Algorithm_sectionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_simple.
    def visitEquation_simple(self, ctx:ModelicaParser.Equation_simpleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_if.
    def visitEquation_if(self, ctx:ModelicaParser.Equation_ifContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_for.
    def visitEquation_for(self, ctx:ModelicaParser.Equation_forContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_connect_clause.
    def visitEquation_connect_clause(self, ctx:ModelicaParser.Equation_connect_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_when.
    def visitEquation_when(self, ctx:ModelicaParser.Equation_whenContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation_function.
    def visitEquation_function(self, ctx:ModelicaParser.Equation_functionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#equation.
    def visitEquation(self, ctx:ModelicaParser.EquationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_component_reference.
    def visitStatement_component_reference(self, ctx:ModelicaParser.Statement_component_referenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_component_function.
    def visitStatement_component_function(self, ctx:ModelicaParser.Statement_component_functionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_break.
    def visitStatement_break(self, ctx:ModelicaParser.Statement_breakContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_return.
    def visitStatement_return(self, ctx:ModelicaParser.Statement_returnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_if.
    def visitStatement_if(self, ctx:ModelicaParser.Statement_ifContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_for.
    def visitStatement_for(self, ctx:ModelicaParser.Statement_forContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_while.
    def visitStatement_while(self, ctx:ModelicaParser.Statement_whileContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement_when.
    def visitStatement_when(self, ctx:ModelicaParser.Statement_whenContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#statement.
    def visitStatement(self, ctx:ModelicaParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#if_equation.
    def visitIf_equation(self, ctx:ModelicaParser.If_equationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#if_statement.
    def visitIf_statement(self, ctx:ModelicaParser.If_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#for_equation.
    def visitFor_equation(self, ctx:ModelicaParser.For_equationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#for_statement.
    def visitFor_statement(self, ctx:ModelicaParser.For_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#for_indices.
    def visitFor_indices(self, ctx:ModelicaParser.For_indicesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#for_index.
    def visitFor_index(self, ctx:ModelicaParser.For_indexContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#while_statement.
    def visitWhile_statement(self, ctx:ModelicaParser.While_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#when_equation.
    def visitWhen_equation(self, ctx:ModelicaParser.When_equationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#when_statement.
    def visitWhen_statement(self, ctx:ModelicaParser.When_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#connect_clause.
    def visitConnect_clause(self, ctx:ModelicaParser.Connect_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expression_simple.
    def visitExpression_simple(self, ctx:ModelicaParser.Expression_simpleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expression_if.
    def visitExpression_if(self, ctx:ModelicaParser.Expression_ifContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#simple_expression.
    def visitSimple_expression(self, ctx:ModelicaParser.Simple_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_neg.
    def visitExpr_neg(self, ctx:ModelicaParser.Expr_negContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_add.
    def visitExpr_add(self, ctx:ModelicaParser.Expr_addContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_exp.
    def visitExpr_exp(self, ctx:ModelicaParser.Expr_expContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_or.
    def visitExpr_or(self, ctx:ModelicaParser.Expr_orContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_primary.
    def visitExpr_primary(self, ctx:ModelicaParser.Expr_primaryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_and.
    def visitExpr_and(self, ctx:ModelicaParser.Expr_andContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_rel.
    def visitExpr_rel(self, ctx:ModelicaParser.Expr_relContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_not.
    def visitExpr_not(self, ctx:ModelicaParser.Expr_notContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expr_mul.
    def visitExpr_mul(self, ctx:ModelicaParser.Expr_mulContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_unsigned_number.
    def visitPrimary_unsigned_number(self, ctx:ModelicaParser.Primary_unsigned_numberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_string.
    def visitPrimary_string(self, ctx:ModelicaParser.Primary_stringContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_false.
    def visitPrimary_false(self, ctx:ModelicaParser.Primary_falseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_true.
    def visitPrimary_true(self, ctx:ModelicaParser.Primary_trueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_function.
    def visitPrimary_function(self, ctx:ModelicaParser.Primary_functionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_derivative.
    def visitPrimary_derivative(self, ctx:ModelicaParser.Primary_derivativeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_initial.
    def visitPrimary_initial(self, ctx:ModelicaParser.Primary_initialContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_component_reference.
    def visitPrimary_component_reference(self, ctx:ModelicaParser.Primary_component_referenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_output_expression_list.
    def visitPrimary_output_expression_list(self, ctx:ModelicaParser.Primary_output_expression_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_expression_list.
    def visitPrimary_expression_list(self, ctx:ModelicaParser.Primary_expression_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_function_arguments.
    def visitPrimary_function_arguments(self, ctx:ModelicaParser.Primary_function_argumentsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#primary_end.
    def visitPrimary_end(self, ctx:ModelicaParser.Primary_endContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#name.
    def visitName(self, ctx:ModelicaParser.NameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#component_reference_element.
    def visitComponent_reference_element(self, ctx:ModelicaParser.Component_reference_elementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#component_reference.
    def visitComponent_reference(self, ctx:ModelicaParser.Component_referenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#function_call_args.
    def visitFunction_call_args(self, ctx:ModelicaParser.Function_call_argsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#function_arguments.
    def visitFunction_arguments(self, ctx:ModelicaParser.Function_argumentsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#named_arguments.
    def visitNamed_arguments(self, ctx:ModelicaParser.Named_argumentsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#named_argument.
    def visitNamed_argument(self, ctx:ModelicaParser.Named_argumentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#argument_function.
    def visitArgument_function(self, ctx:ModelicaParser.Argument_functionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#argument_expression.
    def visitArgument_expression(self, ctx:ModelicaParser.Argument_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#output_expression_list.
    def visitOutput_expression_list(self, ctx:ModelicaParser.Output_expression_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#expression_list.
    def visitExpression_list(self, ctx:ModelicaParser.Expression_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#array_subscripts.
    def visitArray_subscripts(self, ctx:ModelicaParser.Array_subscriptsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#subscript.
    def visitSubscript(self, ctx:ModelicaParser.SubscriptContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#comment.
    def visitComment(self, ctx:ModelicaParser.CommentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#string_comment.
    def visitString_comment(self, ctx:ModelicaParser.String_commentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ModelicaParser#annotation.
    def visitAnnotation(self, ctx:ModelicaParser.AnnotationContext):
        return self.visitChildren(ctx)



del ModelicaParser