grammar Modelica;
// TODO: Update to MLS 3.5 (this appears to be 3.3.0)
//=========================================================
//  B.2.1 Stored Definition - Within
//=========================================================

// B.2.1.1 ------------------------------------------------
stored_definition :
    (WITHIN component_reference? ';')?
    (stored_definition_class)*
    ;

stored_definition_class :
    FINAL? class_definition ';'
    ;

//=========================================================
//  B.2.2 Class Definition
//=========================================================

// B.2.2.1 ------------------------------------------------
class_definition :
    ENCAPSULATED? class_prefixes
    class_specifier
    ;

// B.2.2.2 ------------------------------------------------
class_prefixes :
    PARTIAL?
    class_type
    ;

class_type:
    'class'
        | 'model'
        | 'operator'? 'record'
        | 'block'
        | 'expandable'? 'connector'
        | 'type'
        | 'package'
        | ('pure' | 'impure')? 'operator'? 'function'
        | 'operator'
    ;

// B.2.2.3 ------------------------------------------------
class_specifier :
    IDENT string_comment composition 'end' IDENT                    # class_spec_comp
    | IDENT '=' base_prefix component_reference
        class_modification? comment                                 # class_spec_base
    | IDENT '=' 'enumeration' '(' (enum_list? | ':') ')' comment    # class_spec_enum
    | IDENT '=' 'der' '(' name ',' IDENT (',' IDENT )* ')' comment  # class_spec_der
    | 'extends' IDENT class_modification? string_comment
        composition 'end' IDENT                                     # class_spec_extends
    ;

// B.2.2.4 ------------------------------------------------
base_prefix :
    type_prefix
    ;

// B.2.2.5 ------------------------------------------------
enum_list :
    enumeration_literal (',' enumeration_literal)*
    ;

// B.2.2.6 ------------------------------------------------
enumeration_literal :
    IDENT comment
    ;

// B.2.2.7 ------------------------------------------------
composition :
    epriv=element_list
    (
        'public' epub=element_list
        | 'protected' epro=element_list
        | equation_section
        | algorithm_section
    )*
    ( 'external' language_specification?
        external_function_call?
        ext_annotation=annotation? ';')?
    (comp_annotation=annotation ';')?
    ;

// B.2.2.8 ------------------------------------------------
language_specification :
    STRING
    ;

// B.2.2.9 ------------------------------------------------
external_function_call :
    (component_reference '=')?
    IDENT '(' expression_list? ')'
    ;

// B.2.2.10 ------------------------------------------------
element_list :
    (element ';')*
    ;

// B.2.2.11 ------------------------------------------------
element :
    import_clause
    | extends_clause
    | regular_element
    | replaceable_element
    ;

regular_element:
    REDECLARE? FINAL? INNER? OUTER?
    (class_elem=class_definition | comp_elem=component_clause)
    ;

replaceable_element:
    REDECLARE? FINAL? INNER? OUTER? 'replaceable'
    (class_elem=class_definition | comp_elem=component_clause)
    (constraining_clause comment)?
    ;

// B.2.2.12 ------------------------------------------------
import_clause :
    'import' ( IDENT '=' component_reference
        | component_reference ('.*' | '.{' import_list '}' )? ) comment
    ;

// B.2.2.13 ------------------------------------------------
import_list :
    IDENT (',' import_list)*
    ;

//=========================================================
// B.2.3 Extends
//=========================================================

// B.2.3.1 ------------------------------------------------
extends_clause :
    'extends' component_reference class_modification? annotation?
    ;

// B.2.3.2 ------------------------------------------------
constraining_clause:
    'constrainedby' name class_modification?
    ;

//=========================================================
// B.2.4 Component Clause
//=========================================================

// B.2.4.1 ------------------------------------------------
component_clause :
    type_prefix type_specifier array_subscripts? component_list
    ;

// B.2.4.2 ------------------------------------------------
type_prefix :
    ('flow' | 'stream')?
    ('discrete' | 'parameter' | 'constant')?
    ('input' | 'output')?
    ;

// B.2.4.3 ------------------------------------------------
type_specifier_element :
    IDENT
    ;

type_specifier :
    type_specifier_element ('.' type_specifier_element)*
    ;

// B.2.4.4 ------------------------------------------------
component_list:
    component_declaration ( ',' component_declaration)*
    ;

// B.2.4.5 ------------------------------------------------
component_declaration :
    declaration condition_attribute? comment
    ;

// B.2.4.6 ------------------------------------------------
condition_attribute :
    'if' expression
    ;

// B.2.4.7 ------------------------------------------------
declaration :
    IDENT array_subscripts? modification?
    ;

//=========================================================
// B.2.5 Modification
//=========================================================

// B.2.5.1 ------------------------------------------------
modification :
    class_modification ('=' expression)?    # modification_class
    | '=' expression                        # modification_assignment
    | ':=' expression                       # modification_assignment2
    ;

// B.2.5.2 ------------------------------------------------
class_modification :
    '(' argument_list? ')'
    ;

// B.2.5.3 ------------------------------------------------
argument_list :
    argument (',' argument)*
    ;

// B.2.5.4 ------------------------------------------------
argument :
    element_modification_or_replaceable
    | element_redeclaration
    ;

// B.2.5.5 ------------------------------------------------
element_modification_or_replaceable:
    EACH? FINAL?
    (element_modification | element_replaceable)
    ;

// B.2.5.6 ------------------------------------------------
element_modification :
    component_reference modification? string_comment
    ;

// B.2.5.7 ------------------------------------------------
element_redeclaration :
    REDECLARE EACH? FINAL?
    ( (short_class_definition | component_clause1)
      | element_replaceable)
    ;

// B.2.5.8 ------------------------------------------------
element_replaceable :
    'replaceable'
    (short_class_definition | component_clause1)
    constraining_clause?
    ;

// B.2.5.9 ------------------------------------------------
component_clause1 :
    type_prefix type_specifier component_declaration1
    ;

// B.2.5.10 ------------------------------------------------
component_declaration1 :
    declaration comment
    ;

// B.2.5.11 ------------------------------------------------
short_class_definition :
    class_prefixes IDENT '='
    ( base_prefix component_reference array_subscripts?
        class_modification? comment
        | 'enumeration' '(' (enum_list? | ':') ')' comment)
    ;

//=========================================================
// B.2.6 Equations
//=========================================================

// B.2.6.1 ------------------------------------------------
equation_block :
    (equation ';')*
    ;

equation_section :
    INITIAL? 'equation' equation_block
    ;

// B.2.6.2 ------------------------------------------------
statement_block :
    (statement ';')*
    ;

algorithm_section :
    INITIAL? 'algorithm' statement_block
    ;

// B.2.6.3 ------------------------------------------------
equation_options :
    simple_expression '=' expression    # equation_simple
    | if_equation                       # equation_if
    | for_equation                      # equation_for
    | connect_clause                    # equation_connect_clause
    | when_equation                     # equation_when
    | name function_call_args           # equation_function
    ;

// B.2.6.4 ------------------------------------------------
equation :
    equation_options
    comment
    ;

// B.2.6.5 ------------------------------------------------
statement_options :
    component_reference (':=' expression | function_call_args)  # statement_component_reference
    | '(' component_reference (',' component_reference)* ')' ':='
        component_reference function_call_args                  # statement_component_function
    | 'break'           # statement_break
    | 'return'          # statement_return
    | if_statement      # statement_if
    | for_statement     # statement_for
    | while_statement   # statement_while
    | when_statement    # statement_when
    ;

// B.2.6.6 ------------------------------------------------
statement :
    statement_options
    comment
    ;

// B.2.6.7 ------------------------------------------------
if_equation :
    'if' conditions+=expression 'then'
        blocks+=equation_block
    ('elseif' conditions+=expression 'then'
        blocks+=equation_block
    )*
    ('else'
        blocks+=equation_block
    )?
    'end' 'if'
    ;

// B.2.6.8 ------------------------------------------------
if_statement :
    'if' conditions+=expression 'then'
        blocks+=statement_block
    ('elseif' conditions+=expression 'then'
        blocks+=statement_block
    )*
    ('else'
        blocks+=statement_block
    )?
    'end' 'if'
    ;

// B.2.6.9 ------------------------------------------------
for_equation :
    'for' indices=for_indices 'loop'
        block=equation_block
    'end' 'for'
    ;

// B.2.6.10 ------------------------------------------------
for_statement :
    'for' indices=for_indices 'loop'
        block=statement_block
    'end' 'for'
    ;

// B.2.6.11 ------------------------------------------------
for_indices :
    for_index (',' for_index)*
    ;

// B.2.6.12 ------------------------------------------------
for_index :
    IDENT ('in' expression)?
    ;

// B.2.6.13 ------------------------------------------------
while_statement:
    'while' condition=expression 'loop'
        block=statement_block
    'end' 'while'
    ;

// B.2.6.14 ------------------------------------------------
when_equation:
    'when' conditions+=expression 'then'
        blocks+=equation_block
    ('elsewhen' conditions+=expression 'then'
        blocks+=equation_block
    )*
    'end' 'when'
    ;

// B.2.6.15 ------------------------------------------------
when_statement:
    'when' conditions+=expression 'then'
        blocks+=statement_block
    ('elsewhen' conditions+=expression 'then'
        blocks+=statement_block
    )*
    'end' 'when'
    ;

// B.2.6.16 ------------------------------------------------
connect_clause :
    'connect' '(' component_reference ',' component_reference ')'
    ;

//=========================================================
// B.2.7 Expressions
//=========================================================

// B.2.7.1 ------------------------------------------------
// TODO: What is the difference between expression and simple_expression?
//       Can't we get rid of one of them?
expression :
    simple_expression                                       # expression_simple
    | 'if' conditions+=expression 'then' blocks+=expression
    ( 'elseif' conditions+=expression 'then' blocks+=expression)*
    'else' blocks+=expression                               # expression_if
    ;

// B.2.7.2 ------------------------------------------------
simple_expression :
    expr (':' expr
        (':' expr)?)?
    ;

// B.2.7.3 ------------------------------------------------
expr :
    op=('+' | '-') expr                                     # expr_signed
    | primary op=('^' | '.^') primary                       # expr_exp
    | expr op=('*' | '/' | '.*' | './') expr                # expr_mul
    | expr  op=('+' | '-' | '.+' | '.-') expr               # expr_add
    | expr  op=('<' | '<=' | '>' | '>=' | '==' | '<>') expr # expr_rel
    | 'not' expr                                            # expr_not
    | expr  'and' expr                                      # expr_and
    | expr  'or' expr                                       # expr_or
    | primary                                               # expr_primary
    ;

// B.2.7.4 ------------------------------------------------
// TODO: Figure out what an output_expression_list is (i.e. find an example).
primary :
    UNSIGNED_NUMBER                                     # primary_unsigned_number
    | STRING                                            # primary_string
    | 'false'                                           # primary_false
    | 'true'                                            # primary_true
    | component_reference function_call_args            # primary_function
    | 'der' function_call_args                          # primary_derivative
    | 'initial' function_call_args                      # primary_initial
    | component_reference                               # primary_component_reference
    | '(' output_expression_list ')'                    # primary_output_expression_list
    | '[' expression_list (';' expression_list)* ']'    # primary_expression_list
    | '{' function_arguments '}'                        # primary_function_arguments
    | 'end'                                             # primary_end
    ;

// B.2.7.5 ------------------------------------------------
name :
    '.'? IDENT ('.' IDENT)*
    ;

// B.2.7.6 ------------------------------------------------
component_reference_element :
    IDENT array_subscripts?
    ;

component_reference :
    component_reference_element ('.' component_reference_element)*
    ;

// B.2.7.7 ------------------------------------------------
function_call_args :
    '(' function_arguments? ')'
    ;

// B.2.7.8 ------------------------------------------------
function_arguments :
    function_argument (',' function_argument | 'for' for_indices)*
    | named_arguments
    ;

// B.2.7.9 ------------------------------------------------
named_arguments : named_argument (',' named_argument)*
    ;

// B.2.7.10 ------------------------------------------------
named_argument : IDENT '=' function_argument
    ;

// B.2.7.11 ------------------------------------------------
function_argument :
    'function' name '(' named_arguments? ')'    # argument_function
    | expression                                # argument_expression
    ;

// B.2.7.12 ------------------------------------------------
output_expression_list :
    expression? (',' expression)*
    ;

// B.2.7.13 ------------------------------------------------
expression_list :
    expression (',' expression)*
    ;

// B.2.7.14 ------------------------------------------------
array_subscripts :
    '[' subscript (',' subscript)* ']'
    ;

// B.2.7.15 ------------------------------------------------
subscript :
    ':' | expression
    ;

// B.2.7.16 ------------------------------------------------
comment :
    string_comment annotation?
    ;

// B.2.7.17 ------------------------------------------------
string_comment :
    (STRING ('+' STRING)*)?
    ;

// B.2.7.18 ------------------------------------------------
annotation :
    'annotation' class_modification
    ;

//=========================================================
// Keywords
//=========================================================
EACH : 'each';
PARTIAL : 'partial';
FINAL : 'final';
WITHIN : 'within';
ENCAPSULATED : 'encapsulated';
REDECLARE : 'redeclare';
INNER : 'inner';
OUTER : 'outer';
INITIAL : 'initial';
IDENT : NONDIGIT ( DIGIT | NONDIGIT )* | Q_IDENT;
STRING : '"' ('\\"' | ~('"'))* '"';
//STRING : '"' (S_CHAR | S_ESCAPE | ' ')* '"';
UNSIGNED_NUMBER : UNSIGNED_INTEGER  ( '.' UNSIGNED_NUMBER? )* ( [eE] [+-]? UNSIGNED_INTEGER)?;
COMMENT :
    ('/' '/' .*? '\n' | '/*' .*? '*/') -> channel(HIDDEN)
    ;
WS  :   [ \r\n\t]+ -> skip ; // toss out whitespace

//=========================================================
// Fragments
//=========================================================
fragment Q_IDENT : '\'' ( Q_CHAR | S_ESCAPE)+;
fragment NONDIGIT : [_a-zA-Z];
fragment S_CHAR : [\u0000-\u00FF];
fragment Q_CHAR : NONDIGIT | DIGIT | [!#$%&()*+,-./:;<>=?@[\]^{}| ];
fragment S_ESCAPE : ('\\\'' | '\\"' | '\\\\' | '\\?' | '\\b' |
    '\\f' | '\\n' | '\\r' | '\\t' | '\\v' | '\\a');
fragment DIGIT :  [0-9];
fragment UNSIGNED_INTEGER : DIGIT+;
// vi:ts=4:sw=4:expandtab:
