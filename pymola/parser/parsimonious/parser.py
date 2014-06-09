from parsimonious.grammar import Grammar

# This grammar is based off of the Modelica 3.3 standard.
# https://www.modelica.org/documents/ModelicaSpec33.pdf

modelica_parser = Grammar(r"""
    #===============================================================
    # STORED DEFINITION
    #===============================================================
    stored_definition = _ (within name? semicolon)?
        (_ final? class_definition semicolon)

    #===============================================================
    # CLASS DEFINITION
    #===============================================================
    class_definition = encapsulated? class_prefixes class_specifier

    class_prefixes = partial?
        (class/model/(operator? record)/block/
        (expandable? connector)/type/package/
        ((pure/impure)? operator? function)/
        operator)

    class_specifier = long_class_specifier/
        short_class_specifier/
        der_class_specifier/
        extends_class_specifier

    long_class_specifier = ident string_comment
        composition end ident

    short_class_specifier = (ident equals base_prefix name
        array_subscripts? class_modification? comment) /
        (ident  equals enumeration lparen (enum_list/colon) rparen
        comment)

    der_class_specifier = ident equals der lparen name
        (comma ident)+ rparen comment

    extends_class_specifier = extends ident class_modification?
        string_comment composition end ident

    base_prefix = type_prefix

    enum_list = enumeration_literal (comma enumeration_literal)*

    enumeration_literal = ident comment

    composition = element_list (((public/protected)
       element_list)/ equation_section/ algorithm_section)*
       (external language_specification?
       external_function_call? annotation? semicolon)?
       (annotation semicolon)?

    language_specification = string

    external_function_call = (component_reference equals)? ident
        lparen expression_list? rparen

    # notice we do a lookahead assertion here that is PEG but not EBNF
    # to ensure that end is not consumed as an ident
    element_list = (!(end/equation/algorithm)
        ((element semicolon)/(annotation semicolon)))*

    element = import_clause /  extends_clause
         / (redeclare? final? inner? outer? (
         (class_definition / component_clause)
         /(replaceable (class_definition / component_clause)
         constraining_clause comment)))

    import_clause = import ( (ident equals name) /
        (name (period (times / (lbrace import_list rbrace)) )?)) comment

    import_list = ident (comma import_list)?

    #===============================================================
    # EXTENDS
    #===============================================================
    extends_clause = extends name class_modification? annotation?

    constraining_clause = constrainedby name class_modification?

    #===============================================================
    # COMPONENT CLAUSE
    #===============================================================
    component_clause = type_prefix type_specifier array_subscripts?
       component_list

    type_prefix = (flow/ stream)? (discrete/parameter/constant)?
    (input/output)?

    type_specifier = name

    component_list = component_declaration (comma
      component_declaration)*

    component_declaration = declaration condition_attribute? comment?

    condition_attribute = if expression

    declaration = ident array_subscripts? modification?

    #===============================================================
    # MODIFICATION
    #===============================================================
    modification = (class modification ( equals expression)?) /
        ( equals expression) / (assign expression)

    class_modification = lparen argument_list? rparen

    argument_list = argument (comma argument)*

    argument = element_modification_or_replaceable /
        element_redeclaration

    element_modification_or_replaceable =
        each? final? (element_modification / element_replaceable)

    element_modification = name modification? string_comment

    element_redeclaration = redeclare each? final?

    element_replaceable = replaceable (short_class_definition /
        component_clause1) constraining_clause?

    component_clause1 = type_prefix type_specifier
        component_declaration1

    component_declaration1 = declaration comment

    short_class_definition = class_prefixes ident equals ((
        base_prefix name array_subscripts? class_modification?
        comment) / (enumeration lparen ( enum_list? / colon )
        rparen comment ))


    #===============================================================
    # EQUATION
    #===============================================================
    equation_section = initial? equation (equation_expr semicolon)*

    algorithm_section = initial? algorithm (statement semicolon)*

    # note there is also an equation keywords so we call the
    # expression equation_expr
    equation_expr = ((simple_expression equals expression)
        / if_equation / for_equation
        / connect_clause / when_equation
        / (name function_call_args)) comment

    statement = ((component_reference ( (assign expression)
        / function_call_args )) / ( lparen output_expression_list
        rparen assign component_reference function_call_args)
        / break / return / if_statement
        / for_statement / while_statement / when_statement )

    if_equation = if expression then
            (equation_expr semicolon)*
        (elseif expression then
            (equation_expr semicolon)* )*
        (else
            (equation_expr semicolon)* )?
        end if

    if_statement = if expression then
            (statement semicolon)*
        (elseif expression then
            (statement semicolon)*
        )*
        (else
            (statement semicolon)*
        )?
        end if

    for_equation = for for_indices loop
        (equation_expr semicolon)*
        end for

    for_statement = for for_indices loop
        (statement semicolon)*
        end for

    for_indices = for_index (comma for_index)*

    for_index = ident (in expression)?

    while_statement = while expression loop
        (statement semicolon)*
        end while

    when_equation = when expression then
            (equation semicolon)*
        (elsewhen expression then
            (equation semicolon)*
        )*
        end when

    when_statement = when expression then
            (statement semicolon)*
        (elsewhen expression then
            (statement semicolon)*
        )*
        end when

    connect_clause = connect lparen component_reference comma
        component_reference rparen

    #===============================================================
    # EXPRESSION
    #===============================================================
    expression = simple_expression /
        (if expression then expression
        (elseif expression then expression)* else expression)

    simple_expression = logical_expression
        (semicolon logical_expression
        (semicolon logical_expression)?)?

    logical_expression = logical_term (or logical_term)*

    logical_term = logical_factor (and logical_factor)*

    logical_factor = not? relation

    relation = arithmetic_expression (rel_op arithmetic_expression)?

    rel_op = less_than / less_than_or_equal / greater_than
        / greater_than_or_equal / equality / inequality

    arithmetic_expression = add_op? term (add_op term)*

    add_op = plus / minus / dot_plus / dot_minus

    term = factor (mul_op factor)*

    mul_op = times / divide / dot_times / dot_divide

    factor = primary ( (exp / dot_exp) primary)?

    primary = unsigned_number / string / false / true
        / ((name / der / initial) function_call_args)
        / component_reference
        / (lparen output_expression_list rparen)
        / (lbracket expression_list
            ( semicolon expression_list )* rbracket)
        / (lbrace function_arguments rbrace)
        / end

    name = period ? ident (period ident)*

    component_reference = (period ident array_subscripts?)+

    function_call_args = lparen function_arguments? rparen

    function_arguments = function argument
        ((comma function_arguments) / (for for_indices) /
        named_arguments)

    named_arguments = named_argument (comma named_arguments)?

    named_argument = ident equals function_argument

    function_argument = function name
        ((lparen named_arguments? rparen) / expression)

    output_expression_list = expression? (comma expression?)*

    expression_list = expression (comma expression)*

    array_subscripts = lbracket subscript (comma subscript)* rbracket

    subscript = colon / expression

    comment = string_comment annotation?

    string_comment = string ( plus string)*

    annotation = annotation class_modification

    #===============================================================
    # BASIC
    #===============================================================
    _ = ~'\s*'
    equals = '='_
    assign = ':='_
    semicolon = ';'_
    lparen = '('_
    rparen = ')'_
    lbracket = '{'_
    rbracket = '}'_
    colon = ':'_
    comma = ','_
    double_quote = '"'_
    single_quote = "'"_
    lbrace = '{'_
    rbrace = '}'_
    period = '.'_
    plus = '+'_
    dot_plus = '.+'_
    minus = '-'_
    dot_minus = '.-'_
    times = '*'_
    dot_times = '.*'_
    divide = '/'_
    dot_divide = './'_
    exp = '^'_
    dot_exp = '.^'_
    less_than = '<'_
    less_than_or_equal = '<='_
    greater_than = '>'_
    greater_than_or_equal = '>='_
    equality = '=='_
    inequality = '<>'_
    ident = (nondigit ( digit / nondigit )*_) / q_ident
    q_ident = single_quote (q_char / s_escape)+ single_quote
    string = double_quote (s_char/s_escape)* double_quote
    nondigit = ~'[_a-zA-Z]'
    s_char = ~r'[^"\\]*'u
    q_char = (nondigit/digit/~r'[#$%&()*+,-./:;<>=?@[]^\{}|~ ')
    s_escape = ~r'[\'"\?\\\a\b\f\n\r\t\v]'
    digit = ~'[0-9]'
    unsigned_integer = digit+
    unsigned_number = unsigned_integer ( '.' unsigned_integer?)?
        (('e'/'E') ('+'/'-')? unsigned_integer)?

    #===============================================================
    # KEYWORDS
    #===============================================================
    algorithm = 'algorithm'_
    and = 'and'_
    annotation = 'annotation'_
    assert = 'assert'_
    block = 'block'_
    break = 'break'_
    class = 'class'_
    connect = 'connect'_
    connector = 'connector'_
    constant = 'constant'_
    constrainedby = 'constrainedby'_
    der = 'der'_
    discrete = 'discrete'_
    each = 'each'_
    else = 'else'_
    elseif = 'elseif'_
    elsewhen = 'elsewhen'_
    encapsulated = 'encapsulated'_
    end = 'end'_
    enumeration = 'enumeration'_
    equation = 'equation'_
    expandable = 'expandable'_
    extends = 'extends'_
    external = 'external'_
    false = 'false'_
    final = 'final'_
    flow = 'flow'_
    for= 'for'_
    function = 'function'_
    if = 'if'_
    import = 'import'_
    impure = 'impure'_
    in = 'in'_
    initial = 'initial'_
    inner = 'inner'_
    input = 'input'_
    initial = 'initial'_
    loop = 'loop'_
    model = 'model'_
    not = 'not'_
    operator = 'operator'_
    or = 'or'_
    outer = 'outer'_
    output = 'output'_
    package = 'package'_
    parameter = 'parameter'_
    partial = 'partial'_
    protected = 'protected'_
    public = 'public'_
    pure = 'pure'_
    record = 'record'_
    redeclare = 'redeclare'_
    replaceable = 'replaceable'_
    return = 'return'_
    stream = 'stream'_
    then = 'then'_
    true = 'true'_
    type = 'type'_
    when = 'when'_
    while = 'while'_
    within = 'within'_
    """)
