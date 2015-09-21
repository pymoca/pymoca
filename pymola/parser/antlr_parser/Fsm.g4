grammar Fsm;

fsm_main: fsm_state+;

fsm_state : 'STATE' ID '{'
	fsm_transition*
	'}'
	;

fsm_transition  : 'TRANSITION' ID '{'
	fsm_guard*
	'}';

fsm_expr:   fsm_expr fsm_op=('*'|'/') fsm_expr  # MulDiv
    |   fsm_expr fsm_op=('+'|'-') fsm_expr      # AddSub
    |   INT                         			# int
    |   ID                          			# id
    |   '(' fsm_expr ')'                		# parens
    ;

fsm_guard  : 'GUARD' ID;

MUL :   '*' ; // assigns token name to '*' used above in grammar
DIV :   '/' ;
ADD :   '+' ;
SUB :   '-' ;
ID  :   [a-zA-Z]+ ;      // match identifiers
INT :   [0-9]+ ;         // match integers
WS  :   [ \r\n\t]+ -> skip ; // toss out whitespace
