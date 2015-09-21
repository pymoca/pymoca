/** Derived from http://smc.sourceforge.net/
 */
grammar SMC;

smc_main : source? start_state class_name header_file? include_file*
package_name* smc_import* declare* access* map+;

source : '%{' raw_code '%}';

start_state : '%start' ID;

class_name : '%class' ID;

header_file : '%header' raw_code_line;

include_file : '%include' raw_code_line;

package_name : '%package' ID;

smc_import : '%import' raw_code_line;

declare : '%declare' raw_code_line;

access : '%access' raw_code_line;

map : '%map' ID '%%' states '%%';

states : ID entry? exit? '{' transitions* '}';

entry : 'Entry {' actions* '}';

exit : 'Exit {' actions '}';

transitions : ID transition_args? guard? next_state '{' actions '}';

transition_args : '(' parameters ')';

parameters : parameter |
              parameter ',' parameters;

parameter : ID ':' raw_code;

guard : '[' raw_code ']';

next_state : ID |
              'nil' |
              push_transition |
              pop_transition;

push_transition : ID '/' 'push(' ID ')' |
                   'nil/push(' ID ')' |
                   'push(' ID ')';

pop_transition : 'pop' |
                  'pop(' ID? ')' |
                  'pop(' ID ',' pop_arguments* ')';

pop_arguments : raw_code |
                 raw_code ',' pop_arguments;

actions : dotnet_assignment |
           action |
           action actions;

dotnet_assignment : ID '=' raw_code ';';

action : ID '(' arguments* ');';

arguments : raw_code |
             raw_code ',' arguments;

/** "a numeral [-]?(.[0-9]+ | [0-9]+(.[0-9]*)? )" */
NUMBER      :   '-'? ('.' DIGIT+ | DIGIT+ ('.' DIGIT*)? ) ;
fragment
DIGIT       :   [0-9] ;

/** "any double-quoted string ("...") possibly containing escaped quotes" */
STRING      :   '"' ('\\"'|.)*? '"' ;

/** "Any string of alphabetic ([a-zA-Z\200-\377]) characters, underscores
 *  ('_') or digits ([0-9]), not beginning with a digit"
 */
ID          :   LETTER (LETTER|DIGIT)*;
fragment
LETTER      :   [a-zA-Z\u0080-\u00FF_] ;

// Reads in code verbatim until end-of-line is reached.
raw_code_line : (.)*? '\n\r\f';

// Read in code verbatim.
raw_code : (.)*?;

COMMENT     :   '/*' .*? '*/'       -> skip ;
LINE_COMMENT:   '//' .*? '\r'? '\n' -> skip ;
WS          :   [ \t\n\r]+ -> skip ;
