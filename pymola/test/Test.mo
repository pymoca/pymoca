//
// Modelica test suite
//
// by Pontus Lidman 97-08-05
//
// This test suite covers every rule in the modelica
// grammar completely. I estimate that if code is
// generated correctly for this file, chances are good
// that the parser produces correct output from correct
// input. Nothing can be said about incorrect input.
//
// All permutations of repetitions are not covered.
//

//--------------------------------------------------
// Test different kinds of model_specification
//--------------------------------------------------

// class_definition is tested in the cases for classes,
// import is the alternative left.
//import "filename";

//--------------------------------------------------
// Test different kinds of class_definition
//--------------------------------------------------


//--------------------------------------------------
// without composition, no array or modification
//--------------------------------------------------

record Class3 = Identifier;

//--------------------------------------------------
// without composition, no array, modification
//--------------------------------------------------


connector Class4 = Identifier(Unit="m",sven=7.2);

//--------------------------------------------------
// without composition, array, no modification
//--------------------------------------------------

block Class5 = Identifier[:];

//--------------------------------------------------
// without composition, array, modification
//--------------------------------------------------

type Class6 = Identifier[:] (Unit="m", sven="7");
//--------------------------------------------------
// non-external function
//--------------------------------------------------

//function F1
//  Real simple_composition;
//end F1;

//--------------------------------------------------
// external function
//--------------------------------------------------

//external function F2
//  Real simple_composition;
//end F2;
//
//--------------------------------------------------
// with partial and comment, with composition
//--------------------------------------------------

partial class Class1 "Test"
  Real simple_composition;
end Class1;
//
//--------------------------------------------------
// without partial and comment, with composition
//--------------------------------------------------

model Class2
  Real simple_composition;
end Class2;
//
//--------------------------------------------------
// without partial and comment, no end identifier
//--------------------------------------------------

model Class7
  Real simple_composition;
end Class7;
//
//--------------------------------------------------
// Test different kinds of composition
//--------------------------------------------------

//--------------------------------------------------
// Element list with one element
//--------------------------------------------------

model Comp1
  Real simple_element;
end Comp1;


//--------------------------------------------------
// Element list with several elements, 
// some permutations of possible elements.
//--------------------------------------------------

model Comp2
  Real default_element;

protected 
  Real protected1;
  Real protected2;

public
  Real public1;

equation
  x=y;
  y=z;

algorithm
  x:=y;

public
  Real public3;
  Real public4;

equation
  z=3;

protected
  Real protected3;

end Comp2;

//--------------------------------------------------
// Test different element_lists
//--------------------------------------------------

//--------------------------------------------------
// Empty element list
//--------------------------------------------------

model Element1

end Element1;

//--------------------------------------------------
// Several different elements
//--------------------------------------------------

model Element2

  Real simple_element1;
  //annotation (Simple=Modification);
  Real simple_element2;
end Element2;

//--------------------------------------------------
// Test different elements
//--------------------------------------------------

model Element3
  // final & virtual class definition
  final replaceable class SimpleClass1=Real;
  // non-final, virtual class definition
  replaceable model SimpleClass2=Real;
  // final, non-virtual class definition
  final record SimpleClass3=Real;
  // extends clause
  extends SimpleExtends;
  // component clause
  final flow parameter Real SimpleComponentClause;
end Element3;
////
//--------------------------------------------------
// Test different extends clauses
//--------------------------------------------------

//--------------------------------------------------
// Extends with and without class_modification
//--------------------------------------------------

model Extends_class
  extends Class1;
  extends Class2(Simple=Modification);
end Extends_class;

//--------------------------------------------------
// Test the component clause
//--------------------------------------------------

model ComponentClause
  constant Real a,b;
end ComponentClause;


//--------------------------------------------------
// Test type prefixes along with type specifier
//--------------------------------------------------

model TypePrefix
  Real a;
  flow Real b;
  flow parameter Real c;
  constant Real d;
  flow constant input Real e;
  parameter input Real f;
end TypePrefix;

//--------------------------------------------------
// Test component list
//--------------------------------------------------

model ComponentList
  Real a;
  AnotherType b,c,d;
end ComponentList;

//--------------------------------------------------
// Test component declaration
//--------------------------------------------------

model ComponentDeclaration
  Real x "comment";
  Real y,z "z-comment",g "g-comment";
end ComponentDeclaration;

//--------------------------------------------------
// Test declaration and array declaration
//--------------------------------------------------

model Declaration
  // Trivial
  Real x;
  // Array declaration
  Real y[3];
  // Modification
  Real z(Unit="Ohm");
  // Array and modification
  Real v[7](Unit="Ohm");
end Declaration;

//--------------------------------------------------
// Test subscript list
//--------------------------------------------------

model Subscriptlist
  Real x[7];
  Real y[1,2];
  Real z[1,2,3];
end Subscriptlist;
//
//--------------------------------------------------
// Test subscript
//--------------------------------------------------

model Subscript_class
  // single colon
  Real x[:];
  // colon to left
  Real y[1:7];
  // colon to right
  Real z[5:9];
  // colon in between
  Real v[3:9];
end Subscript_class;

//--------------------------------------------------
// Test modification
//--------------------------------------------------


model Modification

  // class_specialization with no = expression
  Real A(Unit="Ohm");

  // class specialization with = expression
  Real B(Unit="Hz")=50;

  // just = expression
  Real C=50;
end Modification;
//
////--------------------------------------------------
//// Test class modification
////--------------------------------------------------
//
type ClassModification=Real(Unit="Ohm");
//

//--------------------------------------------------
// Test argument list
//--------------------------------------------------

// single argument
type ArgumentList1 = Real(Unit="Ohm");

// several arguments
type ArgumentList2 = Real(Unit="A", Connection=Sum, X=Y);

//--------------------------------------------------
// Test argument
//--------------------------------------------------

// argument is element_modification
type Argument1=Real(final Identifier (Unit="Ohm"));

// argument is element_redeclaration
//type Argument2=Real(redeclare extends AnotherType);

//--------------------------------------------------
// Test element modification
//--------------------------------------------------

// with final
type ElementModification1=Real(final Identifier=SimpleModification);

// without final, with array declaration
//type ElementModification2=Real(Identifier[9]=SimpleModification);

// with final and array declaration
//type ElementModification3=Real(final Identifier[9]=SimpleModification);

//--------------------------------------------------
// Test element_redeclaration and component_clause1
//--------------------------------------------------


// extends clause with final
//type ElementRedeclaration1=
//  Real(redeclare final extends Type);

// extends clause without final
//type ElementRedeclaration2=
//  Real(redeclare extends Type);

// with class definition (& final for good measure)
//type ElementRedeclaration3=
//  Real(redeclare final class SimpleClass equation x=y; end SimpleClass);

// with component_clause1
type ElementRedeclaration4=
  Real(redeclare constant Real x);


//--------------------------------------------------
// Test equation clause
//--------------------------------------------------

class EquationClause

equation
  x=y;
  //annotation (a=b);
  //annotation (u=v);
  n=m;
end EquationClause;

//--------------------------------------------------
// Test algorithm clause
//--------------------------------------------------

class AlgorithmClause

algorithm
  x:=y;
  //annotation (a=b);
  //annotation (u=v);
  n:=m;
  // while_clause
  while x loop x:=y; end while;
end AlgorithmClause;
//
//--------------------------------------------------
// Test equation
//--------------------------------------------------
  
class Equation_class

equation
  // simple_expression
//  x;
  // simple_expression=expression
  y=z;
  // simple_expression:=expression
  u=v;
  // conditional_equation
  if x then y=z; end if;
  // for_clause
  for x in y:z loop x=y; end for;
end Equation_class;

//--------------------------------------------------
// Test conditional equation
//--------------------------------------------------

class ConditionalEquation

equation
  // the test cases have different number of equations in
  // different positions

  // if, then, endif, empty then
// not allowed in objectmath!
//  if x then end if;

  if x then y=z; end if;

  // if, then, else, endif, one then, one else
  if x then y=z; else z=z; end if;

  // if, then, elseif, endif, two then, empty elseif-then
// not allowed in objectmath!
//  if x then y; a; elseif u then end if;
  
  // if, then, elseif, else, endif, two elseif-then, two else
  if x then y=z; elseif u then v=z; w=z; else z=z; x=z; end if;

  // if, then, elseif, elseif, endif, one elseif-then, two else
  if x then y=z; elseif u then v=z; elseif a then b=z; c=z; else j=z; k=z; end if;

  // if, then, elseif, elseif, else, endif, empty else
// not allowed in objectmath!
//  if x then y; elseif u then v; elseif a then b; else end if;
end ConditionalEquation;


class ForClause

equation
  // without third expression, one equation
  for a in 1:9 loop eq=z; end for;

  // with third expression, several equations
  for a in 1:9:19 loop eq=z; eq2=z2; eq3=z3; end for;

  // without third expression, no equations
// not allowed in objectmath!
//  for a in 1:9 loop end for;
end ForClause;

//--------------------------------------------------
// Test while clause
//--------------------------------------------------

class WhileClause

algorithm
  // no equations
// not allowed in objectmath!
//  while 1 loop end while;
  // one equation
  while 2 loop x:=z; end while;
  // several equations
  while 3 loop y:=z; z:=z; end while;
end WhileClause;

//--------------------------------------------------
// Test expression
//--------------------------------------------------

class Expression_class

equation
  // simple expression
  1=x;
  // if-then-else-expression
  1=if x then y else z;
end Expression_class;

//--------------------------------------------------
// Test simple expression
//--------------------------------------------------

class SimpleExpression

equation
  // no alternatives taken, just a primary
  z=1;

//// test
//   z=1 or 2 or 3;

  // all alternatives taken
  //z=1 or 2 and not (+3 == -4) + (7 >= 5) and (3 <= 6) and 3 <> 7 and (3 > 8)
  //+ (7 < 9) + term  - factor * factor / der(2);

  // and backwards
  //z=der(2) / factor * factor - term + (9 < 8) + (6 > 7) + (9 <> 6) + (4 <= 5)
  //+ (3 >= -4 ) + (2 == +3 ) and 2 or not 1;

end SimpleExpression;

//--------------------------------------------------
// Test primary
//--------------------------------------------------

class Primary

equation
  z=(1);
  z=[2];
  z=9.003e+27;
  z=5.E-19;
  z=false;
  z=true;
  z="string";
  z=name.path(arguments);
  z=component[3].reference[4];
end Primary;

//--------------------------------------------------
// Test name path
//--------------------------------------------------

class NamePath

equation
  // one name
  z=name;
  // several names
  z=name1.name2.name3;
end NamePath;

//--------------------------------------------------
// Test component_reference
//--------------------------------------------------

class ComponentReference

equation
  // one component
  z=a[x];
  // several components
  z=a[1].b[2];
  // several components that seem to make up a name path first
  z=a.b.c.d[7].e.f.g[8].h.i.j;
end ComponentReference;

//--------------------------------------------------
// Test column expression and row expression
//--------------------------------------------------

class ColumnExpression

equation
  z=[1,2,3];
  z=[7];
  z=[1,2,3;4,5,6];
  z=[1;2;3];
  z=[1;2];
  z=[1,2];
end ColumnExpression;

//--------------------------------------------------
// Test function arguments
//--------------------------------------------------

class FunctionArguments

equation
  // one argument
  z=func(arg);
  // several arguments
  z=func(arg1,arg2,arg3);  
  // name_path function name
  z=a.func(b.arg1);
end FunctionArguments;


//--------------------------------------------------
// Test comment and annotation
//--------------------------------------------------

// single comment
class Comment1 "comment"

end Comment1;

// single annotation
//class Comment2 annotation (X=y)
//
//end Comment2;

// comment and annotation
class Comment3 "comment" annotation (X=y)

end Comment3;
