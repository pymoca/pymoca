package ExternalObjectCasadi
  class MyTable
    extends ExternalObject;

    function constructor
      input Real[:] vals;
      output MyTable outTable;
    external "C"
      outTable = initMyTable(vals, size(vals, 1));
    end constructor;

    function destructor
      input MyTable inTable;
    external "C"
      closeMyTable(inTable);
    end destructor;
  end MyTable;

  class Shifted
    extends ExternalObject;

    function constructor
      input MyTable table;
      input Integer shift;
      output Shifted shifted;
    external "C"
      shifted = initShifted(table, shift);
    end constructor;

    function destructor
      input Shifted shifted;
    external "C"
      closeShifted(shifted);
    end destructor;
  end Shifted;

  pure function interpolateShifted
    input Shifted shifted;
    input Real u;
    output Real y;
  external "C"
    y = interpolateShifted(shifted, u);
  end interpolateShifted;

  function interpolateShiftedFunc
    input Real[5] vals;
    input Integer shift;
    input Real u;
    output Real y;
  protected
    MyTable myTable = MyTable(vals);
    Shifted shifted = Shifted(myTable, shift);
  algorithm
    if u > 4 then
      y := interpolateShifted(shifted, 4);
    else
      y := interpolateShifted(shifted, u);
    end if;
  end interpolateShiftedFunc;

  pure function interpolateMyTable
    input MyTable interpolTable;
    input Real u;
    output Real y;
  external "C"
    y = interpolateMyTable(interpolTable, u);
    annotation(derivative(noDerivative = interpolTable) = interpolateMyTableDer);
  end interpolateMyTable;

  pure function interpolateMyTableDer
    input MyTable interpolTable;
    input Real u;
    input Real der_u;
    output Real der_y;
  external "C"
    der_y = interpolateMyTableDer(interpolTable, u, der_u);
  end interpolateMyTableDer;

  function interpolateMyTableFunc
    input Real[5] vals;
    input Real u;
    output Real y;
  protected
    MyTable myTable = MyTable(vals);
  algorithm
    y := interpolateMyTable(myTable, u);
  end interpolateMyTableFunc;

  pure function stringLength
    input String s;
    output Integer n;
  external "C"
    n = stringLength(s);
  end stringLength;

  function isBlank
    input String s;
    output Boolean blank;
  algorithm
    blank := stringLength(s) == 0;
  end isBlank;

  model Table
    parameter Real vals[5] = {5, 3, 1, 9, 6};
    parameter Boolean scaled = false;
    parameter String scale = "";
    parameter MyTable myTable = MyTable(if scaled and not isBlank(scale) then 2 * vals else vals);
    MyTable reversed = MyTable({6, 9, 1, 3, 5});
    input Real u;
    Real y = interpolateMyTable(myTable, u);
    Real z = interpolateMyTable(myTable, 3.5);
    Real w = interpolateMyTable(reversed, u);
  end Table;

  model TableInFunction
    parameter Real vals[5] = {5, 3, 1, 9, 6};
    input Real u;
    Real y = interpolateMyTableFunc(vals, u);
  end TableInFunction;

  model ShiftedInFunction
    parameter Real vals[5] = {5, 3, 1, 9, 6};
    parameter Integer shift = 2;
    input Real u;
    Real y = interpolateShiftedFunc(vals, shift, u);
  end ShiftedInFunction;

  model FreeTable
    parameter Real vals[5];
    parameter MyTable myTable = MyTable(vals);
    Real y = interpolateMyTable(myTable, 0.7);
  end FreeTable;
end ExternalObjectCasadi;
