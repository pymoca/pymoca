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

  function interpolateMyTable
    input MyTable interpolTable;
    input Real u;
    output Real y;
  external "C"
    y = interpolateMyTable(interpolTable, u);
  end interpolateMyTable;

  function interpolateMyTableFunc
    input Real[5] vals;
    input Real u;
    output Real y;
  protected
    MyTable myTable = MyTable(vals);
  algorithm
    y := interpolateMyTable(myTable, u);
  end interpolateMyTableFunc;

  model Table
    parameter Real vals[5] = {5, 3, 1, 9, 6};
    parameter MyTable myTable = MyTable(vals);
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

  model FreeTable
    parameter Real vals[5];
    parameter MyTable myTable = MyTable(vals);
    Real y = interpolateMyTable(myTable, 0.7);
  end FreeTable;
end ExternalObjectCasadi;
