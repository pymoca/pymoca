package ExternalObjectTest
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
    annotation(derivative(noDerivative = interpolTable) = interpolateMyTableDer);
  end interpolateMyTable;

  function interpolateMyTableDer
    input MyTable interpolTable;
    input Real u;
    input Real der_u;
    output Real der_y;
  external "C"
    der_y = interpolateMyTableDer(interpolTable, u, der_u);
    annotation(derivative(order = 2, zeroDerivative = interpolTable) = interpolateMyTableDer2);
  end interpolateMyTableDer;

  function interpolateMyTableDer2
    input MyTable interpolTable;
    input Real u;
    input Real der_u;
    input Real der2_u;
    output Real der2_y;
  external "C"
    der2_y = interpolateMyTableDer2(interpolTable, u, der_u, der2_u);
  end interpolateMyTableDer2;

  function interpolateMyTableFunc
    input Real[:] vals;
    input Real val;
    output Real y;
  protected
    MyTable myTable = MyTable(vals);
  algorithm
    y := interpolateMyTable(myTable, val);
  end interpolateMyTableFunc;

  model Table
    MyTable myTable = MyTable({5, 3, 1, 9, 6});
    Real y = interpolateMyTable(myTable, 0.7);
  end Table;

  model TableInFunction
    parameter Real[:] vals = {5, 3, 1, 9, 6};
    Real y = interpolateMyTableFunc(vals, 0.7);
  end TableInFunction;
end ExternalObjectTest;
