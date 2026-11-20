package FunctionBindingTest
  function f
    input Real x;
    input Real k = 2;
    output Real y;
  protected
    Real a = k * x;
  algorithm
    y := a + 1;
  end f;

  model M
    Real y = f(3.0, 3.0);
  end M;
end FunctionBindingTest;
