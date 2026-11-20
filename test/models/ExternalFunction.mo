package ExternalFunctionTest
  constant Integer n = 2;

  function explicitCall
    input Real x[:];
    input Integer k = 1;
    output Real y;
  protected
    Real work[n] = zeros(n);
  external "C" y = cExplicit(x, size(x, 1), k, n, work) annotation(
      Library = {"lib1", "lib2"},
      Include = "#include \"explicit.h\"",
      IncludeDirectory = "modelica://ExternalFunctionTest/Resources/C-Sources");
  end explicitCall;

  function defaultCall
    input Real a[:, 2];
    input Real s;
    output Real y;
  external "C";
  end defaultCall;

  function defaultProcedure
    input Real x;
    output Real y1;
    output Real y2[2];
  external;
  end defaultProcedure;

  function derivedCall
    extends defaultCall;
  end derivedCall;

  model M
    Real y1 = explicitCall({1, 2});
    Real y2 = defaultCall({{1, 2}}, 1.0);
    Real y3 = derivedCall({{1, 2}}, 1.0);
    Real z1;
    Real z2[2];
  equation
    (z1, z2) = defaultProcedure(1.0);
  end M;
end ExternalFunctionTest;
