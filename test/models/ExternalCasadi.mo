package ExternalCasadi
  type Color = enumeration(Red, Green, Blue);
  constant Integer n = 2;

  function weightedSum
    input Real a[:, :];
    input Real w[size(a, 2)];
    output Real y[size(a, 1)];
  external "C" weightedSum(a, size(a, 1), size(a, 2), w, y)
    annotation(Library = "ExternalCasadi");
  end weightedSum;

  function scaleShift
    input Real x;
    input Integer k;
    input Boolean negate;
    input Color c;
    output Real y;
    output Integer m;
  external "C"
    annotation(Library = "ExternalCasadi");
  end scaleShift;

  function describe
    input String name;
    input Real x;
    output String s;
  external "C" s = describe(name, x)
    annotation(Library = "ExternalCasadi");
  end describe;

  function checkedSqrt
    input Real x;
    output Real y;
  external "C" y = checkedSqrt(x)
    annotation(Library = "ExternalCasadi");
  end checkedSqrt;

  function sumWithWork
    input Real x[:];
    output Real y;
  protected
    Real work[n] = {1, 10};
  external "C" y = sumWithWork(x, size(x, 1), work, n)
    annotation(Library = "ExternalCasadi");
  end sumWithWork;

  model Calls
    Real y[2] = weightedSum({{1, 2}, {3, 4}}, {10, 1});
    Real z;
    Integer m;
    String s = describe("x", 1.5);
    Real r = checkedSqrt(4.0);
    Real w = sumWithWork({1, 2});
  equation
    (z, m) = scaleShift(1.5, 2, true, Color.Blue);
  end Calls;
end ExternalCasadi;
