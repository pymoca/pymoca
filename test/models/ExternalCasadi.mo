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

  function lookupScale
    input String key;
    input Real x;
    output Real y;
  external "C" y = lookupScale(key, x)
    annotation(Library = "ExternalCasadi");
  end lookupScale;

  function twiceSqrt
    input Real x;
    output Real y;
  algorithm
    y := 2 * checkedSqrt(x);
  end twiceSqrt;

  function cube
    input Real x;
    input Integer k;
    output Real y;
  external "C" y = cube(x, k)
    annotation(Library = "ExternalCasadi");
    annotation(derivative = cubeDer);
  end cube;

  function cubeDer
    input Real x;
    input Integer k;
    input Real der_x;
    output Real der_y;
  external "C" der_y = cubeDer(x, k, der_x)
    annotation(Library = "ExternalCasadi");
    annotation(derivative(order = 2) = cubeDer2);
  end cubeDer;

  function cubeDer2
    input Real x;
    input Integer k;
    input Real der_x;
    input Real der2_x;
    output Real der2_y;
  external "C" der2_y = cubeDer2(x, k, der_x, der2_x)
    annotation(Library = "ExternalCasadi");
  end cubeDer2;

  model Smooth
    input Real u;
    Real c = cube(u, 2);
    Real r = checkedSqrt(u);
  end Smooth;

  model Numeric
    input Real u;
    parameter String key = "double";
    Real y[2] = weightedSum({{1, 2}, {3, u}}, {10, 1});
    Real z;
    Integer m;
    Real r = checkedSqrt(u);
    Real w = sumWithWork({u, 2});
    Real d = lookupScale(key, u);
    Real q = twiceSqrt(u);
  equation
    (z, m) = scaleShift(u, 2, true, 3);
  end Numeric;

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
