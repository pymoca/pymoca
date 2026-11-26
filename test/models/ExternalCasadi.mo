package ExternalCasadi
  type Color = enumeration(Red, Green, Blue);
  constant Integer n = 2;

  pure function weightedSum
    input Real a[:, :];
    input Real w[size(a, 2)];
    output Real y[size(a, 1)];
  external "C" weightedSum(a, size(a, 1), size(a, 2), w, y)
    annotation(Library = "ExternalCasadi");
  end weightedSum;

  pure function scaleShift
    input Real x;
    input Integer k;
    input Boolean negate;
    input Color c;
    output Real y;
    output Integer m;
  external "C"
    annotation(Library = "ExternalCasadi");
  end scaleShift;

  pure function describe
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

  pure function sumWithWork
    input Real x[:];
    output Real y;
  protected
    Real work[n] = {1, 10};
  external "C" y = sumWithWork(x, size(x, 1), work, n)
    annotation(Library = "ExternalCasadi");
  end sumWithWork;

  impure function lookupScale
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

  pure function cube
    input Real x;
    input Integer k;
    output Real y;
  external "C" y = cube(x, k)
    annotation(Library = "ExternalCasadi");
    annotation(derivative = cubeDer);
  end cube;

  pure function cubeDer
    input Real x;
    input Integer k;
    input Real der_x;
    output Real der_y;
  external "C" der_y = cubeDer(x, k, der_x)
    annotation(Library = "ExternalCasadi");
    annotation(derivative(order = 2) = cubeDer2);
  end cubeDer;

  pure function cubeDer2
    input Real x;
    input Integer k;
    input Real der_x;
    input Real der2_x;
    output Real der2_y;
  external "C" der2_y = cubeDer2(x, k, der_x, der2_x)
    annotation(Library = "ExternalCasadi");
  end cubeDer2;

  pure function addInPlace
    input Real x;
    output Real y = 2.5;
    output Real w[2] = {x, 2 * x};
  external "C" addInPlace(x, y, w, size(w, 1))
    annotation(Library = "ExternalCasadi");
  end addInPlace;

  pure function addTwo
    input Real a;
    input Real b = 2 * a;
    output Real y;
  external "C99" y = addTwo(a, b)
    annotation(Library = "ExternalCasadi");
  end addTwo;

  pure function addNegative
    input Real a;
    output Real y;
  external "C" y = addTwo(a, -1.5)
    annotation(Library = "ExternalCasadi");
  end addNegative;

  pure function dotTable
    input Real x[3];
    output Real y;
  protected
    constant Real table[3] = {1, 2, 3};
    constant Integer n = 3;
  external "C" y = dotTable(x, table, n)
    annotation(Library = "ExternalCasadi");
  end dotTable;

  pure function addIndices
    input Real a[:, :];
    output Real b[size(a, 1), size(a, 2)];
  external "C" addIndices(a, size(a, 1), size(a, 2), b)
    annotation(Library = "ExternalCasadi", arrayLayout = "columnMajor");
  end addIndices;

  pure function lengths
    input String names[:];
    output Real y[size(names, 1)];
  external "C" lengths(names, size(names, 1), y)
    annotation(Library = "ExternalCasadi");
  end lengths;

  pure function square
    input Real x;
    output Real y;
  external "C" y = square(x)
    annotation(Library = "ExternalCasadi");
    annotation(derivative = squareDer);
  end square;

  function squareDer
    input Real x;
    input Real der_x;
    output Real der_y;
  algorithm
    der_y := 2 * x * der_x;
  end squareDer;

  pure function scaled
    input Real x;
    input Real k;
    output Real y;
  external "C" y = scaled(x, k)
    annotation(Library = "ExternalCasadi");
    annotation(derivative(zeroDerivative = k) = scaledDerK, derivative = scaledDer);
  end scaled;

  function scaledDerK
    input Real x;
    input Real k;
    input Real der_x;
    output Real der_y;
  algorithm
    der_y := k * der_x;
  end scaledDerK;

  function scaledDer
    input Real x;
    input Real k;
    input Real der_x;
    input Real der_k;
    output Real der_y;
  algorithm
    der_y := k * der_x + x * der_k;
  end scaledDer;

  function nameLength
    input Real x;
    input String name = "abc";
    output Real y;
  protected
    Real ys[1];
  algorithm
    ys := lengths({name});
    y := x * ys[1];
  end nameLength;

  model Smooth
    input Real u;
    parameter Real p = 3;
    Real c = cube(u, 2);
    Real r = checkedSqrt(u);
    Real s = square(u);
    Real k = scaled(u, 2.0);
    Real kp = scaled(u, p);
  end Smooth;

  model StringArray
    input Real u;
    Real y[2] = lengths({"ab", "c"});
  end StringArray;

  model ForwardedString
    input Real u;
    Real y = nameLength(u);
  end ForwardedString;

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
    Real a;
    Real b[2];
    Real t = addTwo(1.0);
    Real n = addNegative(1.0);
    Real d = dotTable({1, 1, 1});
    Real e[2, 3] = addIndices(zeros(2, 3));
  equation
    (z, m) = scaleShift(1.5, 2, true, Color.Blue);
    (a, b) = addInPlace(1.0);
  end Calls;
end ExternalCasadi;
