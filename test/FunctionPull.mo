within Level1.Level2.Level3;

function f
  input Real x;
  output Real y;
algorithm
  y := TestPackage.times2(x) * Level2.Level3.TestPackage.square(2.0);
end f;

package TestPackage
	function times2
	  input Real x;
	  output Real y;
	algorithm
	  y := x * 2.0;
	end times2;

	function square
	  input Real x;
	  output Real y;
	algorithm
	  y := x * x;
	end square;

	function not_called
	  input Real x;
	  output Real y;
	algorithm
	  y := x * x * x;
	end square;
end TestPackage;

model Function5
  Real a,b;
equation
  a = f(b);
end Function5;
