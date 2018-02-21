model BaseA
	Real x;
	parameter Real k = 1.0;
equation
	der(x) = k * x;
end BaseA;

model BaseB
	Real y;
end BaseB;

model Sub
	extends BaseA(k = -1.0, x(max = 30.0));
	extends BaseB;
equation
	x + y = 3.0;
end Sub;