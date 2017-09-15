model StateAnnotator
	Real x;
	Real y;
	Real z;
equation
	der(x + y) = 1;
	der(x * y) = 2;
	der(x / y) = 3;
	der(x^2) = 4;
	der(z) = 5;
end StateAnnotator;