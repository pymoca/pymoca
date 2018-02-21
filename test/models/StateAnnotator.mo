model StateAnnotator
	Real x;
	Real y;
	Real z;
	Real u[3];
	Real v[3];
	Real w[3];
equation
	der(x + y) = 1;
	der(x * y) = 2;
	der(x / y) = 3;
	der(x^2) = 4;
	der(z) = 5;
	der((x + y) * z) = 4;
	der(1) = 0;
	der(u) = {1, 2, 3};
	der(v[1]) = 4;
	der(v[2]) = 5;
	der(v[3]) = 6;
	for i in 1:3 loop
		der(w[i]) = i + 6;
	end for;
end StateAnnotator;