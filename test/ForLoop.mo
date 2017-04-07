model ForLoop
	parameter Integer n = 10;
	Real x[n];
  Real b;
equation
	for i in 1:n loop
    	x[i] = i+b;
    end for;
end ForLoop;
