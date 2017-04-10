model ForLoop
	parameter Integer n = 10;
	Real x[n];
    Real y[n];
    Real z[n];
    Real b;
equation
	for i in 1:n loop
    	x[i] = i+b;
    end for;
    for j in 1:5 loop
    	y[j] = 0;
    end for;
    for j in 6:10 loop
    	y[j] = 1;
    end for;
    for k in 1:n/2 loop
    	z[k] = 2;
    	z[k+5] = 1;
    end for;
end ForLoop;
