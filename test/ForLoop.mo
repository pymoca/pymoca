model ForLoop
	parameter Integer n = 10;
	Real x[n];
    Real y[n];
    Real z[n];
    Real w[2, n];
    Real b;
equation
	for i in 1:n loop
    	x[i] = i+b;
        w[1, i] = i;
        w[2, i] = 2*i;
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
    for l in 1:0 loop
        z[l] = 1e3;
    end for;
end ForLoop;
