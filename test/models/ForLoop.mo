model ForLoop
	parameter Integer n = 10;
	Real x[n];
    Real y[n];
    Real z[n];
    Real u[n, 2];
    Real v[2, n];
    Real w[2, n];
    Real b;
    Real s[n];
    Real Arr[2, 2];
    Real ArrEmtpy[2, 0];
    Real VecEmtpy[0];
equation
	for i in 1:n loop
    	x[i] = i+b;
        w[1, i] = i;
        w[2, i] = 2*i;
        u[i, :] = ones(2);
        v[:, i] = ones(2);
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
    for l in 1:n loop
        der(s[l]) = 1.0;
    end for;
    for m in 1:2 loop
        Arr[m, 2] = 2.0;
        Arr[m, 1] = 1.0;
        ArrEmtpy[m, :] = 2.0 * VecEmtpy[:];
    end for;
end ForLoop;
