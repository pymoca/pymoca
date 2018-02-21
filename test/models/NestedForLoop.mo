model NestedForLoop
	parameter Integer n = 10;
	parameter Integer m = 20;
	Real[n,m] x;
equation
	for i in 1:n loop
		for j in 1:m loop
	    	x[i,j] = i+j;
	    end for;
    end for;
end NestedForLoop;
