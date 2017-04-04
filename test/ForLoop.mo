model ForLoop
	parameter Integer n = 10;
	Real[n] x;
equation
	for i in 1:n loop
    	der(x[i]) = i;
    end for;
end ForLoop;