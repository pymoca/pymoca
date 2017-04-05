model ForLoop
	parameter Integer n = 10;
	Real[n] x;
  Real b;
equation
	for i in 1:n loop
    	x[i] = i+b;
    end for;
end ForLoop;
