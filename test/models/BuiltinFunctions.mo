model BuiltinFunctions
	input Real x;
	output Real y;
	output Real z;
    output Real w;
    output Real u;
equation
	y = sin(time);
	z = cos(x);
	w = min(y, z);
	u = abs(w);
end BuiltinFunctions;