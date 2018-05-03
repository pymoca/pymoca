model ParameterAttributes
	parameter Real min_;
    parameter Real max_;
	Real a(min=min_, max=10.0);
	Real b(min=0.0,  max=max_);
	Real c;
equation
	a = b;
	c = a * b;
	der(c) = 0.01;
end ParameterAttributes;
