model ParameterAttributes
	parameter String string_parameter = "test_p";
	constant String string_constant = "test_c";
	parameter Real min_ = 3.0;
    parameter Real max_;
	Real a(min=min_, max=10.0);
	Real b(min=0.0,  max=max_);
	Real c;
	Real[3] d(each min=-max_, each max=max_);
equation
	a = b;
	c = a * b;
	der(c) = 0.01;
end ParameterAttributes;
