model NestedAttributes
    parameter Real p1 = 1;
	parameter Real p = 2 * p1;
	Real s = 3 * p;
end NestedAttributes;

model Attributes
    NestedAttributes nested;
	Integer int(min = -5, max = 10);
	Boolean bool;
	Real real(start = 20.0);
	input Real i1(fixed = true);
	input Real i2(fixed = false);
	input Real i3;
	output Real i4;
	constant Real cst = 1;
	parameter Real prm = 2;
	Real test_state = real; // Generates an additional equation.
protected
	output Real protected_variable;
equation
	i4 = i1 + i2 + i3;
	der(real) = i1 + (if bool then 1 else 0) * int;
	protected_variable = i1 + i2;
end Attributes;