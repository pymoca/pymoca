model Attributes
	Integer int(min = -5, max = 10);
	Boolean bool;
	Real real(start = 20.0);
	input Real i1(fixed = true);
	input Real i2(fixed = false);
	input Real i3;
	output Modelica.SIunits.Temperature i4;
protected
	output Real protected_variable;
equation
	i4 = i1 + i2 + i3;
	der(real) = i1 + (if bool then 1 else 0) * int;
	protected_variable = i1 + i2;
end Attributes;