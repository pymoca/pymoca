model IfElse
	input Real x;
	output Real y1;
	output Real y2;
	parameter Real y_max = 10;
equation
	y1 = (if x > 0 then 1 else 0) * y_max;
	if x > 1 then
		y2 = y_max;
	else
		y2 = 0;
	end if;
end IfElse;