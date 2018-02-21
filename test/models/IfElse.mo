model IfElse
	input Real x;
	output Real y1;
	output Real y2;
	output Real y3;
	parameter Real y_max = 10;
equation
	y1 = (if x > 0 then 1 else 0) * y_max;
	if x > 1 then
		y2 = y_max;
		y3 = 100;
	elseif x > 2 then
		y2 = y_max + 1;
		y3 = 1000;
	else
		y2 = 0;
		y3 = 10000;
	end if;
end IfElse;