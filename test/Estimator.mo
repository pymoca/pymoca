model Estimator "a position estimator"
    Real x "position";
    output Real y;
equation
	der(x) = -x "the deriv of position is velocity";
	y = x;
end Estimator;
