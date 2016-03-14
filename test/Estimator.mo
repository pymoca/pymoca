model Estimator "a position estimator"
	parameter Real v_x=1 "velocity";
    Real x "position";
equation
	der(x) = v_x "the deriv of position is velocity";
end BouncingBall;
