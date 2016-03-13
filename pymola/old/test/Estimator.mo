model Estimator "a position estimator"
	parameter Real a_x=1, a_y=1, a_z=1;
    Real x, y, z, v_x, v_y , v_z;
equation
	der(x) = v_x;
	der(y) = v_y;
	der(z) = v_z;
	der(v_x) = a_x;
	der(v_y) = a_y;
	der(v_z) = a_z;
end BouncingBall;
