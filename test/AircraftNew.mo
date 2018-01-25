model AircraftNew

    // parameters
    parameter Real g = 9.8;
    parameter Real m = 1;
    parameter Real J_x = 1;
    parameter Real J_y = 1;
    parameter Real J_z = 1;
    parameter Real J_xz = 0;
	parameter Real imu_dt = 0.001 "imu sampling period";
	parameter Real mag_dt = 0.001 "magnetometer sampling period";
    
    // states
    Real U(start=0), V(start=0), W(start=0);
    Real P(start=0), Q(start=0), R(start=0);
    Real p_N(start=0), p_E(start=0), h(start=0);
    Real theta(start=0), phi(start=0), psi(start=0);
    Real gb_x(start=0), gb_y(start=0), gb_z(start=0);
    Real ab_x(start=0), ab_y(start=0), ab_z(start=0);

    // variables
    Real gamma;
    Real t_imu;
    Real t_mag;
    
    // inputs
    input Real F_x(start=0), F_y(start=0), F_z(start=0);
    input Real M_x(start=0), M_y(start=0), M_z(start=0);
    
    // outputs
    output Real gyro_x, gyro_y, gyro_z;
    output Real acc_x, acc_y, acc_z;
    output Real mag_x, mag_y, mag_z;

initial equation
    t_imu = time;
    t_mag = time;
    gamma = J_x*J_z - J_xz*J_xz;
    
equation

    // force equations
    der(U) = R*V - Q*W - g*sin(theta) + F_x/m;
    der(V) = -R*U + P*W + g*sin(phi)*cos(theta) + F_y/m;
    der(W) = Q*U - P*V + g*cos(phi)*cos(theta) + F_z/m;

    // kinematic equations
    der(phi) = P + tan(theta)*(Q*sin(phi) + R*cos(phi));
    der(theta) = Q*cos(phi) - R*sin(phi);
    der(psi) = (Q*sin(phi)  + R*cos(phi))/cos(theta);

    // moment equations
    gamma*der(P) = J_xz*(J_x - J_y + J_z)*P*Q - (J_z*(J_z - J_y) + J_xz*J_xz)*Q*R + J_z*M_x + J_xz*M_z;
    J_y*der(Q) = (J_z - J_x)*P*R - J_xz*(P*P - R*R) + M_y;
    gamma*der(R) = ((J_x - J_y)*J_x + J_xz*J_xz)*P*Q - J_xz*(J_x - J_y + J_z)*Q*R + J_xz*M_x + J_x*M_z;

    // navigation equations
    der(p_N) = U*cos(phi)*cos(psi)
        + V*(-cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi))
        + W*(sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi));
    der(p_E) = U*cos(phi)*cos(psi)
        + V*(-cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi))
        + W*(-sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi));
    der(h) = U*sin(theta) - V*sin(phi)*cos(theta) - W*cos(phi)*cos(theta);

    // gyro bias random walk
    der(gb_x) = 0;
    der(gb_y) = 0;
    der(gb_z) = 0;

    // accel bias random walk
    der(ab_x) = 0;
    der(ab_y) = 0;
    der(ab_z) = 0;

algorithm

    when t_imu - time > imu_dt then

        t_imu := time;

        // gyroscope
        gyro_x := P + gb_x;
        gyro_y := Q + gb_y;
        gyro_z := R + gb_z;

        // accelerometer
        acc_x := F_x/m - g*sin(theta) + ab_x;
        acc_y := F_y/m + g*sin(phi)*cos(theta) + ab_y;
        acc_z := F_z/m + g*cos(phi)*cos(theta) + ab_z;

    end when;

	when t_mag - time > mag_dt then

		t_mag := time;


		// magnetometer
		// TODO, actual equations
		mag_x := phi;
		mag_y := theta;
		mag_z := psi;

	end when;

end AircraftNew;

// vim: set noet ft=modelica fenc=utf-8 ff=unix sts=0 sw=4 ts=4 :
