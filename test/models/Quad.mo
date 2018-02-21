model Quad
    Real F_x, F_y, F_z;
    Real M_x, M_y, M_z;
    Real phi, theta, psi;
    Real P, Q, R;
    Real x, y, z;
    Real U, V, W;
    parameter Real J_x=1, J_y=1, J_z=1, m=1;
equation
    M_x = -P - phi;
    M_y = -Q - theta;
    M_z = -R - psi;
    F_x = -x;
    F_y = -y;
    F_z = -z;
    der(x) = U;
    der(y) = V;
    der(z) = W;
    -m*V*der(R) + m*W*der(Q) + m*der(U) = F_x;
    m*U*der(R) - m*W*der(P) + m*der(V) = F_y;
    -m*U*der(Q) + m*V*der(P) + m*der(W) = F_z;
    der(phi) = P + Q*sin(phi)*tan(theta) + R*cos(phi)*tan(theta);
    der(theta) = Q*cos(phi) - R*sin(phi);
    cos(theta)*der(psi) = Q*sin(phi) + R*cos(phi);
    der(P) = M_x;
    der(Q) = M_y;
    der(R) = M_z;
end Quad;
