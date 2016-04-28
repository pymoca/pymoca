connector Link
    Real force;
end Link;

model RigidBody
    Link link;
    Real x;
equation
    der(x) = link.force;
end RigidBody;

model Aircraft
    Link link;
    RigidBody body;
equation
    connect(link, body.link);
end Aircraft;

