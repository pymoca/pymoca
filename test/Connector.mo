connector Link
    Real force;
end Link;

model RigidBody
    Link link;
end RigidBody;

model Aircraft
    Link link;
    RigidBody body;
equation
    connect(link.force, body.link.force);
end Aircraft;

