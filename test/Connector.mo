connector Link2
    Real force;
end Link2;

model RigidBody2
    Link link;
    Real x;
equation
    der(x) = link.force;
end RigidBody2;

model Connector
    Link2 link;
    RigidBody2 body;
equation
    connect(link, body.link);
end Connector;

