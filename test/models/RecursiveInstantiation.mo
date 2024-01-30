// Example taken from Section 5.6.1.4 in the Modelica Specification v3.5
model M
    model B
        A a;
        replaceable model A = C;
        type E = Boolean;
    end B;
    B b(redeclare model A = D (p=1));
    partial model C
        E e;
    end C;

    model D
        extends C;
        parameter E p;
        type E = Integer;
    end D;

    type E = Real;
end M;