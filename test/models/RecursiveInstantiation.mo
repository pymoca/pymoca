// Example taken from Section 5.6.1.4 in the Modelica Specification v3.5
// Added unused contents to test partial instatiation
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

    model Unused1
        Real x=2, z;
        Integer i=3;
    equation
        z = x*i;
    end Unused1;

    model Unused2
        extends Unused1;
        Boolean b = false;
    end Unused2;
end M;
