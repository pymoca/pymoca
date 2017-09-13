model A
  Real x;
end A;

model D
  replaceable model C
    replaceable model B
      Real y;
    end B;

    B z;
  end C;

  C c;
end D;

// Redeclaring nested (i.e. "contains dot") component references is not
// allowed (as it would likely horribly complicate things). Parser should
// fail.
model E
  extends D(c.z.x(nominal=2), redeclare model C.B=F);
end E;
