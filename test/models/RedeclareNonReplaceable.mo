model F
  model A
    Real x;
  end A;
  A z;
end F;

model D
  model C
    replaceable model B
      Real x;
    end B;
    B z;
  end C;
  extends C;
end D;

model E
  extends D(redeclare model C=F);
end E;
