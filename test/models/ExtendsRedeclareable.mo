// TODO: Uncertain about correctness of OMC. What do other compilers say?
model F
  model A
    Real x;
  end A;
  A z;
end F;

model D
  replaceable model C
    replaceable model B
      Real y;
    end B;

    B z;
  end C;
  extends C;
end D;

// Apparently OpenModelica (<=v12.0) handles extends clauses before redeclarations take effect.
// It is that, or it does the lookup in the class tree instead of instance tree (which is not what the spec says).
// Maybe try to easily allow switching between the two options?
model E
  extends D(z.y(nominal=2), redeclare model C=F);
end E;
