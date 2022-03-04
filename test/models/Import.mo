// Assumes TreeLookup.mo is parsed first so imports can work
model A
  model B
    import Level1.Level2.Level3.{PackageComponents,Test};
    PackageComponents pcb;
    Test tb;
  end B;
  import Level1.Level2.Level3.*;
  Test ta;
  PackageComponents pca;
  TestClassExtended tce_mod(tcet.b=4);
  model TestClassExtended
    extends Level1.Level2.Level3.TestPackage.TestClass;
    import Level1.Level2.Level3.TestPackage.TestClass;
    Test tcet;
  equation
    tcet.b = 2;
  end TestClassExtended;
  B b;
equation
  b.pcb.tc.i = 3;
  b.pcb.tc.a = 5;
end A;