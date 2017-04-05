model FunctionCallTest
    output Real c, a;
algorithm
    a := OneReturnValue(2.3);
    (c, a) := TwoReturnValues(2.3);
end FunctionCallTest;
