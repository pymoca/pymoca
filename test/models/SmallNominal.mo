model SmallNominal
    Real x;
    Real alias(nominal = 0.1);
equation
    alias = x;
end SmallNominal;
