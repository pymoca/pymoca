model SimplifyDifferentiatedState
    Real y;
    Real _x;
equation
    _x = 3 * y;
    der(_x) = 1.0;
end SimplifyDifferentiatedState;
