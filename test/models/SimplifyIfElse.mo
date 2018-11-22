model SimplifyIfElse
    parameter Boolean b = true;
    Real y;
    Real _tmp;
equation
    y = 1 + _tmp;
    if b then
        _tmp = 2;
    else
        _tmp = 3;
    end if;
end SimplifyIfElse;
