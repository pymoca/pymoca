// Example to test inherited name lookup with a case similar to MSL v4
// Modelica.Electrical.Analog.Basic.Resistor where the symbol type name
// PositivePin is referenced in an inheritance hierarchy two levels deep.
// It also tests modifying something other than value.
package P
  package T
    class Voltage = Real(nominal=1);
  end T;
  package I
    class Level1
      // i.e. OnePort
      extends Level2(p.v(nominal=10), n.v(nominal=10));
    end Level1;
    class Level2
      // i.e. TwoPin
      Level3 p(v(nominal=20)), n(v(nominal=20));
    end Level2;
    class Level3
      // i.e. PostitivePin
      T.Voltage v(nominal=30, max=10.0);
    end Level3;
  end I;
  model M
    // i.e. Resistor
    extends I.Level1(p.v(nominal=0), n.v(nominal=0));
    class Level3 // Test for wrong lookup scope with max=30.0
      T.Voltage v(nominal=30, max=30.0);
    end Level3;
  end M;
  class Level3 // Test for wrong lookup scope with max=20.0
    T.Voltage v(nominal=30, max=20.0);
  end Level3;
end P;
