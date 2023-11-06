// Taken from https://trac.modelica.org/Modelica/ticket/1829#comment:38
package P
  constant Real max_m = 1000;
  constant Real min_m = 1.0;
  model BT
    parameter Real m = -1;
      Real x;
    equation
      x = m;
  end BT;

  model M
    model A
      replaceable model AT
        Real x;
        parameter Real m = -10;
      equation
        x = m;
      end AT;
      AT at;
    end A;

    model B
      model BT
        Real x;
        constant Real bla = 100;
        // Are there references in which the scope remains BT, not bt?
        parameter Real m(nominal=BT.bla, min=P.min_m, max=2 * max_m) = 0;
        parameter Real n(nominal=bla, min=P.min_m, max=2 * max_m) = 0;
      equation
        x = m * max_m;
      end BT;
      BT bt(bla=150);
    end B;

    extends A(redeclare model AT = BT); // Here we expect to find P.M.B.BT, not P.BT
    extends B(BT(bla=200));
  end M;
end P;


// - Scope back to None?
// - New testcase of this ExtendsOrder... not sure if we have something that covers this weird scoping I do now with bla
// - find_symbol can be unified with find_constant_symbol. Finding symbols _outside_ the current class is
//   only allowed for constants, not for other types. Raise Exception if "symbol found, but not constant" just like Openmodelica.
// - Testcase for the above? I.e. symbol found but not constant
// - Can you have constants that are not elementary? I.e. ones that have a type that is a class?