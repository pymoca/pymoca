model A
    Real x, y;
equation
    x = 2 * y;
  annotation(
    Diagram(coordinateSystem(extent = {{-100, -100}, {360, 100}}, initialScale = 0.1), graphics = {Bitmap(origin = {95, 225}, extent = {{-359, 203}, {295, -123}}, fileName = "modelica://path/to/picture.jpg")}),
    Icon(coordinateSystem(extent = {{-100, -100}, {360, 100}})),
    version = "",
    uses,
    __OpenModelica_commandLineOptions = "");
end A;
