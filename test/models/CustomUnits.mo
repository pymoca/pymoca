type DummyUnit = Real(final unit = "m/s");

model CustomUnits "CustomUnits Test Model"
	parameter DummyUnit dummy_parameter = 0.0;
end CustomUnits;

model A
	extends CustomUnits(dummy_parameter=10.0);
end A;
