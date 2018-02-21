within Level1.Level2.Level3;

package TestPackage
	class TestClass
		Integer i;
		Integer a;
	end TestClass;
end TestPackage;

model PackageComponents
	Level2.Level3.TestPackage.TestClass tc;
equation
	tc.i = 1;
end PackageComponents;

model Test
	Level1.Level2.Level3.PackageComponents elem;
	Integer b = 3;
equation
	elem.tc.a = b;
end Test;
