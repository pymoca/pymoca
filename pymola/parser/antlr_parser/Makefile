all: generated

fsm_generated: Fsm.g4
	../../../bin/antlr4 -o generated -Dlanguage=Python2 $<
	touch generated/__init__.py

test_fsm: fsm_generated
	python fsm.py test.fsm

modelica_generated: Modelica.g4
	../../../bin/antlr4 -o generated -Dlanguage=Python2 $<
	touch generated/__init__.py

test_modelica: modelica_generated
	python modelica.py BouncingBall.mo

clean:
	rm -rf ./generated

.PHONY: clean
