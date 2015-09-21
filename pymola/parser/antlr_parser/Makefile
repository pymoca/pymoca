all: generated

fsm_generated: Fsm.g4
	../../../bin/antlr4 -o generated -Dlanguage=Python2 $<
	touch generated/__init__.py

test_fsm: fsm_generated
	python fsm.py test.fsm

modelica_generated: Modelica.g4
	../../../bin/antlr4 -o generated -Dlanguage=Python2 $<
	touch generated/__init__.py

test_modelica: fsm_generated
	python fsm.py test.fsm

clean:
	rm -rf ./generated

.PHONY: clean
