antlr_files = generated/$(1).tokens \
	generated/$(1)Lexer.tokens \
	generated/$(1)Lexer.py \
	generated/$(1)Listener.py \
	generated/$(1)Parser.py

all: build_fsm build_dot build_modelica

generated/__init__.py:
	mkdir -p generated
	touch $@

# FSM
fsm_files = $(call antlr_files,FSM)
$(fsm_files): FSM.g4
	./antlr4 -o generated -Dlanguage=Python2 $<
build_fsm: $(fsm_files) generated/__init__.py
test_fsm: build_fsm
	python fsm.py test/test.fsm

# DOT
dot_files = $(call antlr_files,DOT)
$(dot_files): DOT.g4
	./antlr4 -o generated -Dlanguage=Python2 $<
build_dot: $(dot_files) generated/__init__.py
test_dot: build_dot
	python dot.py ./test/t.dot

# SMC
smc_files = $(call antlr_files,SMC)
$(smc_files): SMC.g4
	./antlr4 -o generated -Dlanguage=Python2 $<
build_smc: $(smc_files) generated/__init__.py
test_smc: build_smc
	python smc.py ./test/test.smc

# Modelica
modelica_files = $(call antlr_files,Modelica)
$(modelica_files): Modelica.g4
	./antlr4 -o generated -Dlanguage=Python2 $<
build_modelica: $(modelica_files) generated/__init__.py
test_modelica: build_modelica
	python modelica.py ./test/BouncingBall.mo

# Generic
test: test_dot test_fsm test_smc

clean:
	rm -rf generated

.PHONY: clean test_dot test_fsm test test_smc
