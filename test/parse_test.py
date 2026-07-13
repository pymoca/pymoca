#!/usr/bin/env python
"""
Parse-only and miscellaneous AST tests.
"""

import io
import os
import re
import threading
import time

from conftest_parse import (
    MODEL_DIR,
    MSL4_DIR,
    _flush,
    parse_model_files,
)

from pymoca import ast
from pymoca import parser
from pymoca import tree

import pytest


def test_ast_element_full_name():
    """Test fully-qualified name lookup to element and back to name"""
    ast_tree = parse_model_files("TreeLookup.mo")
    full_names = (
        "Level1.Level2.Level3.TestPackage.TestClass",  # Class
        "Level1.Level2.Level3.TestPackage.c",  # Symbol
    )
    for name in full_names:
        element = tree.find_name(ast_tree, name)
        assert element is not None
        assert ast.element_full_name(element) == name


def test_connector():
    # ast_tree =
    parse_model_files("Connector.mo")
    # states = ast_tree.classes['Aircraft'].states
    # names = sorted([state.name for state in states])
    # names_set = sorted(list(set(names)))
    # if names != names_set:
    #     raise IOError('{:s} != {:s}'.format(str(names), str(names_set)))
    _flush()


def test_declarations_are_not_builtin_names():
    """Check that declaration names are not builtin types

    See MLS v3.5 section 4.8 Predefined Types and Classes
    """
    # Check that errors are caught in the parser
    error_templates = (
        # Class, model, function and type definitions
        "class {} end {};",
        "model {} end {};",
        "function {} end {};",
        "model A type {} = Real; end A;",
        # Component declarations
        "model A Real {}; end A;",
        "model A Real a, {}; end A;",
        "model A type B = {}; B {}; end A;",
        "model A model B end B; B {}; end A;",
        # Redeclares
        "model A B b(redeclare Real {}); model B Real x; end B; end A;",
        "model A B b(redeclare type {} = Integer); replaceable type B = {}; end A;",
    )
    for template in error_templates:
        for name in ast.Class.BUILTIN:
            txt = template.format(name, name)
            with pytest.raises(
                parser.ModelicaSyntaxError,
                match=f"Predefined type {name} not allowed as identifier",
            ):
                parser.parse(txt)

    ok_examples = (
        "model A parameter Integer i=1; String s=String(i); end A;",
        'model A redeclare type Voltage = Real(unit="V"); end A;',
    )
    for txt in ok_examples:
        ast_tree = parser.parse(txt)
        assert ast_tree is not None, f"for '{txt}'"


def test_parse_nonexistent_file_error():
    """Test parse_file on non-existent file raises an error"""
    with pytest.raises(FileNotFoundError, match="File not found: NonExistentFile.mo"):
        parser.parse_file("NonExistentFile.mo")


def test_deep_copy_timeout():
    ast_tree = parse_model_files("DeepCopyTimeout.mo")

    # Start a background thread which will run the flattening, such that
    # we can kill it if takes to long.
    thread = threading.Thread(
        target=tree.flatten,
        args=(
            ast_tree,
            ast.ComponentRef(name="Test"),
        ),
    )

    # Daemon threads automatically stop when the program stops (and do not
    # prevent the program from exiting)
    thread.setDaemon(True)
    thread.start()

    # Use a timeout of 5 seconds. We check every 100 ms sec, such that the
    # test is fast to succeed when everything works as expected.
    for _ in range(50):
        time.sleep(0.1)
        if not thread.is_alive():
            return
    assert not thread.is_alive(), "Timeout occurred"


def test_visibility_in_ast():
    """Test visibility is set correctly in AST elements"""

    ast_tree = parse_model_files("Visibility.mo")
    A = ast_tree.classes["A"]

    # Test visibility of the classes
    assert ast.Visibility.PUBLIC == A.visibility
    assert ast.Visibility.PROTECTED == A.classes["B"].visibility
    assert ast.Visibility.PUBLIC == A.classes["C"].visibility

    # Test symbols visibility
    assert ast.Visibility.PROTECTED == A.symbols["b"].visibility
    assert ast.Visibility.PUBLIC == A.symbols["c"].visibility
    assert ast.Visibility.PUBLIC == A.symbols["d"].visibility
    assert ast.Visibility.PROTECTED == A.symbols["e"].visibility
    assert ast.Visibility.PROTECTED == A.symbols["f"].visibility

    # Test extends visibility
    C = A.classes["C"]
    assert ast.Visibility.PROTECTED == C.extends[0].visibility

    # Test visibility of public symbols in a protected class is as parsed
    B = A.classes["B"]
    for symbol in B.symbols.values():
        assert ast.Visibility.PUBLIC == symbol.visibility


def test_basemodelica_scalarized():
    """Test parsing BaseModelica example with scalar varables"""
    ast_tree = parse_model_files("BaseModelicaScalarized.mo")
    assert ast_tree is not None
    class_name = "'ManglingTest'"
    comp_ref = ast.ComponentRef.from_string(class_name)
    flat_tree = tree.flatten(ast_tree, comp_ref)
    # Check a few things, by no means exhaustive
    symbol = flat_tree.classes[class_name].symbols["'root.mm[1].p'"]
    assert symbol.value.value == 2.0
    assert "parameter" in symbol.prefixes
    found_y_equation = False
    for eqn in flat_tree.classes[class_name].equations:
        if isinstance(eqn.left, (ast.Symbol, ast.ComponentRef)) and eqn.left.name == "'y'":
            assert eqn.right.name == "'root.m.x'"
            found_y_equation = True
            break
    assert found_y_equation


def test_basemodelica_hierarchical():
    """Test parsing BaseModelica example with array variables"""
    ast_tree = parse_model_files("BaseModelicaHierarchical.mo")
    assert ast_tree is not None
    class_name = "'ManglingTest'"
    comp_ref = ast.ComponentRef.from_string(class_name)
    flat_tree = tree.flatten(ast_tree, comp_ref)
    pass
    # Check a few things, by no means exhaustive
    symbol = flat_tree.classes[class_name].symbols["'root'.'mm'.'p'"]
    dimensions = [dim.value for [dim] in symbol.dimensions]
    assert [None, 2, None] == dimensions
    values = [val.value for val in symbol.value.values]
    assert [2.0, 3.0] == values
    found_y_equation = False
    for eqn in flat_tree.classes[class_name].equations:
        if isinstance(eqn.left, (ast.Symbol, ast.ComponentRef)) and eqn.left.name == "'y'":
            assert eqn.right.name == "'root'.'m'.'x'"
            found_y_equation = True
            break
    assert found_y_equation


def test_signed_expression():
    """Test that both + and - prefix operators work in expressions"""
    txt = """
        model A
          parameter Integer iplus = +1;
          parameter Integer ineg = -iplus;
          parameter Real rplus = +1.0;
          parameter Real rneg = -1.0;
          parameter Real rboth = -1.0 - +1.0;
          parameter Boolean option = true;
          parameter Integer itest = if option then +2 else -2;
        end A;
    """

    ast_tree = parser.parse(txt)

    class_name = "A"
    comp_ref = ast.ComponentRef.from_string(class_name)

    flat_tree = tree.flatten(ast_tree, comp_ref)

    # Test that signed expressions survive flattening.
    # The pipeline evaluates constant unary expressions (+1 → 1, -1.0 → -1.0)
    # while preserving non-constant references and compound expressions.
    symbols = flat_tree.classes["A"].symbols
    # +literal evaluates to the literal itself
    assert isinstance(symbols["iplus"].value, ast.Primary)
    assert symbols["iplus"].value.value == 1
    assert isinstance(symbols["rplus"].value, ast.Primary)
    assert symbols["rplus"].value.value == 1.0
    # -variable reference preserved as expression
    assert symbols["ineg"].value.operator == "-"
    assert len(symbols["ineg"].value.operands) == 1
    # -literal evaluates to negative literal
    assert isinstance(symbols["rneg"].value, ast.Primary)
    assert symbols["rneg"].value.value == -1.0
    # Compound expressions preserve structure
    assert symbols["rboth"].value.operands[1].operator == "+"
    assert len(symbols["rboth"].value.operands[1].operands) == 1
    assert symbols["itest"].value.expressions[0].operator == "+"
    assert symbols["itest"].value.expressions[1].operator == "-"
    assert len(symbols["itest"].value.expressions[0].operands) == 1
    assert len(symbols["itest"].value.expressions[1].operands) == 1


def test_unit_type():
    txt = """
        model A
          parameter Integer x = 1;
          parameter Real y = 1.0;
          parameter Real z = 1;  // Mismatch
          parameter Integer w = 1.0;  // Mismatch
        end A;
    """

    ast_tree = parser.parse(txt)

    class_name = "A"
    comp_ref = ast.ComponentRef.from_string(class_name)

    flat_tree = tree.flatten(ast_tree, comp_ref)

    # For the moment, we do not raise errors/warnings or do
    # auto-conversions in the parser/flattener.
    assert isinstance(flat_tree.classes["A"].symbols["x"].value.value, int)
    assert isinstance(flat_tree.classes["A"].symbols["y"].value.value, float)
    # self.assertIsInstance(flat_tree.classes['A'].symbols['z'].value.value, int)
    # self.assertIsInstance(flat_tree.classes['A'].symbols['w'].value.value, float)


def test_unit_type_array():
    txt = """
        model A
          parameter Integer x[2, 2] = {{1, 2}, {3, 4}};
          parameter Real y[2, 2] = {{1.0, 2.0}, {3.0, 4.0}};
        end A;
    """

    ast_tree = parser.parse(txt)

    class_name = "A"
    comp_ref = ast.ComponentRef.from_string(class_name)

    flat_tree = tree.flatten(ast_tree, comp_ref)

    # For the moment, we leave type conversions to the backends. We only want to
    # be sure that we read in the correct type in the parser.
    for i in range(2):
        for j in range(2):
            assert isinstance(
                flat_tree.classes["A"].symbols["x"].value.values[i].values[j].value, int
            )
            assert isinstance(
                flat_tree.classes["A"].symbols["y"].value.values[i].values[j].value, float
            )


def test_class_comment():
    """Test that class comment/description is retained after flattening"""
    library_tree = parse_model_files("Aircraft.mo")
    comp_ref = ast.ComponentRef.from_string("Aircraft")
    flat_tree = tree.flatten(library_tree, comp_ref)
    aircraft = flat_tree.classes["Aircraft"]
    assert aircraft.comment == "the aircraft"
    assert aircraft.symbols["accel.a_x"].comment == "true acceleration"
    assert aircraft.symbols["accel.ma_x"].comment == "measured acceleration"
    assert aircraft.symbols["body.g"].comment == ""


def test_syntax_error():
    """Test ModelicaSyntaxError parse exception and related functions"""

    # Lexical syntax error at first character
    with pytest.raises(
        parser.ModelicaSyntaxError,
        match=r"token recognition error at: '`' \(input, line 1\)",
    ):
        _ = parser.parse("`", bypass_cache=True)

    # Syntax error at <EOF> without trailing newline
    with pytest.raises(
        parser.ModelicaSyntaxError,
        match=r"missing ';' at '<EOF>' \(input, line 2\)",
    ):
        _ = parser.parse("model A\nend A", bypass_cache=True)

    # Syntax error at <EOF> with trailing newline
    with pytest.raises(
        parser.ModelicaSyntaxError,
        match="missing ';' at '<EOF>' \\(input, line 3\\)",
    ):
        parser.parse("model A\nend A\n", bypass_cache=True)

    # Syntax error in a file
    with pytest.raises(
        parser.ModelicaSyntaxError,
        match=r"missing '=' at '\.' \(RedeclareNestedClass.mo.fail_parse, line 21\)",
    ) as exc_info:
        _ = parse_model_files("RedeclareNestedClass.mo.fail_parse")

    # Test print_syntax_error() of previous exception
    error_text = io.StringIO()
    parser.print_syntax_error(exc_info.value, error_text)
    expected_regex = (
        r".*RedeclareNestedClass\.mo\.fail_parse:21:48:",
        r"  extends D\(c.z.x\(nominal=2\), redeclare model C\.B=F\);",
        r"ModelicaSyntaxError: missing '=' at '\.'",
    )
    expected_regex = "\n".join(expected_regex)
    assert re.search(expected_regex, error_text.getvalue())


def test_inner_outer_final_parsed_on_symbol():
    """inner, outer, and final are set on parsed symbols (MLS 5.4, 7.3)."""
    ast_tree = parser.parse("model M inner Real x; outer Real y; final Integer n = 1; end M;")
    m = ast_tree.classes["M"]
    assert m.symbols["x"].inner is True
    assert m.symbols["x"].outer is False
    assert m.symbols["y"].outer is True
    assert m.symbols["y"].inner is False
    assert m.symbols["n"].final is True
    assert m.symbols["n"].inner is False


def test_modelicapath_lookup():
    """Test modelicapath tree top level classes transformed correctly"""
    test_package = os.path.join(MODEL_DIR, "Package")
    stub_tree = parser.modelicapath_to_tree(dirs=[test_package])
    assert len(stub_tree.classes) > 0
    keys = set(stub_tree.classes.keys())
    assert "Package" in keys
    # Test lookup on some MSL that parses OK
    msl_dir = os.path.join(MSL4_DIR, "Modelica")
    msl = parser.modelicapath_to_tree(dirs=[msl_dir])
    # Units is a .mo file, SI is defined inside Units
    si = tree.find_name(msl, "Modelica.Units.SI")
    assert isinstance(si, ast.Class)
    assert "Angle" in si.classes
    # TODO: More tests to get high enough coverage


def test_lazyparse_syntax_error(tmp_path):
    """LazyParseClass propagates ModelicaSyntaxError on first content access."""
    pkg_dir = tmp_path / "Pkg"
    pkg_dir.mkdir()
    (pkg_dir / "package.mo").write_text("package Pkg\n  INVALID SYNTAX\nend Pkg;")
    stub_tree = parser.modelicapath_to_tree(dirs=[pkg_dir])
    pkg = stub_tree.classes["Pkg"]
    with pytest.raises(parser.ModelicaSyntaxError):
        _ = pkg.classes


def test_lazyparse_missing_expected_class(tmp_path):
    """LazyParseClass raises KeyError when parsed file contains wrong class name."""
    pkg_dir = tmp_path / "Pkg"
    pkg_dir.mkdir()
    (pkg_dir / "package.mo").write_text("package Wrong\nend Wrong;")
    stub_tree = parser.modelicapath_to_tree(dirs=[pkg_dir])
    pkg = stub_tree.classes["Pkg"]
    with pytest.raises(KeyError, match="Pkg"):
        _ = pkg.classes


def test_lazyparse_file_disappeared(tmp_path):
    """LazyParseClass propagates FileNotFoundError when stub path was deleted."""
    pkg_dir = tmp_path / "Pkg"
    pkg_dir.mkdir()
    pkg_mo = pkg_dir / "package.mo"
    pkg_mo.write_text("package Pkg\nend Pkg;")
    stub_tree = parser.modelicapath_to_tree(dirs=[pkg_dir])
    pkg = stub_tree.classes["Pkg"]
    pkg_mo.unlink()
    with pytest.raises(FileNotFoundError):
        _ = pkg.classes


def test_lazyparse_subdir_clashes_with_parsed_class(tmp_path):
    """LazyParseClass raises ModelicaPathError when package.mo declares a class
    whose name also exists as a subdirectory (MLS 3.5 §13.4.1)."""
    pkg_dir = tmp_path / "Pkg"
    pkg_dir.mkdir()
    (pkg_dir / "package.mo").write_text("package Pkg\n  package Bar\n  end Bar;\nend Pkg;")
    bar_dir = pkg_dir / "Bar"
    bar_dir.mkdir()
    (bar_dir / "package.mo").write_text("package Bar\nend Bar;")
    stub_tree = parser.modelicapath_to_tree(dirs=[pkg_dir])
    pkg = stub_tree.classes["Pkg"]
    with pytest.raises(parser.ModelicaPathError, match="Bar"):
        _ = pkg.classes


def test_lazyparse_metadata_matches_direct_parse(tmp_path):
    """Excluded metadata attrs match direct-parse values after lazy parse fires.

    Validates _ATTRS_NEEDING_PARSE exclusions: stub defaults are overwritten by
    _parse_in_place so every attr has the correct value once content access fires parse.
    """
    # Directory-package stubs: encapsulated, partial, final
    dir_cases = [
        ("encapsulated", "encapsulated package Pkg\nend Pkg;", True),
        ("partial", "partial package Pkg\nend Pkg;", True),
        ("final", "final package Pkg\nend Pkg;", True),
    ]
    for attr, content, expected in dir_cases:
        d = tmp_path / f"stub_{attr}"
        d.mkdir()
        pkg_dir = d / "Pkg"
        pkg_dir.mkdir()
        (pkg_dir / "package.mo").write_text(content)
        lazy = parser.modelicapath_to_tree(dirs=[pkg_dir])
        stub = lazy.classes["Pkg"]
        assert isinstance(stub, parser.LazyParseClass), f"{attr}: not a stub before access"
        _ = stub.extends  # trigger parse
        direct = parser.parse_text(content).classes["Pkg"]
        assert getattr(stub, attr) == getattr(direct, attr) == expected, f"mismatch: {attr}"

    # File-based stubs (inside a wrapper Lib/ package so _dir_to_tree adds them)
    lib_dir = tmp_path / "Lib"
    lib_dir.mkdir()
    (lib_dir / "package.mo").write_text("package Lib\nend Lib;")

    # type: force _file_class_type heuristic miss (keyword hidden past 1KB peek)
    long_comment = "// " + "x" * 1025
    type_content = f"{long_comment}\npackage Pkg\nend Pkg;"
    (lib_dir / "Pkg.mo").write_text(type_content)
    lazy = parser.modelicapath_to_tree(dirs=[lib_dir])
    # Get Pkg stub via vars() bypass so accessing it doesn't trigger Lib parse
    pkg_stub = vars(lazy.classes["Lib"])["classes"]["Pkg"]
    assert pkg_stub.type == "class", "heuristic should return 'class' for keyword past 1KB"
    _ = pkg_stub.extends  # trigger Pkg parse
    assert pkg_stub.type == "package"

    # is_short_class_definition: file stub for a short class definition
    (lib_dir / "Angle.mo").write_text("type Angle = Real;")
    lazy2 = parser.modelicapath_to_tree(dirs=[lib_dir])
    angle_stub = vars(lazy2.classes["Lib"])["classes"]["Angle"]
    assert angle_stub.is_short_class_definition is False  # default before parse
    _ = angle_stub.extends  # trigger parse
    assert angle_stub.is_short_class_definition is True


def test_modelicapath_matches_direct_parse():
    """MODELICAPATH-parsed model matches direct-parsed result end-to-end.

    Exercises excluded metadata attrs and content attrs on a representative
    single-class model to detect any stub-default vs. parsed-value divergence.
    """
    pkg_dir = os.path.join(MODEL_DIR, "Package")
    direct_spring = parser.parse_file(os.path.join(pkg_dir, "Spring.mo")).classes["Spring"]

    lazy_tree = parser.modelicapath_to_tree(dirs=[pkg_dir])
    _ = lazy_tree.classes["Package"].classes  # trigger Package parse
    spring_stub = lazy_tree.classes["Package"].classes["Spring"]
    _ = spring_stub.symbols  # trigger Spring parse

    for attr in ("type", "partial", "encapsulated", "final", "is_short_class_definition"):
        assert getattr(spring_stub, attr) == getattr(direct_spring, attr), f"mismatch: {attr}"
    assert set(spring_stub.symbols.keys()) == set(direct_spring.symbols.keys())
    assert len(spring_stub.equations) == len(direct_spring.equations)


def test_lazyparse_remove_class(tmp_path):
    """LazyParseClass.remove_class removes child and clears parent without triggering parse."""
    pkg_dir = tmp_path / "Pkg"
    pkg_dir.mkdir()
    (pkg_dir / "package.mo").write_text("package Pkg\nend Pkg;")
    (pkg_dir / "Child.mo").write_text("model Child\nend Child;")
    stub_tree = parser.modelicapath_to_tree(dirs=[pkg_dir])
    pkg = stub_tree.classes["Pkg"]
    child = vars(pkg)["classes"]["Child"]
    pkg.remove_class(child)
    assert "Child" not in vars(pkg)["classes"]
    assert child.parent is None
    assert isinstance(pkg, parser.LazyParseClass)  # parse never fired


def test_modelicapath_first_root_wins(tmp_path):
    """modelicapath_to_tree loads a colliding top-level name entirely from the first
    root that offers it; a later root's version of that name is never consulted
    (MLS 3.5 §13.3)."""
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    for d in (d1, d2):
        d.mkdir()
        modelica = d / "Modelica"
        modelica.mkdir()
        (modelica / "package.mo").write_text("package Modelica\nend Modelica;")
    (d1 / "Modelica" / "Units").mkdir()
    (d1 / "Modelica" / "Units" / "package.mo").write_text("package Units\nend Units;")
    (d2 / "Modelica" / "Math").mkdir()
    (d2 / "Modelica" / "Math" / "package.mo").write_text("package Math\nend Math;")

    merged = parser.modelicapath_to_tree(dirs=[d1 / "Modelica", d2 / "Modelica"])
    modelica = merged.classes["Modelica"]
    assert isinstance(modelica, parser.LazyParseClass)  # parse never fired
    assert tree.find_name(merged, "Modelica.Units") is not None
    assert tree.find_name(merged, "Modelica.Math") is None
    # Winning stub came entirely from root1
    assert vars(modelica)["path"].parent.parent == d1


def test_modelicapath_ordered_disjoint(tmp_path):
    """Non-colliding top-level names from multiple roots all resolve."""
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    (d1 / "Foo").mkdir()
    (d1 / "Foo" / "package.mo").write_text("package Foo\nend Foo;")
    (d2 / "Bar").mkdir()
    (d2 / "Bar" / "package.mo").write_text("package Bar\nend Bar;")

    merged = parser.modelicapath_to_tree(dirs=[d1, d2])
    assert "Foo" in merged.classes
    assert "Bar" in merged.classes


def test_modelicapath_nonstructured_shadows_structured(tmp_path):
    """A bare top-level .mo file in the winning root shadows a same-named structured
    package (package.mo directory) in a later root."""
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    (d1 / "A.mo").write_text("model A\nend A;")
    (d2 / "A").mkdir()
    (d2 / "A" / "package.mo").write_text("package A\nend A;")

    merged = parser.modelicapath_to_tree(dirs=[d1, d2])
    a = merged.classes["A"]
    assert isinstance(a, parser.LazyParseClass)
    assert vars(a)["path"] == d1 / "A.mo"


def test_lazyparse_extend(tmp_path):
    """LazyParseClass._extend / _update_parent_refs work when the stub is the
    receiver of Tree.extend, without firing a parse."""
    pkg_dir = tmp_path / "Pkg"
    pkg_dir.mkdir()
    (pkg_dir / "package.mo").write_text("package Pkg\nend Pkg;")
    stub_tree = parser.modelicapath_to_tree(dirs=[pkg_dir])
    pkg_stub = stub_tree.classes["Pkg"]
    assert isinstance(pkg_stub, parser.LazyParseClass)

    other = ast.Class(name="Other")
    child = ast.Class(name="Child")
    other.add_class(child)

    pkg_stub._extend(other)
    pkg_stub._update_parent_refs()

    assert isinstance(pkg_stub, parser.LazyParseClass)  # parse never fired
    assert "Child" in vars(pkg_stub)["classes"]
    assert vars(pkg_stub)["classes"]["Child"].parent is pkg_stub


def test_explicit_shadows_library_stub(tmp_path):
    """An explicit top-level class extended into a MODELICAPATH tree replaces a
    same-named library stub wholesale; a second explicit file with the same
    top-level name merges normally with the first (not the stub)."""
    pkg_dir = tmp_path / "A"
    pkg_dir.mkdir()
    (pkg_dir / "package.mo").write_text("package A\n  model Original\n  end Original;\nend A;")
    modelicapath_tree = parser.modelicapath_to_tree(dirs=[tmp_path])
    assert isinstance(modelicapath_tree.classes["A"], parser.LazyParseClass)

    explicit1 = parser.parse("within A;\nmodel Patched\nend Patched;\n")
    modelicapath_tree.extend(explicit1)
    a = modelicapath_tree.classes["A"]
    assert not isinstance(a, parser.LazyParseClass)
    assert "Patched" in a.classes
    assert "Original" not in a.classes

    explicit2 = parser.parse("within A;\nmodel Patched2\nend Patched2;\n")
    modelicapath_tree.extend(explicit2)
    a = modelicapath_tree.classes["A"]
    assert "Patched" in a.classes
    assert "Patched2" in a.classes


def test_lazyparse_reparents_symbols(tmp_path):
    """_parse_in_place re-parents symbols to the demoted class object."""
    pkg_dir = tmp_path / "Lib"
    pkg_dir.mkdir()
    (pkg_dir / "package.mo").write_text("package Lib\n  constant Real pi = 3.14159;\nend Lib;")
    stub_tree = parser.modelicapath_to_tree(dirs=[pkg_dir])
    lib = stub_tree.classes["Lib"]
    syms = lib.symbols  # triggers parse
    assert "pi" in syms
    assert syms["pi"].parent is lib


def test_modelicapath_non_directory(tmp_path):
    """modelicapath_to_tree raises ModelicaPathError when a path is not a directory."""
    a_file = tmp_path / "notadir.mo"
    a_file.write_text("model Nope\nend Nope;")
    with pytest.raises(parser.ModelicaPathError, match="non-directory"):
        parser.modelicapath_to_tree(dirs=[a_file])


def test_modelicapath_file_type_unreadable(tmp_path):
    """_file_class_type returns 'class' when the file cannot be opened."""
    # Pass a directory as the path: open() on a dir raises IsADirectoryError (OSError)
    result = parser._file_class_type(tmp_path)
    assert result == "class"


def test_modelicapath_path_to_class_ignores_other(tmp_path):
    """_path_to_class returns None for paths that are neither files nor directories."""
    dangling = tmp_path / "dangling.mo"
    dangling.symlink_to(tmp_path / "nonexistent.mo")
    assert parser._path_to_class(dangling) is None


def test_parse_enum_basic():
    """Parsing a simple enumeration type builds the correct AST.

    ``type E = enumeration(one, two, three)`` should produce a ``type`` Class
    with ``enumeration=True`` and three :class:`ast.EnumerationLiteral` symbols
    bearing 1-based ordinals in declaration order.
    """
    tree_ = parser.parse("type E = enumeration(one, two, three);")
    enum_class = tree_.classes["E"]
    assert enum_class.type == "type"
    assert enum_class.enumeration is True
    assert list(enum_class.symbols.keys()) == ["one", "two", "three"]
    for ordinal, (name, sym) in enumerate(enum_class.symbols.items(), start=1):
        assert isinstance(sym, ast.EnumerationLiteral), f"{name!r} should be EnumerationLiteral"
        assert sym.ordinal == ordinal
        assert sym.prefixes == ["constant"]
        assert sym.type.name == "E"


def test_parse_enum_literal_comments():
    """Enumeration literal comments are captured."""
    tree_ = parser.parse('type R = enumeration(y "y (year)", d "d (day)");')
    assert tree_.classes["R"].symbols["y"].comment == "y (year)"
    assert tree_.classes["R"].symbols["d"].comment == "d (day)"


def test_parse_enum_unspecified():
    """``enumeration(:)`` (placeholder for redeclaration) parses with no literals."""
    tree_ = parser.parse("type E = enumeration(:);")
    enum_class = tree_.classes["E"]
    assert enum_class.enumeration is True
    assert len(enum_class.symbols) == 0


def test_parse_enum_resolution_mo():
    """The MSL-4.0.x Resolution.mo file parses without error."""
    import pathlib

    res_mo = (
        pathlib.Path(__file__).parent / "libraries/MSL-4.0.x/Modelica/Clocked/Types/Resolution.mo"
    )
    tree_ = parser.parse(res_mo.read_text())
    # within Modelica.Clocked.Types nests the class under Modelica.*
    enum_class = tree_.classes["Modelica"].classes["Clocked"].classes["Types"].classes["Resolution"]
    assert enum_class.enumeration is True
    assert list(enum_class.symbols.keys()) == ["y", "d", "h", "m", "s", "ms", "us", "ns"]


def test_parse_class_extends_basic():
    """The `class extends` form (long_class_specifier extends) parses correctly.

    ``redeclare model extends BaseProperties "cmt" ... end BaseProperties;``
    should produce a Class named "BaseProperties" with:
      - the comment from the string_comment,
      - one ExtendsClause whose component name is "BaseProperties",
      - body members (symbols, equations) wired up normally.
    """
    tree_ = parser.parse(
        "package P\n"
        "  extends Base;\n"
        '  redeclare model extends BaseProperties "cmt"\n'
        "    Real x;\n"
        "  equation\n"
        "    x = 1;\n"
        "  end BaseProperties;\n"
        "end P;\n"
    )
    bp = tree_.classes["P"].classes["BaseProperties"]
    assert bp.name == "BaseProperties"
    assert bp.comment == "cmt"
    assert len(bp.extends) == 1
    assert bp.extends[0].component.name == "BaseProperties"
    assert "x" in bp.symbols
    assert len(bp.equations) == 1


def test_parse_class_extends_with_modification():
    """The `class extends` form with class_modification parses correctly.

    ``model extends BaseProperties(T0 = 300) ... end BaseProperties;`` is the
    pattern used in Modelica/Media/Water/package.mo; the class_modification must
    be attached to the ExtendsClause.
    """
    tree_ = parser.parse(
        "package P\n"
        "  extends Base;\n"
        '  redeclare model extends BaseProperties(p_default = 1e5) ""\n'
        "    Real T;\n"
        "  end BaseProperties;\n"
        "end P;\n"
    )
    bp = tree_.classes["P"].classes["BaseProperties"]
    assert bp.name == "BaseProperties"
    assert len(bp.extends) == 1
    ext = bp.extends[0]
    assert ext.component.name == "BaseProperties"
    # class_modification must carry the p_default argument
    assert len(ext.class_modification.arguments) == 1


def test_modelicapath_root_without_package_mo(tmp_path):
    """A MODELICAPATH entry without package.mo has its children loaded directly.

    MSL-4.0.x/ has no package.mo but contains Modelica/, ModelicaServices/, Complex.mo.
    Passing the parent directory should expose all three as top-level classes.
    """
    # Build a minimal MODELICAPATH root: no package.mo, two sub-packages, one bare .mo
    root = tmp_path / "msl_root"
    root.mkdir()

    pkg_a = root / "PkgA"
    pkg_a.mkdir()
    (pkg_a / "package.mo").write_text("package PkgA\nend PkgA;")

    pkg_b = root / "PkgB"
    pkg_b.mkdir()
    (pkg_b / "package.mo").write_text("package PkgB\nend PkgB;")

    (root / "Standalone.mo").write_text("model Standalone\nend Standalone;")

    stub_tree = parser.modelicapath_to_tree(dirs=[str(root)])
    assert "PkgA" in stub_tree.classes
    assert "PkgB" in stub_tree.classes
    assert "Standalone" in stub_tree.classes


def test_parse_omitted_output_assignment():
    """(a, , c) := f(x) — omitted middle output parses; slot becomes None (MLS §B.2.6.5)."""
    txt = """
function F
  input Real x;
  output Real a;
  output Real b;
  output Real c;
algorithm
  (a, , c) := g(x);
end F;
"""
    tree_ = parser.parse(txt)
    f = tree_.classes["F"]
    stmt = f.statements[0]
    assert isinstance(stmt, ast.AssignmentStatement)
    assert len(stmt.left) == 3
    assert stmt.left[0].name == "a"
    assert stmt.left[1] is None
    assert stmt.left[2].name == "c"


def test_parse_full_output_assignment():
    """(a, b, c) := f(x) — all outputs present; no slot is None (regression guard)."""
    txt = """
function F
  input Real x;
  output Real a;
  output Real b;
  output Real c;
algorithm
  (a, b, c) := g(x);
end F;
"""
    tree_ = parser.parse(txt)
    f = tree_.classes["F"]
    stmt = f.statements[0]
    assert isinstance(stmt, ast.AssignmentStatement)
    assert len(stmt.left) == 3
    assert all(ref is not None for ref in stmt.left)
    assert [ref.name for ref in stmt.left] == ["a", "b", "c"]


def _sym_value_expr(sym):
    """Extract the initialiser expression from a symbol's class_modification.

    Symbols store their ``= <expr>`` value in
    ``class_modification.arguments[0].value.modifications[0]`` at parse time;
    the ``sym.value`` field is only populated during instantiation.
    """
    return sym.class_modification.arguments[0].value.modifications[0]


def test_function_partial_application():
    """Parse a `function f(...)` partial-application argument (MLS §B.2.7.11)."""
    # `function f()` with no bound arguments
    mf = parser.parse(
        "model A\n  parameter Real s = g(function f(), 0, 1);\nend A;\n",
        bypass_cache=True,
    )
    value = _sym_value_expr(mf.classes["A"].symbols["s"])
    # outer call g(...): operands are [partial-application, 0, 1]
    assert isinstance(value, ast.Expression)
    partial = value.operands[0]
    assert isinstance(partial, ast.Expression)
    assert isinstance(partial.operator, ast.ComponentRef)
    assert partial.operator.to_tuple() == ("f",)
    assert partial.operands == []

    # `function f(a=p, b=q)` — names are dropped, bound value expressions kept
    mf2 = parser.parse(
        "model B\n"
        "  parameter Real p = 1;\n"
        "  parameter Real q = 2;\n"
        "  parameter Real s = g(function f(a=p, b=q), 0, 1);\n"
        "end B;\n",
        bypass_cache=True,
    )
    value2 = _sym_value_expr(mf2.classes["B"].symbols["s"])
    partial2 = value2.operands[0]
    assert isinstance(partial2, ast.Expression)
    assert isinstance(partial2.operator, ast.ComponentRef)
    assert partial2.operator.to_tuple() == ("f",)
    assert len(partial2.operands) == 2  # a=p, b=q → [p, q]
    assert isinstance(partial2.operands[0], ast.ComponentRef)
    assert partial2.operands[0].name == "p"
    assert isinstance(partial2.operands[1], ast.ComponentRef)
    assert partial2.operands[1].name == "q"


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__])
