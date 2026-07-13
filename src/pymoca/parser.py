#!/usr/bin/env python
"""
Modelica parse Tree to AST tree.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import pickle
import platform
import re
import sqlite3
import sys
import time
from collections import OrderedDict, deque
from datetime import timedelta
from pathlib import Path
from typing import Any, cast as _cast

import antlr4
import antlr4.Parser
from antlr4 import ParserRuleContext
from antlr4.error.ErrorListener import ErrorListener

import pymoca

from . import ast

# noinspection PyUnresolvedReferences,PyUnresolvedReferences
from .generated.ModelicaLexer import ModelicaLexer  # noqa: I202
from .generated.ModelicaListener import ModelicaListener
from .generated.ModelicaParser import ModelicaParser

# TODO
#  - Named function arguments (note that either all have to be named, or none)
#  - Make sure slice indices (eventually) evaluate to integers


logger = logging.getLogger("pymoca")


DEFAULT_MODEL_CACHE_DB = "model_txt_cache.db"


class ModelicaPathError(Exception):
    """Error in MODELICAPATH or a Modelica library's on-disk structure"""

    def __init__(self, msg):
        self.msg = msg
        super().__init__(self)

    def __str__(self) -> str:
        return str(self.msg)

    def __repr__(self) -> str:
        return type(self).__name__ + "(" + str(self) + ")"


class ModelicaSyntaxError(SyntaxError):
    """Syntax error in Modelica source"""


def syntax_error_from_ctx(
    message: str, ctx: ParserRuleContext, file_name: str = ""
) -> ModelicaSyntaxError:
    """Create a ModelicaSyntaxError from an ANTLR context object"""
    assert ctx.start is not None and ctx.stop is not None
    line1 = ctx.start.line
    col1 = ctx.start.column
    line2 = ctx.stop.line
    col2 = ctx.stop.column
    assert ctx.start.source is not None
    text = ctx.start.source[1].strdata.splitlines()  # type: ignore[union-attr]
    error_text = text[line1 - 1]
    for line in range(line1, line2):
        error_text += text[line]
    # Last two values were added in Python 3.10; earlier versions ignore them
    info: tuple = file_name, line1, col1, error_text, line2, col2
    return ModelicaSyntaxError(message, info)


def print_syntax_error(exception: ModelicaSyntaxError, file=sys.stderr):
    """Print a ModelicaSyntaxError in a readable format

    Python itself will print a traceback with the Modelica source info,
    but this is useful if you want to handle the exception.
    """
    file_name, line1, col1, error_text = exception.args[1][:4]
    # We set the filename attribute if we have the file name,
    # but it is not reflected in the original args, so we get it from the attribute
    file_name = exception.filename
    file_info = ""
    if file_name:
        file_info = f"{file_name}:"
    file_info += f"{line1}:{col1}:"
    print(f"{file_info}\n{error_text}", file=file)
    print(f"ModelicaSyntaxError: {exception.args[0]}", file=file)


class ModelicaFile:
    def __init__(self, **kwargs):
        self.within: list[ast.ComponentRef] = []
        self.classes: OrderedDict[str, ast.Class] = OrderedDict()
        super().__init__(**kwargs)


_TYPE_KEYWORD_RE = re.compile(r"\b(package|model|connector|record|function|type|block|operator)\b")
_WITHIN_RE = re.compile(r"\bwithin\s+[^;]*;")


def _file_class_type(path: Path) -> str:
    """Peek at the first 1024 bytes of a .mo file to detect its Modelica class type keyword."""
    try:
        with path.open(encoding="utf-8", errors="ignore") as f:
            text = f.read(1024)
    except OSError:
        return "class"
    text = _WITHIN_RE.sub("", text, count=1)
    m = _TYPE_KEYWORD_RE.search(text)
    return m.group(1) if m else "class"


class LazyParseClass(ast.Class):
    """Class subtype for an unparsed MODELICAPATH file or package.

    Parses itself in-place on first access of certain unparsed attributes,
    then demotes __class__ back to Class so subsequent accesses have no overhead.
    """

    # Attributes that are empty in LazyParseClass and must be populated by parse.
    # Metadata attributes (type, partial, replaceable, encapsulated, final, comment,
    # is_short_class_definition) are excluded: their stub defaults match the parser's
    # defaults, _parse_in_place overwrites them, and tree code reads them only after a
    # content access has already fired parse. Excluding them avoids triggering parse
    # on every sub-package during stub-tree assembly.
    _ATTRS_NEEDING_PARSE = frozenset(
        {
            "classes",
            "symbols",
            "extends",
            "equations",
            "initial_equations",
            "statements",
            "initial_statements",
            "imports",
            "annotation",
            "functions",
            "enumeration",  # True for enumeration types; governs instantiation/flattening
        }
    )

    def __init__(self, path: Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.path: Path | None = path

    def __getattribute__(self, name: str) -> Any:
        if name in LazyParseClass._ATTRS_NEEDING_PARSE:
            self._parse_in_place()
        return object.__getattribute__(self, name)

    # Use vars(self) to bypass __getattribute__ and avoid triggering parse during tree assembly.
    def _parse_in_place(self) -> None:
        my = vars(self)
        file = parse_text(my["path"].read_text(encoding="utf-8"))
        parsed_class = file.classes[my["name"]]
        # Capture any directory-package sub-stubs added by the filesystem walk before
        # the update overwrites .classes with the parsed content.
        existing_subdir_stubs = list(my["classes"].values())
        saved_parent = my["parent"]

        # Set parent BEFORE __dict__.update so that arg.scope references inside parsed
        # symbols (e.g. stateSelect=stateSelect) resolve via the correct full_name.
        parsed_class.parent = saved_parent
        # Mutate in-place so every existing reference (e.g. captured ast_ref) sees
        # parsed content without needing to know a swap occurred.
        self.__dict__.update(parsed_class.__dict__)
        self.parent = saved_parent
        self.__class__ = ast.Class  # type: ignore[assignment]  # demote: drops .path, removes __getattribute__ overhead

        # Re-parent classes/symbols that arrived via parsed_class (parsed_class is orphaned)
        for sub in dict.values(self.classes):
            if isinstance(sub, ast.Class):
                sub.parent = self
        for sym in self.symbols.values():
            if hasattr(sym, "parent"):
                sym.parent = self

        clash = {c.name for c in existing_subdir_stubs} & set(self.classes.keys())
        if clash:
            raise ModelicaPathError(
                f"Class names {sorted(clash)} declared in both {my['path']} "
                f"and as sibling file/directory (MLS 3.5 §13.4.1)"
            )
        for sub in existing_subdir_stubs:
            self.add_class(sub)

    def add_class(self, c: ast.Class) -> None:
        vars(self)["classes"][c.name] = c
        c.parent = self

    def remove_class(self, c: ast.Class) -> None:
        del vars(self)["classes"][c.name]
        c.parent = None

    def _extend(self, other: ast.Class) -> None:
        my_classes = vars(self)["classes"]
        other_classes = vars(other)["classes"]
        for class_name in other_classes:
            if class_name in my_classes:
                my_classes[class_name]._extend(other_classes[class_name])
            else:
                my_classes[class_name] = other_classes[class_name]

    def _update_parent_refs(self) -> None:
        for c in vars(self)["classes"].values():
            c.parent = self
            c._update_parent_refs()


def _path_to_class(path: Path) -> ast.Class | None:
    """Transform a filesystem path into a LazyParseClass"""
    if path.is_dir():
        try:
            package = next(path.glob("package.mo"))
            return LazyParseClass(path=package, name=path.parts[-1], type="package")
        except StopIteration:
            # Directories must contain a package.mo file (per Modelica spec)
            return None
    elif path.is_file():
        if path.suffix != ".mo" or path.stem == "package":
            return None
        return LazyParseClass(path=path, name=path.stem, type=_file_class_type(path))
    else:
        # Ignore anything else in MODELICAPATH
        return None


def _dir_to_tree(dir_: Path, parent: ast.Class) -> None:
    """Recursively walk a filesystem directory tree to a LazyParseClass tree"""
    if dir_class := _path_to_class(dir_.resolve()):
        parent.add_class(dir_class)
        for path in dir_.iterdir():
            if path.is_dir():
                _dir_to_tree(path, dir_class)
            elif child_class := _path_to_class(path.resolve()):
                dir_class.add_class(child_class)
    else:
        # dir_ is a MODELICAPATH root (no package.mo) — add its children directly to parent.
        # Top-level bare .mo files (e.g. Complex.mo in MSL-4.0.x/) and subdirectories that
        # are packages are all loaded as siblings under the parent tree.
        for path in sorted(dir_.iterdir()):
            if path.is_dir():
                _dir_to_tree(path, parent)
            elif child_class := _path_to_class(path.resolve()):
                parent.add_class(child_class)


def dir_to_tree(dir_: Path) -> ast.Tree:
    """Transform a MODELICAPATH directory into a Tree with LazyParseClass children"""
    root_tree = ast.Tree()
    _dir_to_tree(dir_, root_tree)
    return root_tree


def modelicapath_to_tree(dirs: list[str | Path]) -> ast.Tree:
    """Return ast.Tree for all directories in dirs list

    TODO: Add version handling (spec 18.8.3, 18.8.4)
    """
    modelicapath_tree = ast.Tree(name="root", type="MODELICAPATH")
    for dir_ in dirs:
        # Accept str or Path argument
        dir_ = Path(str(dir_))
        dir_.resolve()
        if not dir_.is_dir():
            raise ModelicaPathError(f"MODELICAPATH contains non-directory: {dir_}")
        dir_tree = dir_to_tree(dir_)
        modelicapath_tree.extend(dir_tree)
    return modelicapath_tree


# noinspection PyPep8Naming
class ASTListener(ModelicaListener):
    def __init__(self):
        self.file_node: ModelicaFile | None = None
        # Keyed by ANTLR parse-tree context objects; values are AST nodes or lists thereof.
        self.ast: dict[Any, Any] = {}
        self.class_nodes: deque[ast.Class] = deque([ast.Class()])
        self.comp_clause: ast.ComponentClause | None = None
        self.comp_clause1: ast.ComponentClause | None = None
        self.eq_sect: ast.EquationSection | None = None
        self.alg_sect: ast.AlgorithmSection | None = None
        self.symbol_nodes: list[ast.Symbol] = []
        self.eq_comment: str | None = None
        self.sym_count: int = 0
        self.in_extends = False
        self.in_class_spec_base = False
        self.in_redeclaration = False

    @property
    def class_node(self):
        return self.class_nodes[-1]

    # FILE ===========================================================

    def enterStored_definition(self, ctx: ModelicaParser.Stored_definitionContext):
        file_node = ModelicaFile()
        self.ast[ctx] = file_node
        self.file_node = file_node

    def exitStored_definition(self, ctx: ModelicaParser.Stored_definitionContext):
        assert self.file_node is not None
        within = []
        if ctx.component_reference() is not None:
            within = [self.ast[ctx.component_reference()]]
        self.file_node.within = within

        for class_node in [self.ast[e] for e in ctx.stored_definition_class()]:  # type: ignore[union-attr]
            self.ast[ctx].classes[class_node.name] = class_node
        self.file_node = self.ast[ctx]

    # CLASS ===========================================================

    def exitStored_definition_class(self, ctx: ModelicaParser.Stored_definition_classContext):
        class_node = self.ast[ctx.class_definition()]
        class_node.final = ctx.FINAL() is not None
        self.ast[ctx] = class_node

    def enterClass_definition(self, ctx: ModelicaParser.Class_definitionContext):
        class_node = ast.Class(
            name=ctx.getText(),  # Temporary name to help when debugging parsing
            parent=self.class_node,
        )
        class_node.encapsulated = ctx.ENCAPSULATED() is not None
        class_prefixes = ctx.class_prefixes()
        assert class_prefixes is not None
        class_node.partial = class_prefixes.PARTIAL() is not None
        class_node.type = class_prefixes.class_type().getText()  # type: ignore[union-attr]

        self.class_nodes.append(class_node)

        self.ast[ctx] = class_node

    def _prevent_builtin_name(self, name: str, ctx: ParserRuleContext) -> None:
        """Prevent the use of built-in names"""
        if name in ast.Class.BUILTIN:
            raise syntax_error_from_ctx(f"Predefined type {name} not allowed as identifier", ctx)

    def exitClass_definition(self, ctx: ModelicaParser.Class_definitionContext):
        class_node = self.class_nodes.pop()
        if class_node.name == ctx.getText():
            assert ctx.start is not None
            raise NotImplementedError(
                f"Unsupported class specifier at line {ctx.start.line}: "
                f"{ctx.class_specifier().getText()[:60]}"  # type: ignore[union-attr]
            )
        assert class_node.name is not None
        self._prevent_builtin_name(class_node.name, ctx)
        self.class_node.classes[class_node.name] = class_node

    def exitShort_class_definition(self, ctx):
        name = ctx.IDENT().getText()  # type: ignore[union-attr]
        self._prevent_builtin_name(name, ctx)
        self.ast[ctx] = ast.ShortClassDefinition(
            name=name,
            type=ctx.class_prefixes().class_type().getText(),  # type: ignore[union-attr]
            component=self.ast[ctx.component_reference()],
        )
        if ctx.class_modification() is not None:
            self.ast[ctx].class_modification = self.ast[ctx.class_modification()]
        else:
            self.ast[ctx].class_modification = ast.ClassModification()

    def exitClass_spec_comp(self, ctx: ModelicaParser.Class_spec_compContext):
        class_node = self.class_node
        class_node.name = ctx.IDENT()[0].getText()  # type: ignore[index, union-attr]
        class_node.comment = self.ast[ctx.string_comment()]

    def enterClass_spec_base(self, ctx: ModelicaParser.Class_spec_baseContext):
        self.in_extends = True
        self.in_class_spec_base = True

    def exitClass_spec_enum(self, ctx: ModelicaParser.Class_spec_enumContext):
        class_node = self.class_node
        class_node.name = ctx.IDENT().getText()  # type: ignore[union-attr]
        class_node.comment = self.ast[ctx.comment()]
        class_node.enumeration = True
        enum_list = ctx.enum_list()
        if enum_list is not None:
            for ordinal, lit in enumerate(enum_list.enumeration_literal(), start=1):  # type: ignore[union-attr]
                lit_name = lit.IDENT().getText()  # type: ignore[union-attr]
                enum_sym = ast.EnumerationLiteral(
                    name=lit_name,
                    type=ast.ComponentRef(name=class_node.name),
                    ordinal=ordinal,
                    class_modification=ast.ClassModification(),
                )
                enum_sym.comment = self.ast[lit.comment()]
                enum_sym.parent = class_node
                class_node.symbols[lit_name] = enum_sym

    def exitClass_spec_extends(self, ctx: ModelicaParser.Class_spec_extendsContext):
        class_node = self.class_node
        # IDENT()[0] is the name after `extends`; IDENT()[1] repeats it after `end`.
        # There is no component_reference() on this production.
        class_node.name = ctx.IDENT()[0].getText()  # type: ignore[index, union-attr]
        class_node.comment = self.ast[ctx.string_comment()]

        if ctx.class_modification() is not None:
            class_modification = self.ast[ctx.class_modification()]
        else:
            class_modification = ast.ClassModification()
        extends_clause = ast.ExtendsClause(
            component=ast.ComponentRef(name=class_node.name),
            class_modification=class_modification,
            # `class extends X` resolves X in the enclosing class (its inherited
            # members), not in this class itself — scope is the parent.
            scope=self.class_nodes[-2],
            is_class_extends=True,
        )
        class_node.extends.append(extends_clause)
        # No self.in_extends flag: unlike class_spec_base (which has no body),
        # this form has a real composition whose declarations must be added to
        # the class symbols normally through enterDeclaration.

    def exitClass_spec_base(self, ctx: ModelicaParser.Class_spec_baseContext):
        class_node = self.class_node
        class_node.name = ctx.IDENT().getText()  # type: ignore[union-attr]
        class_node.comment = self.ast[ctx.comment()]

        if ctx.class_modification() is not None:
            class_modification = self.ast[ctx.class_modification()]
        else:
            class_modification = ast.ClassModification()
        extends_clause = ast.ExtendsClause(
            component=self.ast[ctx.component_reference()],
            class_modification=class_modification,
            # Short class definition (class_spec_base) does not create a new scope
            scope=self.class_nodes[-2],
        )
        class_node.extends.append(extends_clause)
        class_node.is_short_class_definition = True
        self.in_extends = False
        self.in_class_spec_base = False

    def exitComposition(self, ctx: ModelicaParser.CompositionContext):
        # Set visibility if not the default public
        if len(ctx.epro):
            # Flatten the list of element lists from the parser into a list of elements
            element_list = [self.ast[e] for el in ctx.epro for e in el.element()]
            for element in element_list:
                if isinstance(element, ast.ComponentClause):
                    for symbol in element.symbol_list:
                        symbol.visibility = ast.Visibility.PROTECTED
                elif isinstance(element, (ast.ExtendsClause, ast.Class)):
                    element.visibility = ast.Visibility.PROTECTED
                # Visibility does not apply to ImportClause

        for eqlist in [self.ast[e] for e in ctx.equation_section()]:  # type: ignore[union-attr]
            if eqlist is not None:
                if eqlist.initial:
                    self.class_node.initial_equations += eqlist.equations
                else:
                    self.class_node.equations += eqlist.equations

        for alglist in [self.ast[e] for e in ctx.algorithm_section()]:  # type: ignore[union-attr]
            if alglist is not None:
                if alglist.initial:
                    self.class_node.initial_statements += alglist.statements
                else:
                    self.class_node.statements += alglist.statements

        if ctx.comp_annotation is not None:
            self.class_node.annotation = self.ast[ctx.comp_annotation]

    def exitArgument(self, ctx: ModelicaParser.ArgumentContext):
        # class_spec_base does not create a new scope
        if self.in_class_spec_base:
            scope = self.class_nodes[-2]
        else:
            scope = self.class_node
        argument = ast.ClassModificationArgument(scope=scope)
        if ctx.element_modification_or_replaceable() is not None:
            production = ctx.element_modification_or_replaceable()
            argument.value = self.ast[production]
            assert production is not None
            if production.element_replaceable() is not None:
                argument.redeclare = True
            else:
                argument.redeclare = False
        else:
            argument.value = self.ast[ctx.element_redeclaration()]
            argument.redeclare = True
        self.ast[ctx] = argument

    def exitArgument_list(self, ctx: ModelicaParser.Argument_listContext):
        self.ast[ctx] = [self.ast[a] for a in ctx.argument()]  # type: ignore[union-attr]

    def exitClass_modification(self, ctx: ModelicaParser.Class_modificationContext):
        arguments = []
        if ctx.argument_list() is not None:
            arguments = self.ast[ctx.argument_list()]
        self.ast[ctx] = ast.ClassModification(arguments=arguments)

    def enterEquation_section(self, ctx: ModelicaParser.Equation_sectionContext):
        eq_sect = ast.EquationSection(initial=ctx.INITIAL() is not None)
        self.ast[ctx] = eq_sect
        self.eq_sect = eq_sect

    def exitEquation_section(self, ctx: ModelicaParser.Equation_sectionContext):
        eq_sect = self.ast[ctx]
        eq_sect.equations.extend(self.ast[ctx.equation_block()])

    def exitEquation_block(self, ctx: ModelicaParser.Equation_blockContext):
        self.ast[ctx] = [self.ast[e] for e in ctx.equation()]  # type: ignore[union-attr]

    def exitStatement_block(self, ctx):
        self.ast[ctx] = [self.ast[e] for e in ctx.statement()]  # type: ignore[union-attr]

    def enterAlgorithm_section(self, ctx: ModelicaParser.Algorithm_sectionContext):
        alg_sect = ast.AlgorithmSection(initial=ctx.INITIAL() is not None)
        self.ast[ctx] = alg_sect
        self.alg_sect = alg_sect

    def exitAlgorithm_section(self, ctx: ModelicaParser.Algorithm_sectionContext):
        alg_sect = self.ast[ctx]
        alg_sect.statements.extend(self.ast[ctx.statement_block()])

    # EQUATION ===========================================================

    def enterEquation(self, ctx: ModelicaParser.EquationContext):
        pass

    def exitEquation(self, ctx):
        self.ast[ctx] = self.ast[ctx.equation_options()]
        try:
            self.ast[ctx].comment = self.ast[ctx.comment()]
        except AttributeError:
            pass

    def exitEquation_simple(self, ctx: ModelicaParser.Equation_simpleContext):
        self.ast[ctx] = ast.Equation(
            left=self.ast[ctx.simple_expression()],
            right=self.ast[ctx.expression()],
        )

    def exitEquation_if(self, ctx: ModelicaParser.Equation_ifContext):
        self.ast[ctx] = self.ast[ctx.if_equation()]

    def exitEquation_for(self, ctx: ModelicaParser.Equation_forContext):
        self.ast[ctx] = self.ast[ctx.for_equation()]

    def exitEquation_connect_clause(self, ctx: ModelicaParser.Equation_connect_clauseContext):
        self.ast[ctx] = self.ast[ctx.connect_clause()]

    def exitArgument_expression(self, ctx: ModelicaParser.Argument_expressionContext):
        self.ast[ctx] = self.ast[ctx.expression()]

    def exitIf_equation(self, ctx: ModelicaParser.If_equationContext):
        blocks = [self.ast[b] for b in ctx.blocks]
        conditions = [self.ast[c] for c in ctx.conditions]
        if len(conditions) == len(blocks) - 1:
            conditions.append(True)
        self.ast[ctx] = ast.IfEquation(conditions=conditions, blocks=blocks)

    def exitWhen_equation(self, ctx: ModelicaParser.When_equationContext):
        blocks = [self.ast[b] for b in ctx.blocks]
        conditions = [self.ast[c] for c in ctx.conditions]
        if len(conditions) == len(blocks) - 1:
            conditions.append(True)
        self.ast[ctx] = ast.WhenEquation(conditions=conditions, blocks=blocks)

    def exitFor_equation(self, ctx: ModelicaParser.For_equationContext):
        self.ast[ctx] = ast.ForEquation(
            indices=self.ast[ctx.for_indices()],
            equations=self.ast[ctx.equation_block()],
        )

    def exitConnect_clause(self, ctx: ModelicaParser.Connect_clauseContext):
        self.ast[ctx] = ast.ConnectClause(
            left=self.ast[ctx.component_reference()[0]],  # type: ignore[index]
            right=self.ast[ctx.component_reference()[1]],  # type: ignore[index]
        )

    # STATEMENT ==========================================================

    # TODO:
    # - Missing statement types:
    #   + Function calls (inside current function).
    #   + Break
    #   + Return
    #   + While
    #   + When? (also in equation missing)

    def enterStatement(self, ctx: ModelicaParser.StatementContext):
        pass

    def exitStatement(self, ctx: ModelicaParser.StatementContext):
        self.ast[ctx] = self.ast[ctx.statement_options()]
        try:
            self.ast[ctx].comment = self.ast[ctx.comment()]
        except AttributeError:
            pass

    def exitStatement_component_reference(
        self, ctx: ModelicaParser.Statement_component_referenceContext
    ):
        comp_ref = self.ast[ctx.component_reference()]
        if ctx.expression() is not None:
            self.ast[ctx] = ast.AssignmentStatement(
                left=[comp_ref],
                right=self.ast[ctx.expression()],
            )
        else:
            # Standalone function call: f(args)
            self.ast[ctx] = ast.AssignmentStatement(
                left=[],
                right=ast.Expression(
                    operator=comp_ref,
                    operands=self._function_call_operands(ctx.function_call_args()),
                ),
            )

    def exitStatement_component_function(
        self, ctx: ModelicaParser.Statement_component_functionContext
    ):
        # Build the left-hand side from output_expression_list, which permits omitted
        # outputs (MLS §B.2.6.5).  Each comma terminates a slot; omitted slots become
        # None so discard positions are preserved for backends.
        oel = ctx.output_expression_list()
        assert oel is not None
        left = []
        current = None
        for child in oel.getChildren():
            if isinstance(child, ModelicaParser.ExpressionContext):
                current = self.ast[child]
            else:  # TerminalNode ',' — close the current slot
                left.append(current)
                current = None
        left.append(current)

        right = ast.Expression(
            operator=self.ast[ctx.component_reference()],
            operands=self._function_call_operands(ctx.function_call_args()),
        )

        self.ast[ctx] = ast.AssignmentStatement(left=left, right=right)

    def exitStatement_if(self, ctx: ModelicaParser.Statement_ifContext):
        self.ast[ctx] = self.ast[ctx.if_statement()]

    def exitStatement_for(self, ctx: ModelicaParser.Statement_forContext):
        self.ast[ctx] = self.ast[ctx.for_statement()]

    def exitStatement_while(self, ctx: ModelicaParser.Statement_whileContext):
        self.ast[ctx] = self.ast[ctx.while_statement()]

    def exitStatement_break(self, ctx: ModelicaParser.Statement_breakContext):
        self.ast[ctx] = ast.Primary(value="break")

    def exitStatement_return(self, ctx: ModelicaParser.Statement_returnContext):
        self.ast[ctx] = ast.Primary(value="return")

    def exitStatement_when(self, ctx: ModelicaParser.Statement_whenContext):
        self.ast[ctx] = self.ast[ctx.when_statement()]

    def exitIf_statement(self, ctx: ModelicaParser.If_statementContext):
        blocks = [self.ast[b] for b in ctx.blocks]
        conditions = [self.ast[c] for c in ctx.conditions]
        if len(conditions) == len(blocks) - 1:
            conditions.append(True)
        self.ast[ctx] = ast.IfStatement(conditions=conditions, blocks=blocks)

    def exitWhen_statement(self, ctx: ModelicaParser.When_statementContext):
        blocks = [self.ast[b] for b in ctx.blocks]
        conditions = [self.ast[c] for c in ctx.conditions]
        if len(conditions) == len(blocks) - 1:
            conditions.append(True)
        self.ast[ctx] = ast.WhenStatement(conditions=conditions, blocks=blocks)

    def exitFor_statement(self, ctx: ModelicaParser.For_statementContext):
        self.ast[ctx] = ast.ForStatement(
            indices=self.ast[ctx.for_indices()],
            statements=self.ast[ctx.statement_block()],
        )

    def exitWhile_statement(self, ctx: ModelicaParser.While_statementContext):
        self.ast[ctx] = ast.WhileStatement(
            condition=self.ast[ctx.condition],
            statements=self.ast[ctx.statement_block()],
        )

    # EXPRESSIONS ===========================================================

    def exitSimple_expression(self, ctx: ModelicaParser.Simple_expressionContext):
        exprs: list[Any] = _cast(
            list, ctx.expr()
        )  # stubs type as ExprContext|None, runtime is list
        if len(exprs) > 1:
            if len(exprs) > 2:
                step = self.ast[exprs[2]]
            else:
                step = ast.Primary(value=1)
            self.ast[ctx] = ast.Slice(start=self.ast[exprs[0]], stop=self.ast[exprs[1]], step=step)
        else:
            self.ast[ctx] = self.ast[exprs[0]]

    def exitExpression_simple(self, ctx: ModelicaParser.Expression_simpleContext):
        self.ast[ctx] = self.ast[ctx.simple_expression()]

    def exitExpression_if(self, ctx: ModelicaParser.Expression_ifContext):
        all_expr = [self.ast[s] for s in ctx.expression()]  # type: ignore[union-attr]
        # Note that an else block is guaranteed to exist.
        conditions = all_expr[:-1:2]
        expressions = all_expr[1::2] + all_expr[-1:]

        self.ast[ctx] = ast.IfExpression(conditions=conditions, expressions=expressions)

    def exitExpr_primary(self, ctx: ModelicaParser.Expr_primaryContext):
        self.ast[ctx] = self.ast[ctx.primary()]

    def exitExpr_add(self, ctx: ModelicaParser.Expr_addContext):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,  # type: ignore[union-attr]
            operands=[self.ast[e] for e in ctx.expr()],  # type: ignore[union-attr]
        )

    def exitExpr_exp(self, ctx: ModelicaParser.Expr_expContext):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,  # type: ignore[union-attr]
            operands=[self.ast[e] for e in ctx.primary()],  # type: ignore[union-attr]
        )

    def exitExpr_mul(self, ctx: ModelicaParser.Expr_mulContext):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,  # type: ignore[union-attr]
            operands=[self.ast[e] for e in ctx.expr()],  # type: ignore[union-attr]
        )

    def exitExpr_rel(self, ctx: ModelicaParser.Expr_relContext):
        self.ast[ctx] = ast.Expression(
            operator=ctx.op.text,  # type: ignore[union-attr]
            operands=[self.ast[e] for e in ctx.expr()],  # type: ignore[union-attr]
        )

    def exitExpr_not(self, ctx: ModelicaParser.Expr_notContext):
        self.ast[ctx] = ast.Expression(operator="not", operands=[self.ast[ctx.expr()]])

    def exitExpr_and(self, ctx: ModelicaParser.Expr_andContext):
        self.ast[ctx] = ast.Expression(
            operator="and",
            operands=[self.ast[e] for e in ctx.expr()],  # type: ignore[union-attr]
        )

    def exitExpr_or(self, ctx: ModelicaParser.Expr_orContext):
        self.ast[ctx] = ast.Expression(
            operator="or",
            operands=[self.ast[e] for e in ctx.expr()],  # type: ignore[union-attr]
        )

    def exitExpr_signed(self, ctx: ModelicaParser.Expr_signedContext):
        self.ast[ctx] = ast.Expression(operator=ctx.op.text, operands=[self.ast[ctx.expr()]])  # type: ignore[union-attr]

    def exitSubscript(self, ctx: ModelicaParser.SubscriptContext):
        if ctx.expression() is not None:
            self.ast[ctx] = self.ast[ctx.expression()]
        else:
            self.ast[ctx] = ast.Slice()

    def exitArray_subscripts(self, ctx: ModelicaParser.Array_subscriptsContext):
        self.ast[ctx] = [self.ast[s] for s in ctx.subscript()]  # type: ignore[union-attr]

    def exitFor_index(self, ctx):
        self.ast[ctx] = ast.ForIndex(
            name=ctx.IDENT().getText(),  # type: ignore[union-attr]
            expression=self.ast[ctx.expression()],
        )

    def exitFor_indices(self, ctx: ModelicaParser.For_indicesContext):
        self.ast[ctx] = [self.ast[s] for s in ctx.for_index()]  # type: ignore[union-attr]

    # PRIMARY ===========================================================

    def exitPrimary_unsigned_number(self, ctx: ModelicaParser.Primary_unsigned_numberContext):
        number_string = ctx.getText()
        try:
            val = int(number_string)
        except ValueError:
            val = float(number_string)

        self.ast[ctx] = ast.Primary(value=val)

    def exitPrimary_string(self, ctx: ModelicaParser.Primary_stringContext):
        val = ctx.getText()
        assert val.startswith('"') and val.endswith('"')
        self.ast[ctx] = ast.Primary(value=val[1:-1])

    def exitPrimary_false(self, ctx: ModelicaParser.Primary_falseContext):
        self.ast[ctx] = ast.Primary(value=False)

    def exitPrimary_true(self, ctx: ModelicaParser.Primary_trueContext):
        self.ast[ctx] = ast.Primary(value=True)

    def _function_argument_ast(self, arg_ctx):
        """Convert one function_argument context to an AST node (MLS §B.2.7.11).

        Handles both plain ``expression`` arguments and ``function name(...)``
        partial-application arguments.  Argument names within a partial
        application are not retained, matching the handling of ordinary
        named arguments.
        """
        if isinstance(arg_ctx, ModelicaParser.Argument_functionContext):
            named = arg_ctx.named_arguments()
            # TODO: argument names (e.g. `A=`, `w=`) are dropped here; only
            #       the bound value expressions are kept.  Retaining names
            #       requires named-argument support in the AST (see also
            #       _function_call_operands, which likewise ignores top-level
            #       `named_arguments`).
            operands = (
                [
                    self._function_argument_ast(na.function_argument())  # type: ignore[union-attr]
                    for na in named.named_argument()  # type: ignore[union-attr]
                ]
                if named is not None
                else []
            )
            return ast.Expression(
                operator=ast.ComponentRef.from_string(arg_ctx.name().getText()),  # type: ignore[union-attr]
                operands=operands,
            )
        return self.ast[arg_ctx.expression()]

    def _function_call_operands(self, func_call_args_ctx):
        """Safely extract expression operands from a function_call_args context."""
        func_args = func_call_args_ctx.function_arguments()
        if func_args is None:
            return []
        return [self._function_argument_ast(x) for x in func_args.function_argument()]  # type: ignore[union-attr]

    def exitPrimary_function(self, ctx: ModelicaParser.Primary_functionContext):
        # TODO: Could possible be cleaner if we let the expression in the ast bubble up.
        #       E.g. self.ast[x] below, instead of self.ast[x.expression].
        self.ast[ctx] = ast.Expression(
            operator=self.ast[ctx.component_reference()],
            operands=self._function_call_operands(ctx.function_call_args()),
        )

    def exitPrimary_derivative(self, ctx: ModelicaParser.Primary_derivativeContext):
        self.ast[ctx] = ast.Expression(
            operator="der",
            operands=self._function_call_operands(ctx.function_call_args()),
        )
        # TODO 'state' is not a standard prefix;  disable this for now as it does not work
        # when differentiating states defined in superclasses.
        # if 'state' not in self.class_node.symbols[comp_name].prefixes:
        #    self.class_node.symbols[comp_name].prefixes += ['state']

    def exitPrimary_initial(self, ctx: ModelicaParser.Primary_initialContext):
        self.ast[ctx] = ast.Expression(
            operator="initial",
            operands=self._function_call_operands(ctx.function_call_args()),
        )

    def exitPrimary_end(self, ctx: ModelicaParser.Primary_endContext):
        self.ast[ctx] = ast.ComponentRef(name="end", indices=[[None]], child=[])

    def exitType_specifier_element(self, ctx: ModelicaParser.Type_specifier_elementContext):
        self.ast[ctx] = ast.ComponentRef(
            name=ctx.IDENT().getText(), indices=[[None]], child=[]  # type: ignore[union-attr]
        )

    def exitType_specifier(self, ctx: ModelicaParser.Type_specifierContext):
        for element in reversed([self.ast[x] for x in ctx.type_specifier_element()]):  # type: ignore[union-attr]
            if ctx in self.ast:
                element.child = [self.ast[ctx]]
            self.ast[ctx] = element
        # A leading dot means global name lookup (MLS 5.3.3); represent it as an
        # empty-name head element so str() round-trips (".A.B") and lookup can detect it
        if ctx.getChild(0).getText() == ".":  # type: ignore[union-attr]
            self.ast[ctx] = ast.ComponentRef(name="", child=[self.ast[ctx]])

    def exitComponent_reference_element(
        self, ctx: ModelicaParser.Component_reference_elementContext
    ):
        if ctx.array_subscripts() is not None:
            arr_sub = ctx.array_subscripts()
            assert arr_sub is not None
            indices = [[self.ast[x] for x in arr_sub.subscript()]]  # type: ignore[union-attr]
        else:
            indices = [[None]]
        self.ast[ctx] = ast.ComponentRef(
            name=ctx.IDENT().getText(), indices=indices, child=[]  # type: ignore[union-attr]
        )

    def exitComponent_reference(self, ctx: ModelicaParser.Component_referenceContext):
        for element in reversed([self.ast[x] for x in ctx.component_reference_element()]):  # type: ignore[union-attr]
            if ctx in self.ast:
                element.child = [self.ast[ctx]]
            self.ast[ctx] = element
        # A leading dot means global name lookup (MLS 5.3.3); represent it as an
        # empty-name head element so str() round-trips (".A.B") and lookup can detect it
        if ctx.getChild(0).getText() == ".":  # type: ignore[union-attr]
            self.ast[ctx] = ast.ComponentRef(name="", child=[self.ast[ctx]])

    def exitPrimary_component_reference(
        self, ctx: ModelicaParser.Primary_component_referenceContext
    ):
        self.ast[ctx] = self.ast[ctx.component_reference()]

    def exitPrimary_output_expression_list(
        self, ctx: ModelicaParser.Primary_output_expression_listContext
    ):
        oel = ctx.output_expression_list()
        assert oel is not None
        self.ast[ctx] = [self.ast[x] for x in oel.expression()]  # type: ignore[union-attr]
        # Collapse lists containing a single expression
        if len(self.ast[ctx]) == 1:
            self.ast[ctx] = self.ast[ctx][0]

    def exitPrimary_expression_list(self, ctx: ModelicaParser.Primary_expression_listContext):
        rows = [
            ast.Array(values=[self.ast[x] for x in expr_list.expression()])  # type: ignore[union-attr]
            for expr_list in ctx.expression_list()  # type: ignore[union-attr]
        ]
        self.ast[ctx] = rows[0] if len(rows) == 1 else ast.Array(values=rows)

    def exitPrimary_function_arguments(self, ctx: ModelicaParser.Primary_function_argumentsContext):
        # TODO: This does not support for generators yet.
        #       Only expressions are supported, e.g. {1.0, 2.0, 3.0}.
        func_args = ctx.function_arguments()
        assert func_args is not None
        v = [self.ast[x.expression()] for x in func_args.function_argument()]  # type: ignore[union-attr]
        self.ast[ctx] = ast.Array(values=v)

    def exitEquation_function(self, ctx: ModelicaParser.Equation_functionContext):
        self.ast[ctx] = ast.Function(
            name=ctx.name().getText(),  # type: ignore[union-attr]
            arguments=self._function_call_operands(ctx.function_call_args()),
        )

    def exitEquation_when(self, ctx: ModelicaParser.Equation_whenContext):
        self.ast[ctx] = self.ast[ctx.when_equation()]

    # COMPONENTS ===========================================================

    def exitElement_list(self, ctx: ModelicaParser.Element_listContext):
        self.ast[ctx] = [self.ast[e] for e in ctx.element()]  # type: ignore[union-attr]

    def exitElement(self, ctx: ModelicaParser.ElementContext):
        self.ast[ctx] = self.ast[ctx.getChild(ctx.getAltNumber())]

    # TODO: Rewrite this as follows:
    # Simple Name Lookup (the first step of name lookup) is by import name as follows:
    # Import name of A.B.C is C, so ast.Class.imports dictionary key is C for simple name lookup.
    # Import name of D = A.B.C is D, so D is the dictionary key.
    # Import names of A.B.{C,D} are C and D, both added as keys.
    # The special case of import A.B.* is processed during lookup (it is uncommon and may be expensive);
    # in this case the dictionary key is "*" and the value is a list containing all of the
    # A.B, E.F, G, ... package refs, each of which is prepended to the import name for
    # Global Name Lookup (MLS 5.3.3). We could choose to make the Class.imports values
    # always a list to make it homogeneous, with most cases being len() == 1,
    # but this might make most cases slower. I'm leaning toward heterogeneous values.
    # The Class.imports dictionary values are the fully qualified ComponentRef for
    # Composite Name Lookup (see MLS), but imports should be skipped for this lookup.
    # All Class.imports values are ComponentRef, or List[ComponentRef] for A.B.* type imports.
    # Clean up the logic (inheritance or different import clause classes?, ANTLR rule labels?)
    # Checks:
    # 1. ast.Class.imports key is not already there (see spec 13.2.2 last bullet).
    #    This check should be in a separate function to be used also during the just-in-time
    #    A.B.* processing.
    # 2. For A.B.C or A.B.*, A.B must be a package.
    def exitImport_clause(self, ctx: ModelicaParser.Import_clauseContext):
        import_clause = ast.ImportClause()
        self.ast[ctx] = import_clause
        import_clause.components = [self.ast[ctx.component_reference()]]
        if ctx.IDENT() is not None:
            import_clause.short_name = ctx.IDENT().getText()  # type: ignore[union-attr]
        else:
            import_list = ctx.import_list()
            if import_list is not None:
                package_name = import_clause.components.pop()
                # Append list of names to package_name to get fully qualified name(s)
                # Skip the comma separators in import_list.children
                for ident in import_list.children[::2]:  # type: ignore[index]
                    qualified_name = package_name.concatenate(
                        package_name.from_string(ident.getText())  # type: ignore[union-attr]
                    )
                    import_clause.components.append(qualified_name)
            elif ctx.getChildCount() > 3:
                import_clause.unqualified = True
        if import_clause.short_name:
            # import_clause instead of comp_ref signifies short_name
            self._check_not_already_imported(import_clause.short_name, ctx)
            self.class_node.imports[import_clause.short_name] = import_clause
        elif import_clause.unqualified:
            # Postpone processing this uncommon case until actually needed
            # In this case import_clause.components contains list of packages of all unqualified imports
            if "*" not in self.class_node.imports:
                self.class_node.imports["*"] = import_clause
            else:
                existing = self.class_node.imports["*"]
                assert isinstance(existing, ast.ImportClause)
                existing.components.append(import_clause.components[0])
        else:
            # Simple case, fast lookup
            for comp in import_clause.components:
                name = comp.to_tuple()[-1]
                self._check_not_already_imported(name, ctx)
                self.class_node.imports[name] = comp

    def _check_not_already_imported(self, import_name: str, ctx: ParserRuleContext) -> None:
        """Check for import name clashes"""
        if import_name in self.class_node.imports:
            raise syntax_error_from_ctx(f"{import_name} already imported", ctx)

    def enterExtends_clause(self, ctx: ModelicaParser.Extends_clauseContext):
        self.in_extends = True

    def exitExtends_clause(self, ctx: ModelicaParser.Extends_clauseContext):
        if ctx.class_modification() is not None:
            class_modification = self.ast[ctx.class_modification()]
        else:
            class_modification = ast.ClassModification()
        self.ast[ctx] = ast.ExtendsClause(
            component=self.ast[ctx.component_reference()],
            class_modification=class_modification,
            scope=self.class_node,
        )
        self.class_node.extends += [self.ast[ctx]]

        self.in_extends = False

    def exitRegular_element(self, ctx: ModelicaParser.Regular_elementContext):
        if ctx.comp_elem is not None:
            self.ast[ctx] = self.ast[ctx.comp_elem]
            for sym in self.ast[ctx].symbol_list:
                if ctx.INNER() is not None:
                    sym.inner = True
                if ctx.OUTER() is not None:
                    sym.outer = True
                if ctx.FINAL() is not None:
                    sym.final = True
        else:
            self.ast[ctx] = self.ast[ctx.class_elem]

    def exitReplaceable_element(self, ctx: ModelicaParser.Replaceable_elementContext):
        if ctx.comp_elem is not None:
            self.ast[ctx] = self.ast[ctx.comp_elem]
            for sym in self.ast[ctx].symbol_list:
                sym.replaceable = True
                if ctx.INNER() is not None:
                    sym.inner = True
                if ctx.OUTER() is not None:
                    sym.outer = True
                if ctx.FINAL() is not None:
                    sym.final = True
        else:
            self.ast[ctx] = self.ast[ctx.class_elem]
        self.ast[ctx].replaceable = True

    def enterComponent_clause(self, ctx: ModelicaParser.Component_clauseContext):
        prefixes = ctx.type_prefix().getText().split(" ")  # type: ignore[union-attr]
        if prefixes[0] == "":
            prefixes = []
        self.ast[ctx] = ast.ComponentClause(
            prefixes=prefixes,
        )
        self.comp_clause = self.ast[ctx]

    def enterComponent_clause1(self, ctx: ModelicaParser.Component_clause1Context):
        prefixes = ctx.type_prefix().getText().split(" ")  # type: ignore[union-attr]
        if prefixes[0] == "":
            prefixes = []
        self.ast[ctx] = ast.ComponentClause(
            prefixes=prefixes,
        )
        self.comp_clause1 = self.ast[ctx]

    def exitComponent_clause(self, ctx: ModelicaParser.Component_clauseContext):
        clause = self.ast[ctx]
        # The component clause and all its symbols share the same type.
        # However, the type will only be turned into a component reference
        # somewhere between the enterDeclaration and exitDeclaration functions
        # of the symbols. Therefore, we need to keep the component clause's
        # type, and all its symbols' types, pointing at the same empty
        # (ComponentRef) object until we can fill it.
        clause.type.__dict__.update(self.ast[ctx.type_specifier()].__dict__)
        assert self.comp_clause is not None
        if ctx.array_subscripts() is not None:
            clause.dimensions = [self.ast[ctx.array_subscripts()]]
            for sym in self.comp_clause.symbol_list:
                s = self.class_node.symbols[sym.name]
                s.dimensions = clause.dimensions

        # We make sure that all references to the objects are unique per
        # symbol making copies. Note that if there is only one symbol in the
        # component clause, it is already unique.
        for sym in self.comp_clause.symbol_list[1:]:
            s = self.class_node.symbols[sym.name]
            s.dimensions = list(s.dimensions)
            s.prefixes = list(s.prefixes)
            s.type = copy.deepcopy(clause.type)
        self.comp_clause = None

    def exitComponent_clause1(self, ctx: ModelicaParser.Component_clause1Context):
        clause = self.ast[ctx]
        clause.type.__dict__.update(self.ast[ctx.type_specifier()].__dict__)
        self.comp_clause1 = None

    def enterComponent_declaration(self, ctx: ModelicaParser.Component_declarationContext):
        sym = ast.Symbol(order=self.sym_count, parent=self.class_node)
        self.sym_count += 1
        self.ast[ctx] = sym
        self.symbol_nodes.append(sym)
        assert self.comp_clause is not None
        self.comp_clause.symbol_list += [sym]

    def enterComponent_declaration1(self, ctx: ModelicaParser.Component_declaration1Context):
        sym = ast.Symbol(order=self.sym_count, parent=self.class_node)
        self.sym_count += 1
        self.ast[ctx] = sym
        self.symbol_nodes.append(sym)
        assert self.comp_clause1 is not None
        self.comp_clause1.symbol_list += [sym]

    def exitComponent_declaration(self, ctx: ModelicaParser.Component_declarationContext):
        self.ast[ctx].comment = self.ast[ctx.comment()]
        self.symbol_nodes.pop()

    def exitComponent_declaration1(self, ctx: ModelicaParser.Component_declaration1Context):
        self.ast[ctx].comment = self.ast[ctx.comment()]
        self.symbol_nodes.pop()

    def enterDeclaration(self, ctx: ModelicaParser.DeclarationContext):
        sym = self.symbol_nodes[-1]
        dimensions = None
        comp_clause = self.comp_clause1 if self.comp_clause1 else self.comp_clause
        assert comp_clause is not None
        if comp_clause.dimensions is not None:
            dimensions = comp_clause.dimensions
        sym.name = ctx.IDENT().getText()  # type: ignore[union-attr]
        self._prevent_builtin_name(sym.name, ctx)
        if dimensions is not None:
            sym.dimensions = dimensions
        sym.prefixes = comp_clause.prefixes
        sym.type = comp_clause.type

        # Declarations can also occur in extends_clause, class_spec_base (also an extends),
        # and redeclare, in which case we do not have to add it to the class's symbols.
        if not (self.in_extends or self.in_redeclaration):
            if sym.name in self.class_node.symbols:
                raise IOError(sym.name, "already defined")
            self.class_node.symbols[sym.name] = sym

    def exitDeclaration(self, ctx: ModelicaParser.DeclarationContext):
        sym = self.symbol_nodes[-1]
        if ctx.array_subscripts() is not None:
            sym.dimensions = [self.ast[ctx.array_subscripts()]]
        if ctx.modification() is not None:
            for mod in self.ast[ctx.modification()]:
                if isinstance(mod, ast.ClassModification):
                    sym.class_modification = mod
                else:
                    # Assignment of value, which we turn into a modification here.
                    vmod_arg = ast.ClassModificationArgument(scope=self.class_node)
                    em = ast.ElementModification()
                    em.component = ast.ComponentRef(name="value")
                    em.modifications = [mod]
                    vmod_arg.value = em

                    if sym.class_modification is None:
                        sym_mod = ast.ClassModification()
                        sym_mod.arguments.append(vmod_arg)
                        sym.class_modification = sym_mod
                    else:
                        sym.class_modification.arguments.append(vmod_arg)
        else:
            sym.class_modification = ast.ClassModification()

    def exitElement_modification(self, ctx: ModelicaParser.Element_modificationContext):
        component = self.ast[ctx.component_reference()]
        if ctx.modification() is not None:
            modifications = self.ast[ctx.modification()]
        else:
            modifications = []

        self.ast[ctx] = ast.ElementModification(component=component, modifications=modifications)

    def exitModification_class(self, ctx: ModelicaParser.Modification_classContext):
        self.ast[ctx] = [self.ast[ctx.class_modification()]]
        if ctx.expression() is not None:
            self.ast[ctx] += [self.ast[ctx.expression()]]

    def exitModification_assignment(self, ctx: ModelicaParser.Modification_assignmentContext):
        self.ast[ctx] = [self.ast[ctx.expression()]]

    def exitModification_assignment2(self, ctx: ModelicaParser.Modification_assignment2Context):
        self.ast[ctx] = [self.ast[ctx.expression()]]

    def exitElement_replaceable(self, ctx: ModelicaParser.Element_replaceableContext):
        if ctx.component_clause1() is not None:
            self.ast[ctx] = self.ast[ctx.component_clause1()]
            self.ast[ctx].symbol_list[0].replaceable = True
        else:
            self.ast[ctx] = self.ast[ctx.short_class_definition()]
        self.ast[ctx].replaceable = True

    def exitElement_modification_or_replaceable(
        self, ctx: ModelicaParser.Element_modification_or_replaceableContext
    ):
        if ctx.element_modification() is not None:
            self.ast[ctx] = self.ast[ctx.element_modification()]
        else:
            self.ast[ctx] = self.ast[ctx.element_replaceable()]

    def enterElement_redeclaration(self, ctx: ModelicaParser.Element_redeclarationContext):
        self.in_redeclaration = True

    def exitElement_redeclaration(self, ctx: ModelicaParser.Element_redeclarationContext):
        if ctx.component_clause1() is not None:
            self.ast[ctx] = self.ast[ctx.component_clause1()]
        elif ctx.element_replaceable() is not None:
            self.ast[ctx] = self.ast[ctx.element_replaceable()]
        else:
            self.ast[ctx] = self.ast[ctx.short_class_definition()]
        self.in_redeclaration = False

    # COMMENTS ==============================================================

    def exitComment(self, ctx: ModelicaParser.CommentContext):
        self.ast[ctx] = self.ast[ctx.string_comment()]

    def exitString_comment(self, ctx: ModelicaParser.String_commentContext):
        self.ast[ctx] = ctx.getText()[1:-1]

    # ANNOTATIONS ==========================================================

    def exitAnnotation(self, ctx: ModelicaParser.AnnotationContext):
        self.ast[ctx] = self.ast[ctx.class_modification()]


# UTILITY FUNCTIONS ========================================================
def file_to_tree(f: ModelicaFile) -> ast.Tree:
    # TODO: We can only insert where classes exist. For example, if we have a
    # within statement, we have to check if the nodes of the within statement
    # are actually in the tree, and if not raise an exception.
    root = ast.Tree()
    insert_node = root
    if f.within:
        for p in f.within[0].to_tuple():
            package = ast.Class(name=p, type="package")
            insert_node.classes[p] = package
            insert_node = package

    insert_node.classes.update(f.classes)

    root.update_parent_refs()

    return root


class ModelicaParserErrorListener(ErrorListener):
    """Raise a ModelicaSyntaxError when ANTLR finds a syntax error"""

    def syntaxError(self, recognizer, offending_symbol, line, column, msg, e):  # type: ignore[override]
        """Handle syntax errors from both the lexer and parser"""
        # There is probably a better way to get the text
        if isinstance(recognizer, antlr4.Lexer):
            text = recognizer.inputStream.strdata.split("\n")
        else:
            text = offending_symbol.source[1].strdata.split("\n")
        error_text = text[line - 1]
        # ANTLR column is 0-based, but Python is 1-based
        column += 1
        # End line and column were were added in Python 3.10, previous will ignore
        # Set to same as start to get Python >= 3.10 stack trace point to the right place
        end_line = line
        end_column = column
        info: tuple = "input", line, column, error_text, end_line, end_column
        raise ModelicaSyntaxError(msg, info)


def _parse_text(text: str, trace: bool) -> ModelicaFile:
    """Parse Modelica code given in text, return ModelicaFile"""
    input_stream = antlr4.InputStream(text)
    lexer = ModelicaLexer(input_stream)
    stream = antlr4.CommonTokenStream(lexer)
    parser = ModelicaParser(stream)
    parser.setTrace(trace)
    # parser.buildParseTrees = False
    listener = ModelicaParserErrorListener()
    lexer.removeErrorListeners()
    lexer.addErrorListener(listener)
    parser.removeErrorListeners()
    parser.addErrorListener(listener)
    parse_tree = parser.stored_definition()
    ast_listener = ASTListener()
    parse_walker = antlr4.ParseTreeWalker()
    parse_walker.walk(ast_listener, parse_tree)
    assert ast_listener.file_node is not None
    return ast_listener.file_node


def _parse(text: str, trace: bool) -> ast.Tree:
    """Parse Modelica code given in text, return Tree"""
    modelica_file = _parse_text(text, trace)
    return file_to_tree(modelica_file)


def _microseconds_since_epoch(timedelta_: timedelta | None = None) -> int:
    if timedelta_ is None:
        timedelta_ = timedelta()
    return time.time_ns() // 1000 + int(timedelta_.total_seconds() * 1e6)


def _check_database_structure(conn: sqlite3.Connection):
    """
    Function to check if the existing database file matches the expected table structure
    """
    cursor = conn.cursor()

    cursor.execute("BEGIN IMMEDIATE;")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='models'")
    table_exists = cursor.fetchone()
    table_correct = False

    if table_exists:
        cursor.execute("PRAGMA table_info('models')")
        columns = cursor.fetchall()
        expected_columns = [
            (0, "txt_hash", "TEXT", 0, None, 1),
            (1, "pymoca_version", "TEXT", 0, None, 2),
            (2, "data", "BLOB", 0, None, 0),
            (3, "last_hit", "TIMESTAMP INTEGER", 0, None, 0),
        ]

        if columns != expected_columns:
            logger.warning("Model text cache table layout didn't match, recreating")
            table_correct = False
        else:
            table_correct = True

    if not table_correct:
        cursor.execute("DROP TABLE IF EXISTS models")

        logger.debug("Creating model text cache table in database")
        cursor.execute(
            """
            CREATE TABLE models (
                txt_hash TEXT,
                pymoca_version TEXT,
                data BLOB,
                last_hit TIMESTAMP INTEGER,
                PRIMARY KEY (txt_hash, pymoca_version)
            )
        """
        )

    # Index last_hit so the prune (DELETE ... WHERE last_hit < ?) is
    # index-assisted instead of a full-table scan. Without this, pruning a
    # multi-GB cache scans the entire table on every worker startup. Building
    # the index on a pre-existing cache is a one-time cost on first upgrade.
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_last_hit ON models (last_hit)")

    conn.commit()

    # For metadata we check if the table layout is correct, but also whether
    # the metadata keys exist.
    cursor.execute("BEGIN IMMEDIATE;")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
    metadata_table_exists = cursor.fetchone()
    metadata_table_correct = False

    if metadata_table_exists:
        cursor.execute("PRAGMA table_info('metadata')")
        columns = cursor.fetchall()
        expected_columns = [
            (0, "key", "TEXT", 0, None, 1),
            (1, "value", "TEXT", 0, None, 0),
        ]

        if columns != expected_columns:
            logger.warning("Metadata table layout didn't match, recreating")
            metadata_table_correct = False
        else:
            metadata_table_correct = True

    if not metadata_table_correct:
        cursor.execute("DROP TABLE IF EXISTS metadata")

        logger.debug("Creating metadata table in database")
        cursor.execute(
            """
            CREATE TABLE metadata (
                key TEXT,
                value TEXT,
                PRIMARY KEY (key)
            )
        """
        )
    conn.commit()

    cursor.execute("BEGIN IMMEDIATE;")
    cursor.execute(
        "INSERT OR IGNORE INTO metadata (key, value) VALUES (?, ?)",
        ("created_at", _microseconds_since_epoch()),
    )

    cursor.execute(
        "INSERT OR IGNORE INTO metadata (key, value) VALUES (?, ?)",
        ("last_prune", _microseconds_since_epoch()),
    )

    conn.commit()


def _initialize_cache_db(conn: sqlite3.Connection, cache_expiration_days: int) -> None:
    """Prepare a freshly opened cache connection: enable WAL, ensure the schema,
    and prune expired entries.

    Raises sqlite3.DatabaseError if the file is not a usable database, letting
    the caller recreate it. We deliberately do NOT run ``PRAGMA integrity_check``
    here: on a multi-GB cache that full-file scan costs ~1s per worker and, run
    concurrently by every pytest-xdist worker at startup, serializes parallel
    runs. Corruption instead surfaces lazily as a DatabaseError on first access.
    """
    # WAL mode lets readers and a single writer coexist without blocking each
    # other, preventing SQLITE_BUSY deadlocks when many xdist workers hit the
    # cache at once. Once set, WAL persists in the DB file so all future
    # connections use it automatically. On a non-database file this is the first
    # access that touches the header, so it raises DatabaseError for the caller.
    conn.execute("PRAGMA journal_mode=WAL")

    _check_database_structure(conn)

    # Prune entries not hit recently. The DELETE is index-assisted (see
    # idx_models_last_hit), so on a warm cache with nothing to expire it stays
    # cheap even though it runs on every worker's first parse.
    cursor = conn.cursor()
    cursor.execute("BEGIN IMMEDIATE;")
    cutoff_time = _microseconds_since_epoch(timedelta(days=-cache_expiration_days))
    cursor.execute("DELETE FROM models WHERE last_hit < ?", (cutoff_time,))
    # Sometimes Windows time resolution is a bit coarse, so we make
    # sure that if we update the last_prune time, it is actually newer
    # than the previous one.
    cursor.execute(
        "UPDATE metadata SET value = max(value + 1, ?) WHERE key = ?",
        (_microseconds_since_epoch(), "last_prune"),
    )
    conn.commit()


# Tracks which database files have already been initialised in this process.
_initialized_dbs: set[Path] = set()


def _calculate_txt_hash(txt: str):
    hasher = hashlib.sha256()
    hasher.update(txt.encode("utf-8"))
    return hasher.hexdigest()


def _get_default_cache_path() -> Path:
    """
    Heavily inspired by https://github.com/davidhalter/parso/blob/master/parso/cache.py

    The path where the cache is stored.
    On Linux, this defaults to ``~/.cache/pymoca/``, on OS X to
    ``~/Library/Caches/Pymoca/`` and on Windows to ``%LOCALAPPDATA%\\Pymoca\\Pymoca\\``.
    On Linux, if environment variable ``$XDG_CACHE_HOME`` is set,
    ``$XDG_CACHE_HOME/pymoca`` is used instead of the default one.
    """
    if platform.system().lower() == "windows":
        dir_ = Path(os.getenv("LOCALAPPDATA") or "~") / "Pymoca" / "Pymoca"
    elif platform.system().lower() == "darwin":
        dir_ = Path("~") / "Library" / "Caches" / "Pymoca"
    else:
        dir_ = Path(os.getenv("XDG_CACHE_HOME") or "~/.cache") / "pymoca"

    return dir_.expanduser()


def parse(
    txt: str,
    /,
    trace: bool = False,
    model_cache_folder: Path | None = None,
    cache_db: str = DEFAULT_MODEL_CACHE_DB,
    cache_expiration_days: int = 30,
    always_update_last_hit: bool = False,
    bypass_cache: bool = False,
) -> ast.Tree:
    """
    Parse the Modelica code given in text and return the Abstract Syntax Tree (AST).

    This function uses a cache to avoid re-parsing the same text multiple times. The
    cache is stored in a SQLite database file. The cache entries are pruned based on
    their last access time. If an entry has not been accessed for a certain number of
    days (defined by cache_expiration_days), it is removed from the cache.

    Args:
        txt (str): The Modelica code to parse.
        trace (bool, optional): If True, the parser will print debug information. Default
            is False
        model_cache_folder (Path, optional): The folder where the cache database is
            stored. If not provided, a default location based on the operating system is
            used.
        cache_db (str, optional): The name of the cache database file. Default is
            DEFAULT_MODEL_CACHE_DB.
        cache_expiration_days (int, optional): The number of days after which a cache
            entry is considered expired and is pruned. Defaultis 30.
        always_update_last_hit (bool, optional): If True, the last access time of a
            cache entry is always updated when it is accessed. If False, the last access
            time is only updated if it is more than a day old. Default is False.
        bypass_cache (bool, optional): If True, the cache is bypassed and the parsing is
            performed directly. Default is False.

    Returns:
        ast.Tree: The AST of the parsed Modelica code

    Raises:
        ModelicaSyntaxError: If there is a syntax error in the Modelica code.
        FileNotFoundError: If the specified cache folder does not exist.
        OSError: If there is an error creating the cache folder.

    """
    modelica_file = parse_text(
        txt,
        model_cache_folder=model_cache_folder,
        cache_db=cache_db,
        cache_expiration_days=cache_expiration_days,
        always_update_last_hit=always_update_last_hit,
        bypass_cache=bypass_cache,
    )
    return file_to_tree(modelica_file)


def parse_text(
    txt: str,
    /,
    trace: bool = False,
    model_cache_folder: Path | None = None,
    cache_db: str = DEFAULT_MODEL_CACHE_DB,
    cache_expiration_days: int = 30,
    always_update_last_hit: bool = False,
    bypass_cache: bool = False,
) -> ModelicaFile:
    """Parse the Modelica code given in text and return the parsed ModelicaFile."""
    if bypass_cache:
        return _parse_text(txt, trace=trace)

    pymoca_version = pymoca.__version__

    # Do not use caching if we have a dirty work tree, as the source
    # code can't be uniquely identified.
    if pymoca_version.endswith(".dirty"):
        logger.debug("Bypassing cache because working directory is dirty")
        return _parse_text(txt, trace=trace)

    if model_cache_folder is not None:
        db_folder = model_cache_folder
    else:
        db_folder = _get_default_cache_path()
    db_folder.mkdir(parents=True, exist_ok=True)

    full_db_path = db_folder / cache_db
    conn = sqlite3.connect(full_db_path, isolation_level=None, timeout=30)

    cursor = conn.cursor()

    if full_db_path not in _initialized_dbs:
        try:
            _initialize_cache_db(conn, cache_expiration_days)
        except sqlite3.OperationalError:
            # Transient contention, not corruption: SQLITE_BUSY/SQLITE_LOCKED
            # map to OperationalError, and opening a WAL database uses a fixed
            # internal retry loop (not the busy timeout), so parallel workers
            # racing at startup can hit "database is locked". Skip the cache
            # for this parse; initialization is retried on the next one.
            logger.debug("Model cache database is locked, parsing without cache")
            conn.close()
            return _parse_text(txt, trace=trace)
        except sqlite3.DatabaseError:
            # A corrupt or truncated cache file surfaces as a DatabaseError on
            # first access. Recreating it is far cheaper than the full-file
            # PRAGMA integrity_check we used to run on every worker startup,
            # which scans the entire (multi-GB) cache and serializes parallel
            # test runs.
            logger.warning("Model cache database is corrupt, recreating...")
            conn.close()
            try:
                for suffix in ("", "-wal", "-shm"):
                    try:
                        os.remove(f"{full_db_path}{suffix}")
                    except FileNotFoundError:
                        pass
                conn = sqlite3.connect(full_db_path, isolation_level=None, timeout=30)
                _initialize_cache_db(conn, cache_expiration_days)
            except OSError:
                # On Windows the files cannot be removed while another process
                # has them open. Skip the cache for this parse rather than
                # break every concurrent worker by deleting files under them.
                logger.debug("Cannot recreate model cache database, parsing without cache")
                return _parse_text(txt, trace=trace)
            except sqlite3.DatabaseError:
                # Another process may be recreating the database concurrently.
                logger.debug("Model cache database is unusable, parsing without cache")
                conn.close()
                return _parse_text(txt, trace=trace)

        cursor = conn.cursor()
        _initialized_dbs.add(full_db_path)

    # Check if the txt exists in the database
    txt_hash = _calculate_txt_hash(txt)

    # The cache-hit lookup is read-only and is by far the hottest path. In WAL
    # mode reads never block other readers or the single writer, so we issue a
    # plain autocommit SELECT instead of taking the exclusive write lock that
    # BEGIN IMMEDIATE acquires. Wrapping this read in BEGIN IMMEDIATE would
    # serialize every parallel xdist worker on each cache hit, making parallel
    # runs slower than serial ones.
    cursor.execute(
        "SELECT last_hit, data FROM models WHERE txt_hash=? AND pymoca_version=?",
        (txt_hash, pymoca_version),
    )
    result = cursor.fetchone()

    file = None

    if result:
        logger.debug(f"Model with hash '{txt_hash}' ({pymoca_version}) found in cache")
        last_hit, pickled_data = result

        yesterday = _microseconds_since_epoch(timedelta(days=-1))

        if always_update_last_hit or last_hit < yesterday:
            cursor.execute("BEGIN IMMEDIATE;")
            # Sometimes Windows time resolution is a bit coarse, so we make
            # sure that if we update the last_hit time, it is actually newer
            # than the previous one.
            cursor.execute(
                "UPDATE models SET last_hit = max(last_hit + 1, ?) WHERE txt_hash = ? "
                "AND pymoca_version = ?",
                (_microseconds_since_epoch(), txt_hash, pymoca_version),
            )
            conn.commit()
        try:
            file = pickle.loads(pickled_data)
        except pickle.UnpicklingError:
            logger.warning(f"Model with hash '{txt_hash}' ({pymoca_version}) failed to unpickle")
    else:
        logger.debug(f"Model with hash '{txt_hash}' ({pymoca_version}) not in cache")

    if file is None:
        # We get here if we didn't find anything in the cache, or if the
        # unpickling of the cache failed
        try:
            file = _parse_text(txt, trace=trace)
        except Exception:
            conn.close()
            raise

        pickled_data = pickle.dumps(file)

        # Note that we do an 'INSERT OR REPLACE' because concurrent access
        # might mean two processes/threads try to insert an entry
        cursor.execute("BEGIN IMMEDIATE;")
        cursor.execute(
            "INSERT OR REPLACE INTO models (txt_hash, pymoca_version, data, last_hit) VALUES (?, ?, ?, ?)",
            (txt_hash, pymoca_version, pickled_data, _microseconds_since_epoch()),
        )
        conn.commit()

    conn.close()

    return file


def parse_file(file_path: str | Path, trace: bool = False) -> ast.Tree:
    """Parse given file path and return parsed ast.Tree

    Args:
        file_path (str or Path): Path to the Modelica file to parse.
        trace (bool, optional): If True, print a trace of the parse for debugging purposes.
    Returns:
        ast.Tree: The Abstract Syntax Tree (AST) of the parsed Modelica code.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ModelicaSyntaxError: If there is a syntax error in the Modelica code.
        See also: parse()

    If you need control of the parse cache, use parse() instead.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    txt = path.read_text(encoding="utf-8")
    try:
        ast_tree = parse(txt, trace=trace)
    except ModelicaSyntaxError as exception:
        # Add filename to the exception
        info = (str(path),) + exception.args[1][1:]
        raise ModelicaSyntaxError(
            exception.msg,
            info,
        ) from exception
    return ast_tree
