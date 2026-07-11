# Pymoca Development Overview

Pymoca: Modelica-to-CAS translator (Python). Parses `.mo` → AST → flattens class hierarchies → generates CasADi/SymPy/XML output.

## Commands

```bash
pip install -e ".[all,test]"      # dev editable install
python antlr/antlr_build.py       # regenerate parser after editing Modelica.g4
pre-commit run --all-files        # run pre-commit hook on all (automatic on staged before commit)
pytest test -n auto               # all tests (parallel)
pytest test/parse_test.py -k X -n auto # single file (parallel)
pytest test/parse_test.py -k X    # single test (serial, for debugging)
pytest test/msl_examples_test.py  # run MSL examples pipeline tests (excluded from normal collection)
python test/msl_examples_test.py  # MSL pipeline CLI (pass -h for options)
tox -e py                         # via tox
tox -e coverage                   # with coverage
```

## Code Style

- Python 3.10+, black (100 char), flake8 (bugbear, comprehensions, import-order), pyright
- **Comments concise** — Only comment if you can't determine local context from code within surrounding ~80 lines. One line comments is goal, not hard limit. **Don't reference context outside the function, class, module, etc.**
- **Docstrings concise** — {} denotes optional: """One line summary{\n\n¶ to clarify}{\n\nReturns: if not None}{\n\nRaises:}""". **Optional parts only if surrounding context is > ~80 lines.**
- Generated code in `src/pymoca/generated/` and `test/generated/` — do not edit

## Git Style

- **Atomic commits**:
  - Each commit covers one concern
  - One Modelica spec concept = one commit, even if it touches ast.py, parser.py, tree/, doc/ and tests together.
  - **Commits with logic changes should include a test** and **tests should pass at every commit**.
- **Commit message**:
  - Subject: like "parser: Add global name syntax" or "tree: Fix connect equation" — **50 chars max goal**. Describes the *what*, not the *how* (omit implementation details) and **covers the full scope of changes**. Use "Fix" for bug fixes, not the mechanism ("Propagate", "Copy", etc.). Prefixes: `parser:`, `tree:`, `casadi:`, `sympy:`, `xml:` represent pipeline stages - for diffs touching multiple pipeline stages, rightmost wins. Use `cli:` for command line interface changes. Only use `test:`, `doc:`, `ci/cd:`, etc, for infrastructure-only changes.
  - Body:
    - ¶1 is 1–3 sentences, high-level and durable. Open with an imperative sentence stating what was changed, using spec/domain terms (not code locations or step numbers), with the spec reference at the end if applicable. Follow with *concise* statement of the problem fixed ("Previously these were lost.") or meaning of a new feature. Do not explain the mechanism in ¶1. The mechanism belongs in ¶2.
    - ¶2: non-obvious design context summary including mechanism — what ties the changes together, alternatives considered, tradeoffs. Don't reiterate the code diff.
    - Additional paragraphs and/or list: If necessary to describe anything important to understanding the diffs, but not covered above. Keep paragraphs to one topic. **If not related to ¶1 or ¶2, it should be a separate commit.**
    - Subject and body use the present imperative form, such as "Add," "Fix," or "Remove", especially on initial sentences in paragraphs and list items.
    - **No meta-commentary** — don't mention code review, how an issue was found, or process artifacts in commit messages.
- **Linear history**
  - This project uses a rebase workflow.
  - Force-pushing a feature branch after a rebase or commit reword is expected and correct.

## References

The ground truth for all language syntax and semantics decisions, in priority order:

- [**Modelica Language Specification v3.5**](https://specification.modelica.org/maint/3.5/MLS.html)
- [**Modelica Compliance Library**](https://github.com/modelica/Modelica-Compliance)
- [**MLS Repo**](https://github.com/modelica/ModelicaSpecification) may have additional
  clarification from the v3.5 spec development, version diffs, and later versions

