"""Regenerate all cell outputs in doc/pymoca_011_vs_new.ipynb.

The notebook compares new pymoca against the 0.11.2 release.  It requires two
side-by-side environments in a *clean* clone so that tracebacks contain tidy
``/tmp/pymoca/...`` paths rather than developer-specific worktree paths:

* ``.venv-pymoca-head`` — editable install of the cloned branch (new pymoca),
  used as the Jupyter kernel.
* ``.venv-pymoca-011`` — ``pip install "pymoca[casadi]==0.11.2"``, subprocessed
  by the notebook's ``run_old()`` helper with ``PYTHONPATH`` stripped.

Usage examples::

    # First run: set up /tmp/pymoca from scratch, execute, write back.
    doc/pymoca_011_vs_new_run_all.py

    # Force a fresh clone and venvs (e.g. after a large rebase).
    doc/pymoca_011_vs_new_run_all.py --fresh

    # Use a different clone directory.
    doc/pymoca_011_vs_new_run_all.py --clone-dir ~/tmp/pymoca-nb

    # Set up the environments but skip notebook execution.
    doc/pymoca_011_vs_new_run_all.py --no-execute

Run this script with any supported Python (e.g. the head venv's interpreter or the
system python3).  It manages its own subprocesses; the driver Python is *not*
used inside the clone.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV_NO_PYTHONPATH = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def _run(cmd: list, *, cwd=None, env=None, check=True, capture=False):
    """Echo and execute *cmd*, raising on non-zero exit by default."""
    print("$", " ".join(str(c) for c in cmd), flush=True)
    kwargs = {"cwd": cwd, "env": env or _ENV_NO_PYTHONPATH, "check": check}
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    return subprocess.run(cmd, **kwargs)


def _run_output(cmd: list, *, cwd=None) -> str:
    """Return stripped stdout of *cmd* (no echo)."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=_ENV_NO_PYTHONPATH,
    )
    return result.stdout.strip()


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _venv_pip(venv_dir: Path) -> list:
    return [str(_venv_python(venv_dir)), "-m", "pip", "install", "--quiet"]


# ---------------------------------------------------------------------------
# Step 1 — clone / update
# ---------------------------------------------------------------------------


def _detect_source() -> Path:
    """Return the main repo root (not this worktree root, if we're in one)."""
    script_dir = Path(__file__).resolve().parent.parent  # repo root from doc/
    try:
        # --git-common-dir points at the shared .git; its parent is the main repo.
        common = _run_output(["git", "rev-parse", "--git-common-dir"], cwd=script_dir)
        return Path(common).resolve().parent
    except subprocess.CalledProcessError:
        return script_dir


def _detect_branch() -> str:
    script_dir = Path(__file__).resolve().parent.parent
    try:
        return _run_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=script_dir)
    except subprocess.CalledProcessError:
        return "master"


def _detect_head_sha() -> str:
    script_dir = Path(__file__).resolve().parent.parent
    return _run_output(["git", "rev-parse", "HEAD"], cwd=script_dir)


def _is_valid_git_repo(clone_dir: Path) -> bool:
    """Return whether *clone_dir* is a usable git checkout.

    /tmp is subject to periodic OS cleanup that can delete files (e.g.
    ``.git/HEAD``, ``.git/config``) older than a few days while leaving
    empty directories behind, producing a "not a git repository" error on
    the next fetch.
    """
    try:
        subprocess.run(
            ["git", "-C", str(clone_dir), "rev-parse", "--git-dir"],
            check=True,
            capture_output=True,
            env=_ENV_NO_PYTHONPATH,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_clone(clone_dir: Path, source: Path, fresh: bool) -> None:
    if fresh and clone_dir.exists():
        print(f"[clone] Removing {clone_dir} (--fresh)")
        shutil.rmtree(clone_dir)

    if clone_dir.exists() and not _is_valid_git_repo(clone_dir):
        print(f"[clone] {clone_dir} is not a valid git repository (stale /tmp?); removing")
        shutil.rmtree(clone_dir)

    head_sha = _detect_head_sha()

    if not clone_dir.exists():
        print(f"[clone] Cloning {source} → {clone_dir}")
        _run(["git", "clone", str(source), str(clone_dir)])
    else:
        print(f"[clone] Fetching updates from {source}")

    # Fetch and check out head_sha directly rather than a branch name: in a
    # worktree with a detached HEAD, the branch is the literal string "HEAD",
    # which checks out a no-op ref rather than the commit we actually want.
    # Reset --hard also discards any local modifications left behind by a
    # previous run (e.g. execute_notebook's in-place notebook execution).
    _run(["git", "-C", str(clone_dir), "fetch", str(source), head_sha])
    _run(["git", "-C", str(clone_dir), "checkout", "--detach", "FETCH_HEAD"])
    _run(["git", "-C", str(clone_dir), "reset", "--hard", "FETCH_HEAD"])

    # Only init the MSL submodule that Part 2 of the notebook needs.
    print("[clone] Initialising test/libraries/MSL-4.0.x submodule")
    _run(
        [
            "git",
            "-C",
            str(clone_dir),
            "submodule",
            "update",
            "--init",
            "test/libraries/MSL-4.0.x",
        ]
    )


# ---------------------------------------------------------------------------
# Step 2 — head venv (editable install + Jupyter tools)
# ---------------------------------------------------------------------------


def _venv_has_pkg(venv_dir: Path, pkg: str) -> bool:
    """Return True if *pkg* is importable in the venv (no echo)."""
    result = subprocess.run(
        [str(_venv_python(venv_dir)), "-c", f"import {pkg}"],
        env=_ENV_NO_PYTHONPATH,
        capture_output=True,
    )
    return result.returncode == 0


def ensure_venv_head(clone_dir: Path, python: str, fresh: bool) -> None:
    venv_dir = clone_dir / ".venv-pymoca-head"
    if fresh and venv_dir.exists():
        print(f"[venv-head] Removing {venv_dir} (--fresh)")
        shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        print(f"[venv-head] Creating {venv_dir}")
        _run([python, "-m", "venv", str(venv_dir)])

    # Always re-run pip so version bumps are picked up on non-fresh reuse.
    if not _venv_has_pkg(venv_dir, "pymoca") or fresh:
        _run(_venv_pip(venv_dir) + ["-e", ".[all]"], cwd=clone_dir)

    if not _venv_has_pkg(venv_dir, "nbconvert") or fresh:
        _run(_venv_pip(venv_dir) + ["nbconvert", "nbclient", "ipykernel"])

    # Register a 'python3' kernelspec local to this venv so nbconvert's kernel
    # lookup resolves to the head venv interpreter, not a global python3.
    _run(
        [
            str(_venv_python(venv_dir)),
            "-m",
            "ipykernel",
            "install",
            "--sys-prefix",
            "--name",
            "python3",
            "--display-name",
            "Python 3 (pymoca-head)",
        ]
    )
    print(
        "[venv-head] Ready:",
        _run_output(
            [
                str(_venv_python(venv_dir)),
                "-c",
                "import pymoca; print('pymoca', pymoca.__version__)",
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Step 3 — 0.11.2 venv
# ---------------------------------------------------------------------------


def ensure_venv_011(clone_dir: Path, python: str, fresh: bool) -> None:
    venv_dir = clone_dir / ".venv-pymoca-011"
    if fresh and venv_dir.exists():
        print(f"[venv-011] Removing {venv_dir} (--fresh)")
        shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        print(f"[venv-011] Creating {venv_dir}")
        _run([python, "-m", "venv", str(venv_dir)])

    # Check whether 0.11.2 (exactly) is installed.
    need_install = True
    if _venv_has_pkg(venv_dir, "pymoca"):
        ver = _run_output(
            [str(_venv_python(venv_dir)), "-c", "import pymoca; print(pymoca.__version__)"]
        )
        need_install = ver != "0.11.2"

    if need_install or fresh:
        _run(_venv_pip(venv_dir) + ["pymoca[casadi]==0.11.2"])
    print(
        "[venv-011] Ready:",
        _run_output(
            [
                str(_venv_python(venv_dir)),
                "-c",
                "import pymoca; print('pymoca', pymoca.__version__)",
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Step 4 — execute notebook and write back
# ---------------------------------------------------------------------------

_NOTEBOOK = "doc/pymoca_011_vs_new.ipynb"


def execute_notebook(worktree_root: Path, clone_dir: Path) -> None:
    src = worktree_root / _NOTEBOOK
    dst = clone_dir / _NOTEBOOK
    print(f"[notebook] Copying {src} → {dst}")
    shutil.copy2(src, dst)

    venv_dir = clone_dir / ".venv-pymoca-head"
    # Pin JUPYTER_DATA_DIR to the venv so kernel lookup finds only our registered
    # 'python3' kernelspec and not any user- or system-level python3 kernel.
    # Use 'python -m nbconvert' (not 'python -m jupyter nbconvert') to avoid PATH
    # dispatch which can pick up a jupyter-nbconvert from a different environment.
    nb_env = {**_ENV_NO_PYTHONPATH, "JUPYTER_DATA_DIR": str(venv_dir / "share" / "jupyter")}
    _run(
        [
            str(_venv_python(venv_dir)),
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.kernel_name=python3",
            "--ExecutePreprocessor.timeout=600",
            _NOTEBOOK,
        ],
        cwd=clone_dir,
        env=nb_env,
    )

    print(f"[notebook] Writing back → {src}")
    shutil.copy2(dst, src)
    print(f"[notebook] Done.  Outputs written to {src}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--clone-dir",
        type=Path,
        default=Path("/tmp/pymoca"),
        metavar="DIR",
        help="Path for the clean clone (default: /tmp/pymoca)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        metavar="REPO",
        help="Source repo to clone from (default: auto-detected local main repo)",
    )
    parser.add_argument(
        "--branch",
        default=None,
        metavar="BRANCH",
        help="Branch to check out in the clone (default: current branch)",
    )
    parser.add_argument(
        "--python",
        default=None,
        metavar="PYTHON",
        help="Python interpreter for creating venvs (default: python3.14, then sys.executable)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Remove and recreate the clone and both venvs from scratch",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Set up the clone and venvs but skip notebook execution",
    )
    return parser.parse_args()


def _resolve_python(requested) -> str:
    if requested:
        return requested
    for candidate in ("python3.14", "python3"):
        if shutil.which(candidate):
            return candidate
    return sys.executable


def main() -> None:
    args = parse_args()

    source = args.source or _detect_source()
    branch = args.branch or _detect_branch()
    python = _resolve_python(args.python)
    worktree_root = Path(__file__).resolve().parent.parent

    print(f"[config] source   : {source}")
    print(f"[config] branch   : {branch}")
    print(f"[config] clone_dir: {args.clone_dir}")
    print(f"[config] python   : {python}")
    print(f"[config] fresh    : {args.fresh}")
    print()

    ensure_clone(args.clone_dir, source, args.fresh)
    print()
    ensure_venv_head(args.clone_dir, python, args.fresh)
    print()
    ensure_venv_011(args.clone_dir, python, args.fresh)

    if args.no_execute:
        print("\n[done] --no-execute: skipping notebook execution.")
        return

    print()
    execute_notebook(worktree_root, args.clone_dir)


if __name__ == "__main__":
    main()
