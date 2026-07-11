#!/usr/bin/env python
"""
Tests for the parser's SQLite caching layer.
"""

import contextlib
import enum
import logging
import sqlite3
import tempfile
from pathlib import Path

import pymoca
from pymoca import parser
from pymoca.parser import DEFAULT_MODEL_CACHE_DB

import pytest


class WorkDirState(enum.Enum):
    CLEAN = "clean"
    DIRTY = "dirty"


@contextlib.contextmanager
def modify_version(version_type: WorkDirState):
    pymoca_version = pymoca.__version__
    if pymoca_version.endswith(".dirty"):
        clean_version = pymoca_version[:-6]
    else:
        clean_version = pymoca_version

    dirty_version = clean_version + ".dirty"

    if version_type == WorkDirState.CLEAN:
        pymoca.__version__ = clean_version
    elif version_type == WorkDirState.DIRTY:
        pymoca.__version__ = dirty_version
    else:
        raise ValueError("Unknown version type")

    try:
        yield
    finally:
        pymoca.__version__ = pymoca_version


def test_parse_cache_hit(caplog):
    """Test caching of models"""

    txt = """
        model A
          parameter Real x, y;
        equation
          der(y) = x;
        end A;
    """

    with modify_version(WorkDirState.CLEAN), tempfile.TemporaryDirectory() as tmpdirname:
        full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB

        # The cache database should not exist yet
        assert not full_db_path.exists()

        _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

        # And now the database _should exist_, and we check its contents
        # where we expect to find a single cached entry
        assert full_db_path.exists()

        conn = sqlite3.connect(full_db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM metadata WHERE key='created_at'")
        first_created_at = int(cursor.fetchone()[0])

        cursor.execute("SELECT last_hit FROM models")
        first_hit_time = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM models")
        assert cursor.fetchone()[0] == 1

        # Check that we get log messages saying the cache entry was found
        # We also force an update to the cache hit time
        with caplog.at_level(logging.DEBUG, logger="pymoca"):
            _ = parser.parse(txt, model_cache_folder=Path(tmpdirname), always_update_last_hit=True)
            assert any(") found in cache" in record.message for record in caplog.records)

        cursor.execute("SELECT value FROM metadata WHERE key='created_at'")
        second_created_at = int(cursor.fetchone()[0])

        cursor.execute("SELECT last_hit FROM models")
        second_hit_time = cursor.fetchone()[0]

        # Check that the created_at time was _not_ updated, i.e. the
        # database was not recreated for some reason.
        assert first_created_at == second_created_at

        # Check that, if we parse it _again_, the `last_hit` updates
        assert second_hit_time > first_hit_time

        # Check that there's still only one model in the cache
        cursor.execute("SELECT COUNT(*) FROM models")
        assert cursor.fetchone()[0] == 1

        cursor.close()
        conn.close()


def test_parse_cache_purge():
    """Test that models that have not been hit in N days are purged"""

    model_a = """
        model A
          parameter Real x, y;
        equation
          der(y) = x;
        end A;
    """

    model_b = """
        model B
          parameter Real x, y;
        equation
          der(y) = x;
        end B;
    """

    with modify_version(WorkDirState.CLEAN), tempfile.TemporaryDirectory() as tmpdirname:
        full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB

        # Parse the models to add them to the cache
        for txt in [model_a, model_b]:
            _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

        conn = sqlite3.connect(full_db_path)
        cursor = conn.cursor()

        # Check that the models are in the cache
        cursor.execute("SELECT COUNT(*) FROM models")
        assert cursor.fetchone()[0] == 2

        cursor.execute("SELECT value FROM metadata WHERE key='last_prune'")
        first_prune_time = int(cursor.fetchone()[0])

        cursor.execute("SELECT value FROM metadata WHERE key='created_at'")
        first_created_at = int(cursor.fetchone()[0])

        # Reimport the module to force a cache purge check, but with an
        # expiration time such that the models should not be purged
        import importlib

        importlib.reload(parser)

        _ = parser.parse(
            model_b,
            model_cache_folder=Path(tmpdirname),
            cache_expiration_days=1,
        )

        cursor.execute("SELECT COUNT(*) FROM models")
        assert cursor.fetchone()[0] == 2

        # Reimport the module again, but now we force a purge by setting
        # expiration to zero
        importlib.reload(parser)

        _ = parser.parse(
            model_b,
            model_cache_folder=Path(tmpdirname),
            cache_expiration_days=0,
        )

        cursor.execute("SELECT value FROM metadata WHERE key='last_prune'")
        second_prune_time = int(cursor.fetchone()[0])

        cursor.execute("SELECT value FROM metadata WHERE key='created_at'")
        second_created_at = int(cursor.fetchone()[0])

        # Check that the other model has been purged from the cache.
        cursor.execute("SELECT COUNT(*) FROM models")
        assert cursor.fetchone()[0] == 1

        # Check that the last prune time was updated
        assert second_prune_time > first_prune_time

        # And that the creation time was not
        assert first_created_at == second_created_at

        cursor.close()
        conn.close()


def test_dirty_no_caching():
    """Test cache and cache creation bypass if working directory is dirty"""

    txt = """
        model A
          parameter Real x, y;
        equation
          der(y) = x;
        end A;
    """

    with modify_version(WorkDirState.DIRTY), tempfile.TemporaryDirectory() as tmpdirname:
        full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB

        # The cache database should not exist yet
        assert not full_db_path.exists()

        _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

        # And now the database should not exist
        assert not full_db_path.exists()


def test_unpickling_error(caplog):
    """Test that we can handle unpickling errors, and then recreate the cache entry"""

    txt = """
        model A
          parameter Real x, y;
        equation
          der(y) = x;
        end A;
    """

    with modify_version(WorkDirState.CLEAN), tempfile.TemporaryDirectory() as tmpdirname:
        full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB

        # The cache database should not exist yet
        assert not full_db_path.exists()

        _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

        assert full_db_path.exists()

        # Modify the single entry in the cache to make it unpickleable
        conn = sqlite3.connect(full_db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT txt_hash FROM models")
        txt_hash = cursor.fetchone()[0]
        cursor.execute("UPDATE models SET data = ? WHERE txt_hash = ?", (b"not a pickle", txt_hash))
        conn.commit()

        cursor.close()
        conn.close()

        # Check that we get log messages saying the cache entry is corrupt
        with caplog.at_level(logging.WARNING, logger="pymoca"):
            _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

            assert any("failed to unpickle" in record.message for record in caplog.records)
            n_warnings = len([r for r in caplog.records if r.levelno >= logging.WARNING])

            # Check that we get no additional warnings when unpickling again
            _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))
            assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == n_warnings


def test_incorrect_table_layout(caplog):
    """Test that a corrupt cache file is ignored"""

    txt = """
        model A
          parameter Real x, y;
        equation
          der(y) = x;
        end A;
    """

    with modify_version(WorkDirState.CLEAN), tempfile.TemporaryDirectory() as tmpdirname:
        full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB

        # The cache database should not exist yet
        assert not full_db_path.exists()

        # Create a database with incorrectly structured tables
        conn = sqlite3.connect(full_db_path)
        cursor = conn.cursor()

        dummy_table_str = """
            CREATE TABLE {} (
                wrong_key TEXT,
                wrong_value TEXT,
                PRIMARY KEY (wrong_key)
            )
        """

        cursor.execute(dummy_table_str.format("models"))
        cursor.execute(dummy_table_str.format("metadata"))

        conn.close()

        # And now the database should exist
        assert full_db_path.exists()

        # Check that we get log messages saying the layout is incorrect
        with caplog.at_level(logging.WARNING, logger="pymoca"):
            _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

            warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
            assert any("Model text cache table layout didn't match" in m for m in warning_messages)
            assert any("Metadata table layout didn't match" in m for m in warning_messages)


def test_corrupt_cache_file(caplog):
    """Test that a corrupt cache file is ignored"""

    txt = """
        model A
          parameter Real x, y;
        equation
          der(y) = x;
        end A;
    """

    with modify_version(WorkDirState.CLEAN), tempfile.TemporaryDirectory() as tmpdirname:
        full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB

        # The cache database should not exist yet
        assert not full_db_path.exists()

        # Create a corrupt cache file
        with open(full_db_path, "w") as f:
            f.write("This is not a valid SQLite database file")

        with caplog.at_level(logging.WARNING, logger="pymoca"):
            _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))

            assert any(
                "Model cache database is corrupt" in record.message for record in caplog.records
            )


def test_locked_cache_db(caplog, monkeypatch):
    """Test that lock contention during cache setup falls back to uncached parsing"""

    txt = """
        model A
          parameter Real x, y;
        equation
          der(y) = x;
        end A;
    """

    with modify_version(WorkDirState.CLEAN), tempfile.TemporaryDirectory() as tmpdirname:
        full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB

        def raise_locked(conn, cache_expiration_days):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(parser, "_initialize_cache_db", raise_locked)

        with caplog.at_level(logging.DEBUG, logger="pymoca"):
            tree = parser.parse(txt, model_cache_folder=Path(tmpdirname))

        assert tree.classes["A"] is not None
        # A locked database must not be mistaken for a corrupt one and deleted
        assert not any("corrupt" in r.message for r in caplog.records)
        assert any("locked" in r.message for r in caplog.records)
        assert full_db_path.exists()

        # Initialization is retried and succeeds once the lock clears
        monkeypatch.undo()
        _ = parser.parse(txt, model_cache_folder=Path(tmpdirname))
        conn = sqlite3.connect(full_db_path)
        assert conn.execute("SELECT COUNT(*) FROM models").fetchone()[0] == 1
        conn.close()


def test_corrupt_cache_file_undeletable(caplog, monkeypatch):
    """Test that a corrupt cache file that cannot be removed falls back to
    uncached parsing (on Windows, files opened by other processes cannot be
    deleted)"""

    txt = """
        model A
          parameter Real x, y;
        equation
          der(y) = x;
        end A;
    """

    with modify_version(WorkDirState.CLEAN), tempfile.TemporaryDirectory() as tmpdirname:
        full_db_path = Path(tmpdirname) / DEFAULT_MODEL_CACHE_DB
        corrupt_content = "This is not a valid SQLite database file"
        with open(full_db_path, "w") as f:
            f.write(corrupt_content)

        def raise_permission_error(path):
            raise PermissionError(f"The process cannot access the file: {path}")

        monkeypatch.setattr(parser.os, "remove", raise_permission_error)

        with caplog.at_level(logging.DEBUG, logger="pymoca"):
            tree = parser.parse(txt, model_cache_folder=Path(tmpdirname))

        assert tree.classes["A"] is not None
        assert any("Model cache database is corrupt" in r.message for r in caplog.records)
        assert any("parsing without cache" in r.message for r in caplog.records)
        assert full_db_path.read_text() == corrupt_content


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__])
