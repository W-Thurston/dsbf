# tests/test_storage/test_writer_paths.py
"""Paths recorded in the database are absolute, so any directory can reopen them."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from dsbf.storage.schema import get_connection
from dsbf.storage.writer import persist_run

pytestmark = pytest.mark.unit


def _engine(output_dir: Path, **metadata) -> SimpleNamespace:
    return SimpleNamespace(
        context=None,
        results={},
        output_dir=str(output_dir),
        config={"metadata": {"dataset_name": "toy", **metadata}},
    )


def _stored_source_path(db_path: Path) -> str | None:
    with get_connection(db_path) as conn:
        return conn.execute("SELECT source_path FROM datasets").fetchone()[0]


def test_relative_dataset_path_is_stored_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "runs.db"

    persist_run(
        _engine(tmp_path / "20260101_000000", dataset_path="data/toy.csv"),
        db_path=db_path,
    )

    assert _stored_source_path(db_path) == str(tmp_path / "data" / "toy.csv")


def test_builtin_dataset_stores_no_path(tmp_path):
    db_path = tmp_path / "runs.db"

    persist_run(
        _engine(tmp_path / "20260101_000000", dataset_source="seaborn"),
        db_path=db_path,
    )

    assert _stored_source_path(db_path) is None
