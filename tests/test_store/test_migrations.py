"""The database is built by the migration scripts, and a newer one is refused."""

import sqlite3

import pytest
import sqlalchemy as sa
from alembic.config import Config
from alembic.script import ScriptDirectory

from dsbf.store import SqliteRunRepository, StoreError, open_repository
from dsbf.store import sqlite as store_sqlite
from dsbf.store.tables import metadata

pytestmark = pytest.mark.unit


def head() -> str | None:
    config = Config()
    config.set_main_option("script_location", str(store_sqlite._MIGRATIONS))
    return ScriptDirectory.from_config(config).get_current_head()


def test_an_empty_database_is_upgraded_to_head_and_has_every_table(tmp_path):
    repo = SqliteRunRepository(tmp_path / "dsbf.db", tmp_path / "artifacts")
    repo.close()

    connection = sqlite3.connect(tmp_path / "dsbf.db")
    tables = {r[0] for r in connection.execute("select name from sqlite_master")}
    version = connection.execute("select version_num from alembic_version").fetchone()
    mode = connection.execute("pragma journal_mode").fetchone()
    connection.close()

    assert set(metadata.tables) <= tables
    assert version == (head(),)
    assert mode == ("wal",)


def test_the_scripts_build_the_columns_the_tables_declare(tmp_path):
    repo = SqliteRunRepository(tmp_path / "dsbf.db", tmp_path / "artifacts")
    inspector = sa.inspect(repo._engine)  # noqa: SLF001
    try:
        for name, table in metadata.tables.items():
            built = {c["name"] for c in inspector.get_columns(name)}
            assert built == {c.name for c in table.columns}, name
    finally:
        repo.close()


def test_reopening_changes_nothing(tmp_path):
    SqliteRunRepository(tmp_path / "dsbf.db", tmp_path / "artifacts").close()
    SqliteRunRepository(tmp_path / "dsbf.db", tmp_path / "artifacts").close()


def test_a_database_from_a_newer_revision_is_refused(tmp_path):
    SqliteRunRepository(tmp_path / "dsbf.db", tmp_path / "artifacts").close()
    connection = sqlite3.connect(tmp_path / "dsbf.db")
    connection.execute("update alembic_version set version_num = '9999'")
    connection.commit()
    connection.close()

    with pytest.raises(StoreError, match="newer DSBF"):
        SqliteRunRepository(tmp_path / "dsbf.db", tmp_path / "artifacts")

    connection = sqlite3.connect(tmp_path / "dsbf.db")
    assert connection.execute("select * from alembic_version").fetchone() == ("9999",)
    connection.close()


def test_the_default_database_is_under_the_dsbf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DSBF_HOME", str(tmp_path / "home"))

    repo = open_repository()
    repo.close()

    assert (tmp_path / "home" / "dsbf.db").is_file()
