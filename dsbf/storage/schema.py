# dsbf/storage/schema.py
"""
SQLite connection management and schema initialisation.

The database location is resolved in this order:
    1. DSBF_DB_PATH environment variable
    2. db_path argument passed to get_connection() / init_db()
    3. Default: <repo_root>/dsbf/dsbf.db

Usage
-----
from dsbf.storage.schema import init_db, get_connection

init_db()                          # create tables if they don't exist
with get_connection() as conn:     # context manager — commits or rolls back
    conn.execute("SELECT ...")
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

# Default database location
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH: Path = _REPO_ROOT / "dsbf" / "dsbf.db"


def _resolve_db_path(db_path: str | Path | None = None) -> Path:
    """
    Return the resolved Path for the database file.

    Args:
        db_path (str | Path | None, optional): Path for the database file.
        Defaults to None.

    Returns:
        Path: Path for the database file

    """
    if db_path:
        return Path(db_path)
    env: str | None = os.environ.get("DSBF_DB_PATH")
    if env:
        return Path(env)
    return DEFAULT_DB_PATH


# Schema DDL
_DDL = """
-- One row per distinct dataset (identified by name).
CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,   -- e.g. "titanic", "customer_churn"
    source_path TEXT,                      -- original CSV / data path
    created_at  TIMESTAMP DEFAULT (datetime('now'))
);

-- One row per profiling run.
CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id       INTEGER NOT NULL REFERENCES datasets(id),
    run_key          TEXT    NOT NULL UNIQUE,  -- e.g. "20260303_075308"
    profiling_depth  TEXT,                     -- basic | standard | full
    inferred_stage   TEXT,                     -- raw | processed | modelling | ...
    config_snapshot  TEXT,                     -- full config serialised as JSON
    row_count        INTEGER,
    col_count        INTEGER,
    quality_score    REAL,
    ran_at           TIMESTAMP DEFAULT (datetime('now'))
);

-- One row per task per run.
CREATE TABLE IF NOT EXISTS task_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    task_name   TEXT    NOT NULL,
    status      TEXT,                          -- success | failed | skipped
    summary     TEXT,                          -- JSON blob
    data        TEXT,                          -- JSON blob
    UNIQUE (run_id, task_name)
);

-- One row per figure asset per run.
-- column_name is NULL for dataset-level figures (e.g. missingness matrix).
CREATE TABLE IF NOT EXISTS figures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    column_name TEXT,
    task_name   TEXT    NOT NULL,
    plot_type   TEXT    NOT NULL,   -- histogram | boxplot | bar | correlation_matrix...
    theme       TEXT    NOT NULL,   -- dark | light
    format      TEXT    NOT NULL,   -- interactive | static
    file_path   TEXT    NOT NULL    -- path on disk, relative to repo root
);

-- Indexes for the queries we know we'll run most often.
CREATE INDEX IF NOT EXISTS idx_runs_dataset   ON runs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_runs_ran_at    ON runs(ran_at);
CREATE INDEX IF NOT EXISTS idx_task_run       ON task_results(run_id);
CREATE INDEX IF NOT EXISTS idx_task_name      ON task_results(task_name);
CREATE INDEX IF NOT EXISTS idx_figures_run    ON figures(run_id);
CREATE INDEX IF NOT EXISTS idx_figures_col    ON figures(column_name);
"""

# Public API


def init_db(db_path: str | Path | None = None) -> Path:
    """
    Create the database file and all tables if they do not already exist.

    Args:
        db_path (str | Path | None, optional): Path for the database file.
        Defaults to None.

    Returns:
        Path: Path for the database file

    """
    resolved: Path = _resolve_db_path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(resolved) as conn:
        conn.executescript(_DDL)
        conn.execute("PRAGMA journal_mode=WAL;")  # safe for concurrent reads
        conn.execute("PRAGMA foreign_keys=ON;")

    return resolved


@contextmanager
def get_connection(db_path: str | Path | None = None):
    """
    Context manager that yields a configured sqlite3.connection.

    Args:
        db_path (str | Path | None, optional): Path for the database file.
        Defaults to None.

    """
    resolved: Path = _resolve_db_path(db_path)

    if not resolved.exists():
        msg: str = f"Database not found at {resolved}. Run init_db() first."
        raise FileNotFoundError(msg)

    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    conn.execute("PRAGMA foreign_keys=ON;")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
