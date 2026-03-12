# dsbf/api/db.py

"""
Read-only query helpers for the DSBF SQLite database.

All functions return plain dicts/lists — no SQLite Row objects leak out.

Row objects are always converted to dicts INSIDE the connection context
manager to avoid accessing them after the connection is closed.
"""

import contextlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from dsbf.storage.schema import get_connection


def _row_to_dict(row) -> dict:
    return dict(row)


def _parse_json_fields(d: dict, *fields: str) -> dict:
    """Deserialise named fields from JSON strings to Python objects in-place."""
    for field in fields:
        if isinstance(d.get(field), str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                d[field] = json.loads(d[field])
    return d


# ── Datasets ───────────────────────────────────────────────────────────────────


def list_datasets(db_path=None) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                d.id,
                d.name,
                d.source_path,
                d.created_at,
                COUNT(r.id)          AS run_count,
                latest.run_key       AS latest_run_key,
                latest.ran_at        AS latest_ran_at,
                latest.quality_score AS latest_quality_score
            FROM datasets d
            LEFT JOIN runs r ON r.dataset_id = d.id
            LEFT JOIN (
                SELECT dataset_id, run_key, ran_at, quality_score
                FROM runs
                WHERE (dataset_id, ran_at) IN (
                    SELECT dataset_id, MAX(ran_at)
                    FROM runs
                    GROUP BY dataset_id
                )
            ) latest ON latest.dataset_id = d.id
            GROUP BY d.id
            ORDER BY d.name
        """,
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_dataset(name: str, db_path=None) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM datasets WHERE name = ?", (name,)).fetchone()
        return _row_to_dict(row) if row else None


# ── Runs ───────────────────────────────────────────────────────────────────────


def list_runs(dataset_name: str, db_path=None) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                r.id,
                r.run_key,
                r.profiling_depth,
                r.inferred_stage,
                r.row_count,
                r.col_count,
                r.quality_score,
                r.ran_at,
                COUNT(DISTINCT f.id) AS fig_count,
                COUNT(DISTINCT t.id) AS task_count
            FROM runs r
            JOIN datasets d          ON d.id = r.dataset_id
            LEFT JOIN figures f      ON f.run_id = r.id
            LEFT JOIN task_results t ON t.run_id = r.id
            WHERE d.name = ?
            GROUP BY r.id
            ORDER BY r.ran_at DESC
        """,
            (dataset_name,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_run(run_key: str, db_path=None) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                r.*,
                d.name AS dataset_name,
                d.source_path AS source_path
            FROM runs r
            JOIN datasets d ON d.id = r.dataset_id
            WHERE r.run_key = ?
        """,
            (run_key,),
        ).fetchone()
        if not row:
            return None
        d = _row_to_dict(row)
    _parse_json_fields(d, "config_snapshot")

    # Surface dataset_source and dataset_name from config snapshot so the
    # frontend can distinguish file-based vs built-in datasets without
    # treating source_path=None as the only signal.
    cfg_meta = {}
    if isinstance(d.get("config_snapshot"), dict):
        cfg_meta = d["config_snapshot"].get("metadata", {}) or {}
    # source_path is the ground truth — if a file path is stored the run is
    # file-based regardless of what dataset_source says in the config snapshot
    # (the profile command sets dataset_path but may leave dataset_source as
    # "seaborn" from the default config).
    if d.get("source_path"):
        d["dataset_source"] = "file"
    else:
        d["dataset_source"] = cfg_meta.get("dataset_source") or "unknown"
    # dataset_name from config_snapshot is more reliable than the datasets table
    # name for built-in datasets (e.g. "titanic" vs a generated fallback name)
    if cfg_meta.get("dataset_name"):
        d["dataset_name"] = cfg_meta["dataset_name"]

    sp = d.get("source_path")
    # source_path may be relative to the repo root (same convention as figure paths)
    resolved = None
    if sp:
        p = Path(sp)
        if not p.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            p = repo_root / p
        if p.exists():
            resolved = p

    if resolved:
        stat = resolved.stat()
        d["source_file_size_bytes"] = stat.st_size
        d["source_last_modified"] = stat.st_mtime
    else:
        d["source_file_size_bytes"] = None
        d["source_last_modified"] = None
    return d


def _load_dataframe(
    run_key: str,
    source_path: str | None,
    columns: list[str] | None = None,
    db_path=None,
) -> "pd.DataFrame | None":
    """
    Load data for a run from either a file path or a built-in dataset.
    Returns a DataFrame, None if unavailable, or a dict {"error": ...} on failure.

    Built-in datasets (seaborn / sklearn) are identified when source_path is null
    but the run's dataset has a known name and the config records the source.
    """
    # ── File-based load ───────────────────────────────────────────────────────
    if source_path:
        p = Path(source_path)
        if not p.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            p = repo_root / p
        if not p.exists():
            return None
        ext = p.suffix.lower()
        try:
            if ext == ".parquet":
                df = pd.read_parquet(str(p), columns=columns)
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(str(p), usecols=columns)
            else:
                df = pd.read_csv(str(p), usecols=columns)
            return df
        except Exception as e:
            return {"error": str(e)}  # type: ignore[return-value]

    # ── Built-in dataset fallback (seaborn / sklearn) ─────────────────────────
    # Look up the dataset name and source from the run config snapshot in the DB
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT r.config_snapshot, d.name AS dataset_name
            FROM runs r
            JOIN datasets d ON d.id = r.dataset_id
            WHERE r.run_key = ?
            """,
            (run_key,),
        ).fetchone()
    if not row:
        return None

    d = _row_to_dict(row)
    _parse_json_fields(d, "config_snapshot")
    cfg = d.get("config_snapshot") or {}
    meta = cfg.get("metadata", {}) if isinstance(cfg, dict) else {}
    ds_name = meta.get("dataset_name") or d.get("dataset_name") or ""
    ds_source = meta.get("dataset_source") or "seaborn"

    if not ds_name:
        return None

    try:
        from dsbf.utils.data_loader import load_dataset

        df = load_dataset(name=ds_name, source=ds_source, backend="pandas")
        if columns:
            existing = [c for c in columns if c in df.columns]
            df = df[existing] if existing else df
        return df
    except Exception as e:
        return {"error": str(e)}  # type: ignore[return-value]


def get_run_sample(run_key: str, n: int = 10, db_path=None) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT d.source_path
            FROM runs r
            JOIN datasets d ON d.id = r.dataset_id
            WHERE r.run_key = ?
        """,
            (run_key,),
        ).fetchone()
    if not row:
        return None
    source_path = _row_to_dict(row).get("source_path")
    df = _load_dataframe(run_key, source_path, db_path=db_path)
    if df is None:
        return None
    if isinstance(df, dict):  # error dict from _load_dataframe
        return {"error": df.get("error", "Unknown error"), "columns": [], "rows": []}
    df = df.head(n)
    # Serialise safely:
    # 1. Convert datetime columns to ISO strings
    for col in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
        df[col] = df[col].astype(str)
    # 2. Serialise row-by-row using pd.isna() which catches ALL NA types:
    #    numpy.float64 NaN, numpy.bool_, NaT, pd.NA, None — everything.
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            try:
                if pd.isna(val):
                    record[col] = None
                    continue
            except (TypeError, ValueError):
                pass
            # Convert numpy scalars to native Python types so json.dumps works
            record[col] = val.item() if hasattr(val, "item") else val
        records.append(record)
    return {"columns": list(df.columns), "rows": records}


# ── Task results ───────────────────────────────────────────────────────────────


def get_run_tasks(run_key: str, db_path=None) -> dict[str, Any]:
    with get_connection(db_path) as conn:
        run = conn.execute(
            "SELECT id FROM runs WHERE run_key = ?",
            (run_key,),
        ).fetchone()
        if not run:
            return {}
        rows = conn.execute(
            """
            SELECT task_name, status, summary, data
            FROM task_results
            WHERE run_id = ?
            ORDER BY task_name
        """,
            (run["id"],),
        ).fetchall()
        raw = [_row_to_dict(r) for r in rows]

    result = {}
    for d in raw:
        task_name = d.pop("task_name")
        _parse_json_fields(d, "summary", "data")
        result[task_name] = d
    return result


def get_task(run_key: str, task_name: str, db_path=None) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT tr.task_name, tr.status, tr.summary, tr.data
            FROM task_results tr
            JOIN runs r ON r.id = tr.run_id
            WHERE r.run_key = ? AND tr.task_name = ?
        """,
            (run_key, task_name),
        ).fetchone()
        if not row:
            return None
        d = _row_to_dict(row)
    _parse_json_fields(d, "summary", "data")
    return d


def get_column_data(
    run_key: str,
    columns: list[str],
    max_rows: int = 3000,
    db_path=None,
) -> dict | None:
    """
    Return raw column values from the source file for a run.
    Subsamples to max_rows if the dataset is larger.
    Used by the Relationships tab for on-the-fly pair plots.
    """
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT d.source_path
            FROM runs r
            JOIN datasets d ON d.id = r.dataset_id
            WHERE r.run_key = ?
            """,
            (run_key,),
        ).fetchone()
    if not row:
        return None
    source_path = _row_to_dict(row).get("source_path")
    df = _load_dataframe(run_key, source_path, columns=columns, db_path=db_path)
    if df is None:
        return None
    if isinstance(df, dict):  # error dict
        return df

    total_rows = len(df)

    # Subsample deterministically if needed
    if total_rows > max_rows:
        import math

        step = math.ceil(total_rows / max_rows)
        df = df.iloc[::step].head(max_rows)

    # Serialise: None for NaN/NaT, native Python types for numpy scalars
    result: dict[str, list] = {"total_rows": total_rows, "sampled_rows": len(df)}
    for col in columns:
        if col not in df.columns:
            result[col] = []
            continue
        series = df[col]
        vals = []
        for v in series:
            try:
                if pd.isna(v):
                    vals.append(None)
                    continue
            except (TypeError, ValueError):
                pass
            vals.append(v.item() if hasattr(v, "item") else v)
        result[col] = vals

    return result


# ── Figures ────────────────────────────────────────────────────────────────────


def get_run_figures(run_key: str, db_path=None) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT f.id, f.column_name, f.task_name, f.plot_type,
                   f.theme, f.format, f.file_path
            FROM figures f
            JOIN runs r ON r.id = f.run_id
            WHERE r.run_key = ?
            ORDER BY f.column_name NULLS FIRST, f.plot_type, f.theme, f.format
        """,
            (run_key,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_figure_path(figure_id: int, db_path=None) -> str | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT file_path FROM figures WHERE id = ?",
            (figure_id,),
        ).fetchone()
        return row["file_path"] if row else None


# ── Run comparison ─────────────────────────────────────────────────────────────


def compare_runs(run_keys: list[str], task_name: str, db_path=None) -> dict:
    with get_connection(db_path) as conn:
        # Use a temp table to keep every value fully parameterised —
        # avoids f-string SQL construction flagged by Ruff S608.
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS _run_key_filter (run_key TEXT)")
        conn.execute("DELETE FROM _run_key_filter")
        conn.executemany(
            "INSERT INTO _run_key_filter VALUES (?)",
            [(k,) for k in run_keys],
        )
        rows: list[Any] = conn.execute(
            """
            SELECT r.run_key, tr.summary
            FROM task_results tr
            JOIN runs r ON r.id = tr.run_id
            JOIN _run_key_filter f ON f.run_key = r.run_key
            WHERE tr.task_name = ?
            ORDER BY r.ran_at
        """,
            (task_name,),
        ).fetchall()
        raw = [_row_to_dict(r) for r in rows]

    result = {}
    for d in raw:
        summary = d["summary"]
        if isinstance(summary, str):
            with contextlib.suppress(json.JSONDecodeError):
                summary = json.loads(summary)
        result[d["run_key"]] = summary
    return result
