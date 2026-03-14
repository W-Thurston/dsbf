# dsbf/storage/writer.py
"""
Persists a completed ProfileEngine run into the DSBF SQLite database.
"""

import json
import logging
import os
from typing import Any

from dsbf.storage.schema import get_connection, init_db

logger = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _safe_json(obj: Any) -> str | None:
    """Serialise obj to a JSON string, returning None if it fails."""
    if obj is None:
        return None
    try:
        return json.dumps(obj, default=str)
    except Exception as exc:
        logger.warning("Could not serialise object to JSON: %s", exc)
        return None


def _upsert_dataset(conn, name: str, source_path: str | None) -> int:
    """
    Insert a dataset row if it doesn't exist, return its id either way.

    Only updates source_path when the new value is non-null AND the existing
    row has no path yet.  This prevents a stale file path from a previous run's
    config being written over a correctly-null built-in dataset entry.
    """
    conn.execute(
        """
        INSERT INTO datasets (name, source_path)
        VALUES (?, ?)
        ON CONFLICT(name) DO UPDATE SET
            source_path = CASE
                WHEN excluded.source_path IS NOT NULL AND datasets.source_path IS NULL
                THEN excluded.source_path
                ELSE datasets.source_path
            END
        """,
        (name, source_path),
    )
    row = conn.execute("SELECT id FROM datasets WHERE name = ?", (name,)).fetchone()
    return row["id"]


def _insert_run(conn, dataset_id: int, run_key: str, meta: dict) -> int:
    """
    Insert a run row, skipping gracefully if run_key already exists.

    Returns the run id (existing or newly created).
    """
    existing = conn.execute(
        "SELECT id FROM runs WHERE run_key = ?", (run_key,)
    ).fetchone()
    if existing:
        logger.info("Run %s already in database - skipping insert.", run_key)
        return existing["id"]

    conn.execute(
        """
        INSERT INTO runs (
            dataset_id, run_key, profiling_depth, inferred_stage,
            config_snapshot, row_count, col_count, quality_score, ran_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset_id,
            run_key,
            meta.get("profiling_depth"),
            meta.get("inferred_stage"),
            _safe_json(meta.get("config")),
            meta.get("row_count"),
            meta.get("col_count"),
            meta.get("quality_score"),
            meta.get("ran_at"),
        ),
    )
    row = conn.execute("SELECT id FROM runs WHERE run_key = ?", (run_key,)).fetchone()
    return row["id"]


def _insert_task_results(conn, run_id: int, results: dict) -> None:
    """Insert one row per task result. Skips tasks already recorded for this run."""
    for task_name, result in results.items():
        if hasattr(result, "__dict__"):
            status = getattr(result, "status", None)
            summary = getattr(result, "summary", None)
            data = getattr(result, "data", None)
            guidance = getattr(result, "guidance", None)
        else:
            status = result.get("status")
            summary = result.get("summary")
            data = result.get("data")
            guidance = result.get("guidance")

        try:
            conn.execute(
                """
INSERT OR IGNORE INTO task_results (run_id, task_name, status, summary, data, guidance)
VALUES (?, ?, ?, ?, ?, ?)
        """,
                (
                    run_id,
                    task_name,
                    status,
                    _safe_json(summary),
                    _safe_json(data),
                    _safe_json(guidance),
                ),
            )
        except Exception as exc:
            logger.warning("Could not insert task result for %s: %s", task_name, exc)


def _insert_figures(conn, run_id: int, results: dict) -> None:
    """Walk the 'data' payload of plot-generating tasks and record figure paths."""
    PLOT_TASKS = {"generate_univariate_plots", "generate_dataset_summary_plots"}

    for task_name, result in results.items():
        if task_name not in PLOT_TASKS:
            continue

        if hasattr(result, "data"):
            data = result.data or {}
        else:
            data = result.get("data") or {}

        is_dataset_level = task_name == "generate_dataset_summary_plots"

        for outer_key, plot_dict in data.items():
            if not isinstance(plot_dict, dict):
                continue

            if is_dataset_level:
                _record_figure_paths(
                    conn,
                    run_id,
                    task_name,
                    column_name=None,
                    plot_type=outer_key,
                    path_dict=plot_dict,
                )
            else:
                for plot_type, path_dict in plot_dict.items():
                    if not isinstance(path_dict, dict):
                        continue
                    _record_figure_paths(
                        conn,
                        run_id,
                        task_name,
                        column_name=outer_key,
                        plot_type=plot_type,
                        path_dict=path_dict,
                    )


def _record_figure_paths(
    conn,
    run_id: int,
    task_name: str,
    column_name: str | None,
    plot_type: str,
    path_dict: dict,
) -> None:
    for fmt, theme_level in path_dict.items():
        if isinstance(theme_level, str):
            _insert_figure_row(
                conn,
                run_id,
                column_name,
                task_name,
                plot_type,
                theme="default",
                fmt=fmt,
                file_path=theme_level,
            )
            continue

        if not isinstance(theme_level, dict):
            continue

        for theme, variant_level in theme_level.items():
            if isinstance(variant_level, str):
                _insert_figure_row(
                    conn,
                    run_id,
                    column_name,
                    task_name,
                    plot_type,
                    theme=theme,
                    fmt=fmt,
                    file_path=variant_level,
                )
            elif isinstance(variant_level, dict):
                for variant, file_path in variant_level.items():
                    if isinstance(file_path, str) and file_path:
                        _insert_figure_row(
                            conn,
                            run_id,
                            column_name,
                            task_name,
                            plot_type=f"{plot_type}_{variant}",
                            theme=theme,
                            fmt=fmt,
                            file_path=file_path,
                        )


def _insert_figure_row(
    conn,
    run_id: int,
    column_name: str | None,
    task_name: str,
    plot_type: str,
    theme: str,
    fmt: str,
    file_path: str,
) -> None:
    if not file_path:
        return
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO figures
                (run_id, column_name, task_name, plot_type, theme, format, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, column_name, task_name, plot_type, theme, fmt, file_path),
        )
    except Exception as exc:
        logger.warning("Could not insert figure record: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────


def persist_run(engine, db_path=None) -> int:
    """
    Persist a completed ProfileEngine run to the database.
    """
    init_db(db_path)

    context = engine.context
    config = getattr(engine, "config", {}) or {}
    results = getattr(engine, "results", {}) or {}
    cfg_meta = config.get("metadata", {}) or {}

    # Run identity - run_key is the basename of the timestamped output dir
    run_key = (
        getattr(engine, "run_key", None)
        or os.path.basename(getattr(engine, "output_dir", "") or "")
        or None
    )
    if not run_key:
        raise ValueError(
            "Cannot persist run: could not determine run_key from engine.output_dir."
        )

    # Dataset identity
    dataset_name = cfg_meta.get("dataset_name") or "unknown"
    # Determine source_path:
    # - File-based runs (profile command): dataset_path is set, use it.
    # - Built-in runs (quickstart): dataset_path is None AND dataset_source is
    #   seaborn/sklearn/openml. Store None so no stale path pollutes the DB.
    # - If dataset_path is present, always trust it regardless of dataset_source,
    #   since the profile command sets the path explicitly.
    raw_path = cfg_meta.get("dataset_path") or None
    dataset_source = cfg_meta.get("dataset_source") or "file"
    is_builtin = (raw_path is None) and (
        dataset_source in ("seaborn", "sklearn", "openml")
    )
    source_path = None if is_builtin else raw_path

    # Shape stats - result may be a TaskResult object or plain dict
    _shape_result = results.get("summarize_dataset_shape") or {}
    shape_data = (
        _shape_result.data
        if hasattr(_shape_result, "data")
        else _shape_result.get("data") or {}
    ) or {}
    row_count = shape_data.get("num_rows")
    col_count = shape_data.get("num_columns")

    # Quality score — the scorer no longer produces a single numeric score.
    # The runs table quality_score column is retained for schema compatibility
    # but always written as None. The dashboard uses the per-category levels
    # from the scorer's data field instead.
    quality_score = None

    # Profiling depth and stage
    profiling_depth = cfg_meta.get("profiling_depth") or getattr(
        context, "profiling_depth", None
    )
    inferred_stage = getattr(engine, "inferred_stage", None) or getattr(
        context, "stage", None
    )

    # Derive ran_at from run_key timestamp
    from datetime import datetime

    try:
        ran_at = datetime.strptime(run_key, "%Y%m%d_%H%M%S").isoformat(sep=" ")
    except ValueError:
        ran_at = None

    run_meta = {
        "profiling_depth": profiling_depth,
        "inferred_stage": inferred_stage,
        "config": config,
        "row_count": row_count,
        "col_count": col_count,
        "quality_score": quality_score,
        "ran_at": ran_at,
    }

    with get_connection(db_path) as conn:
        dataset_id = _upsert_dataset(conn, dataset_name, source_path)
        run_id = _insert_run(conn, dataset_id, run_key, run_meta)
        _insert_task_results(conn, run_id, results)
        _insert_figures(conn, run_id, results)

    logger.info(
        "Run %s persisted to database (dataset=%s, run_id=%d).",
        run_key,
        dataset_name,
        run_id,
    )
    return run_id


def persist_run_from_dict(
    run_key: str,
    dataset_name: str,
    source_path: str | None,
    run_meta: dict,
    results: dict,
    db_path=None,
) -> int:
    """
    Persist a run from raw dicts rather than a live engine instance.
    Used by the migration script to load completed runs from disk.
    """
    init_db(db_path)

    with get_connection(db_path) as conn:
        dataset_id = _upsert_dataset(conn, dataset_name, source_path)
        run_id = _insert_run(conn, dataset_id, run_key, run_meta)
        _insert_task_results(conn, run_id, results)
        _insert_figures(conn, run_id, results)

    logger.info(
        "Run %s persisted from dict (dataset=%s, run_id=%d).",
        run_key,
        dataset_name,
        run_id,
    )
    return run_id
