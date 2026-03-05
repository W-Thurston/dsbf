# dsbf/storage/writer.py

"""
Persists a completed ProfileEngine run into the DSBF SQLite database.

Intended usage
--------------
Called automatically at the end of ProfileEngine.run():

    from dsbf.storage.writer import persist_run
    persist_run(engine)

Can also be called manually against a completed engine instance.
"""

import json
import logging
from typing import Any

from dsbf.storage.schema import get_connection, init_db

logger = logging.getLogger(__name__)


# ── Internal helpers ───────────────────────────────────────────────────────────


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
    """Insert a dataset row if it doesn't exist, return its id either way."""
    conn.execute(
        """
        INSERT INTO datasets (name, source_path)
        VALUES (?, ?)
        ON CONFLICT(name) DO UPDATE SET
            source_path = COALESCE(excluded.source_path, source_path)
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
        logger.info("Run %s already in database — skipping insert.", run_key)
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
    """Insert one row per task result.  Skips tasks already recorded for this run."""
    for task_name, result in results.items():
        # result may be a TaskResult object or a plain dict (from migration)
        if hasattr(result, "__dict__"):
            status = getattr(result, "status", None)
            summary = getattr(result, "summary", None)
            data = getattr(result, "data", None)
        else:
            status = result.get("status")
            summary = result.get("summary")
            data = result.get("data")

        try:
            conn.execute(
                """
        INSERT OR IGNORE INTO task_results (run_id, task_name, status, summary, data)
        VALUES (?, ?, ?, ?, ?)
        """,
                (
                    run_id,
                    task_name,
                    status,
                    _safe_json(summary),
                    _safe_json(data),
                ),
            )
        except Exception as exc:
            logger.warning("Could not insert task result for %s: %s", task_name, exc)


def _insert_figures(conn, run_id: int, results: dict) -> None:
    """
    Walk the 'data' payload of plot-generating tasks and record figure paths.

    Understands the nested structure:
        result.data[column_name][plot_type][format][theme] = file_path

    Dataset-level figures (no column key) follow the same structure but
    the top-level keys are plot type names rather than column names.
    """
    plot_tasks: set[str] = {
        "generate_univariate_plots",
        "generate_dataset_summary_plots",
    }

    for task_name, result in results.items():
        if task_name not in plot_tasks:
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
                # outer_key is plot_type (e.g. "dtype_stacked_bar")
                _record_figure_paths(
                    conn,
                    run_id,
                    task_name,
                    column_name=None,
                    plot_type=outer_key,
                    path_dict=plot_dict,
                )
            else:
                # outer_key is column_name
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
    """
    Walk a figure path dict and insert one row per file path found.

    Handles three nesting shapes that exist in the wild:

      Shape A — standard (format → theme → path):
        {"interactive": {"dark": "fig.json", "light": "fig.json"},
         "static":      {"dark": "fig.png",  "light": "fig.png"}}

      Shape B — composite (format → theme → variant → path):
        {"static": {"dark":  {"box_above": "fig.png", "hist_above": "fig.png"},
                    "light": {"box_above": "fig.png", "hist_above": "fig.png"}}}

      Shape C — theme-less (format → path):
        {"static": "fig.png", "interactive": "fig.json"}

    In Shape B the variant name is appended to plot_type so each row
    is still uniquely identifiable, e.g. plot_type="composite_box_above".
    In Shape C theme is stored as "default".
    """
    for fmt, theme_level in path_dict.items():
        # Shape C — value is already a path string
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
            # Shape A — value is a path string
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

            # Shape B — value is a dict of variant → path
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
    """Insert a single figure row, logging on failure."""
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


# ── Public API ─────────────────────────────────────────────────────────────────


def persist_run(engine, db_path=None) -> int:
    """
    Persist a completed ProfileEngine run to the database.

    Parameters
    ----------
    engine:
        A ProfileEngine instance after engine.run() has been called.
    db_path:
        Optional override for the database path.
        Falls back to DSBF_DB_PATH env var, then dsbf/dsbf.db.

    Returns
    -------
    int
        The database row id of the inserted run.
    """
    init_db(db_path)

    # ── Pull what we need from the engine ──────────────────────────────────
    context = engine.context
    metadata = getattr(engine, "metadata", {}) or {}
    results = getattr(engine, "results", {}) or {}

    # Dataset identity
    dataset_name = (
        getattr(context, "dataset_name", None)
        or metadata.get("dataset_name")
        or "unknown"
    )
    source_path = getattr(context, "dataset_path", None) or metadata.get("dataset_path")

    # Run identity
    run_key = (
        getattr(engine, "run_key", None)
        or getattr(context, "run_key", None)
        or metadata.get("run_key")
    )
    if not run_key:
        raise ValueError(
            "Cannot persist run: no run_key found on engine or context. "
            "Ensure ProfileEngine sets a run_key before calling persist_run()."
        )

    # Shape stats
    shape_data = results.get("summarize_dataset_shape") or {}
    if hasattr(shape_data, "data"):
        shape_data = shape_data.data or {}
    else:
        shape_data = shape_data.get("data") or {}

    row_count = shape_data.get("num_rows") or metadata.get("row_count")
    col_count = shape_data.get("num_columns") or metadata.get("col_count")

    # Quality score
    quality_score = (
        (results.get("data_quality_scorer") or {})
        .get("summary", {})
        .get("overall_score")
        if isinstance((results.get("data_quality_scorer") or {}), dict)
        else getattr(results.get("data_quality_scorer"), "summary", {}).get(
            "overall_score"
        )
    )

    run_meta = {
        "profiling_depth": getattr(context, "profiling_depth", None)
        or metadata.get("profiling_depth"),
        "inferred_stage": getattr(context, "inferred_stage", None)
        or metadata.get("inferred_stage"),
        "config": getattr(engine, "config", None),
        "row_count": row_count,
        "col_count": col_count,
        "quality_score": quality_score,
        "ran_at": metadata.get("ran_at") or getattr(context, "ran_at", None),
    }

    # ── Write everything in one transaction ────────────────────────────────
    with get_connection(db_path) as conn:
        dataset_id = _upsert_dataset(
            conn, dataset_name, str(source_path) if source_path else None
        )
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

    Parameters
    ----------
    run_key:      e.g. "20260303_075308"
    dataset_name: e.g. "titanic"
    source_path:  original CSV path or None
    run_meta:     dict with keys: profiling_depth, inferred_stage, config,
                  row_count, col_count, quality_score, ran_at
    results:      the "results" dict from report.json
    db_path:      optional database path override
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
