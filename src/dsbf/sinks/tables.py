# dsbf/sinks/tables.py
"""A run as three flat tables: its findings, its column profiles and its check results.

Each table has a fixed Arrow schema, so a column's type never depends on the data and
an empty table has the same columns as a full one. The column listing is published in
``schema/tables.schema.json``; ``TABLES_SCHEMA_VERSION`` is bumped by the rules that
govern the run schema (a removed or retyped column is breaking, a new nullable column
is additive).
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from dsbf.model import Run
from dsbf.model.profile import (
    BooleanProfile,
    CategoricalProfile,
    DatetimeProfile,
    NumericProfile,
    TextProfile,
)

TABLES_SCHEMA_VERSION = "0.1.0"
SCHEMA_VERSION_KEY = b"schema_version"

_STRING = pa.string()
_INT = pa.int64()
_FLOAT = pa.float64()
_BOOL = pa.bool_()
_STRINGS = pa.list_(pa.string())


@dataclass(frozen=True)
class Column:
    """One column of a table: its name, Arrow type, nullability and meaning."""

    name: str
    type: pa.DataType
    description: str
    nullable: bool = True

    def field(self) -> pa.Field:
        return pa.field(self.name, self.type, nullable=self.nullable)


@dataclass(frozen=True)
class Table:
    """A table's name, file, columns and the function that fills it from a run."""

    name: str
    filename: str
    description: str
    columns: tuple[Column, ...]
    rows: Callable[[Run], list[dict[str, Any]]]

    @property
    def schema(self) -> pa.Schema:
        return pa.schema(
            [column.field() for column in self.columns],
            metadata={SCHEMA_VERSION_KEY: TABLES_SCHEMA_VERSION.encode()},
        )

    def build(self, run: Run) -> pa.Table:
        """The run as an Arrow table with this table's schema."""
        return pa.Table.from_pylist(self.rows(run), schema=self.schema)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


_RUN_ID = Column(
    "run_id", _STRING, "The run these rows belong to; joins the three tables.", False
)


# -- findings ---------------------------------------------------------------------


def _finding_rows(run: Run) -> list[dict[str, Any]]:
    run_id = str(run.run_id)
    rows = []
    for f in run.findings:
        rows.append(
            {
                "run_id": run_id,
                "id": f.id,
                "check_id": f.check_id,
                "check_version": f.check_version,
                "code": f.code,
                "dimension": f.dimension.value,
                "severity": f.severity.value,
                "confidence": f.confidence.value,
                "subject_kind": f.subject.kind,
                "columns": list(f.subject.columns),
                "title": f.title,
                "message": f.message,
                "tags": list(f.tags),
                "method": f.evidence.method,
                "n_used": f.evidence.n_used,
                "n_total": f.evidence.n_total,
                "metrics": _json(f.evidence.metrics),
                "non_finite_metrics": list(f.evidence.non_finite_metrics),
            }
        )
    return rows


FINDINGS = Table(
    "findings",
    "findings.parquet",
    "One row per finding, in the run's order (most severe first).",
    (
        _RUN_ID,
        Column(
            "id",
            _STRING,
            "Stable across runs for the same check, subject and code.",
            False,
        ),
        Column("check_id", _STRING, "The check that raised the finding.", False),
        Column("check_version", _STRING, "The version of that check.", False),
        Column(
            "code",
            _STRING,
            "Machine-readable kind, for example `high_null_rate`.",
            False,
        ),
        Column(
            "dimension",
            _STRING,
            "The quality dimension: `completeness`, `validity` and so on.",
            False,
        ),
        Column("severity", _STRING, "`info`, `notice`, `warn` or `error`.", False),
        Column("confidence", _STRING, "`low`, `medium` or `high`.", False),
        Column(
            "subject_kind",
            _STRING,
            "`dataset`, `column`, `column_pair`, `column_group` or `row_set`.",
            False,
        ),
        Column(
            "columns",
            _STRINGS,
            "The columns the finding is about, in order; empty for the dataset.",
            False,
        ),
        Column("title", _STRING, "A short heading.", False),
        Column(
            "message",
            _STRING,
            "A self-contained sentence with the observed values.",
            False,
        ),
        Column("tags", _STRINGS, "Free-form labels.", False),
        Column(
            "method", _STRING, "The statistic or method behind the evidence.", False
        ),
        Column("n_used", _INT, "Rows the evidence used.", False),
        Column("n_total", _INT, "Rows the evidence could have used.", False),
        Column(
            "metrics",
            _STRING,
            "The evidence metrics as a JSON object; read one with `json_extract`.",
            False,
        ),
        Column(
            "non_finite_metrics",
            _STRINGS,
            "Metrics that were NaN or infinite, so null in `metrics`.",
            False,
        ),
    ),
    _finding_rows,
)


# -- profile ----------------------------------------------------------------------


def _profile_rows(run: Run) -> list[dict[str, Any]]:
    run_id = str(run.run_id)
    rows = []
    for c in run.profile.columns:
        row: dict[str, Any] = {
            "run_id": run_id,
            "name": c.name,
            "position": c.position,
            "semantic_type": c.semantic_type,
            "physical_dtype": c.physical_dtype,
            "semantic_confidence": c.semantic_confidence.value,
            "rows": c.counts.rows,
            "nulls": c.counts.nulls,
            "distinct": c.counts.distinct,
            "distinct_exact": c.counts.distinct_exact,
            "content_hash": c.content_hash,
        }
        if isinstance(c, NumericProfile):
            row |= {"zeros": c.zeros, "non_finite": c.non_finite}
            if c.stats is not None:
                row |= {
                    name: getattr(c.stats, name)
                    for name in ("min", "max", "mean", "std", "skewness", "kurtosis")
                }
                row["quantiles"] = [
                    {"p": q.p, "value": q.value} for q in c.stats.quantiles
                ]
            if c.histogram is not None:
                row["histogram_edges"] = list(c.histogram.edges)
                row["histogram_counts"] = list(c.histogram.counts)
        if isinstance(c, CategoricalProfile | TextProfile):
            row["empty_strings"] = c.empty_strings
            row["entropy_bits"] = c.entropy_bits
            row["top_values"] = [
                {"value": str(v.value), "count": v.count} for v in c.top_values
            ]
            if c.lengths is not None:
                row |= {
                    "length_min": c.lengths.min,
                    "length_max": c.lengths.max,
                    "length_mean": c.lengths.mean,
                }
        if isinstance(c, CategoricalProfile):
            row["mode_count"] = c.mode_count
        if isinstance(c, BooleanProfile):
            row |= {
                "true_count": c.true_count,
                "false_count": c.false_count,
                "entropy_bits": c.entropy_bits,
            }
        if isinstance(c, DatetimeProfile):
            row |= {
                "min_time": c.min.isoformat() if c.min else None,
                "max_time": c.max.isoformat() if c.max else None,
                "timezone": c.timezone,
                "inferred_frequency": c.inferred_frequency,
                "n_gaps": c.n_gaps,
                "unparsed": c.unparsed,
            }
        rows.append(row)
    return rows


_QUANTILE = pa.struct([("p", _FLOAT), ("value", _FLOAT)])
_TOP_VALUE = pa.struct([("value", _STRING), ("count", _INT)])

PROFILE = Table(
    "profile",
    "profile.parquet",
    "One row per column, in the dataset's column order.",
    (
        _RUN_ID,
        Column("name", _STRING, "The column's name.", False),
        Column("position", _INT, "Zero-based place in the dataset.", False),
        Column(
            "semantic_type",
            _STRING,
            "The kind of column: `numeric`, `categorical`, `text` and so on.",
            False,
        ),
        Column("physical_dtype", _STRING, "The storage type.", False),
        Column(
            "semantic_confidence",
            _STRING,
            "How firmly the semantic type is held.",
            False,
        ),
        Column("rows", _INT, "Rows profiled, after any sampling.", False),
        Column("nulls", _INT, "Null values.", False),
        Column("distinct", _INT, "Distinct non-null values.", False),
        Column("distinct_exact", _BOOL, "False when `distinct` is an estimate.", False),
        Column(
            "content_hash",
            _STRING,
            "A 64-bit digest of the values (16 hex digits).",
        ),
        Column("zeros", _INT, "Zero values (numeric columns)."),
        Column("non_finite", _INT, "NaN and infinite values (numeric columns)."),
        Column("min", _FLOAT, "Smallest finite value (numeric columns)."),
        Column("max", _FLOAT, "Largest finite value (numeric columns)."),
        Column("mean", _FLOAT, "Mean of the finite values (numeric columns)."),
        Column("std", _FLOAT, "Standard deviation (numeric columns)."),
        Column("skewness", _FLOAT, "Skewness (numeric columns)."),
        Column("kurtosis", _FLOAT, "Excess kurtosis (numeric columns)."),
        Column(
            "quantiles",
            pa.list_(_QUANTILE),
            "Quantiles as (p, value) pairs in increasing p (numeric columns).",
        ),
        Column(
            "histogram_edges",
            pa.list_(_FLOAT),
            "Equal-width bin edges: one more than the counts (numeric columns).",
        ),
        Column("histogram_counts", pa.list_(_INT), "Values per bin (numeric columns)."),
        Column("empty_strings", _INT, "Empty strings (categorical and text columns)."),
        Column(
            "entropy_bits",
            _FLOAT,
            "Shannon entropy in bits (categorical, text and boolean columns).",
        ),
        Column(
            "top_values",
            pa.list_(_TOP_VALUE),
            "The most common values as text, most common first (categorical, text).",
        ),
        Column(
            "mode_count",
            _INT,
            "Rows holding the most frequent value (categorical columns).",
        ),
        Column(
            "length_min",
            _INT,
            "Shortest string in characters (categorical and text columns).",
        ),
        Column(
            "length_max",
            _INT,
            "Longest string in characters (categorical and text columns).",
        ),
        Column(
            "length_mean", _FLOAT, "Mean string length (categorical and text columns)."
        ),
        Column("true_count", _INT, "True values (boolean columns)."),
        Column("false_count", _INT, "False values (boolean columns)."),
        Column("min_time", _STRING, "Earliest instant, ISO 8601 (datetime columns)."),
        Column("max_time", _STRING, "Latest instant, ISO 8601 (datetime columns)."),
        Column("timezone", _STRING, "The time zone (datetime columns)."),
        Column(
            "inferred_frequency",
            _STRING,
            "An offset alias such as `D`, when the spacing is regular (datetime).",
        ),
        Column("n_gaps", _INT, "Gaps in a regular series (datetime columns)."),
        Column(
            "unparsed",
            _INT,
            "Values that did not read as an instant (datetime columns).",
        ),
    ),
    _profile_rows,
)


# -- results ----------------------------------------------------------------------


def _result_rows(run: Run) -> list[dict[str, Any]]:
    run_id = str(run.run_id)
    rows = []
    for r in run.results.values():
        note = r.notes[0] if r.notes else None
        rows.append(
            {
                "run_id": run_id,
                "check_id": r.check_id,
                "check_version": r.check_version,
                "status": r.status.value,
                "findings": len(r.findings),
                "note_code": note.code if note else None,
                "note_message": note.message if note else None,
                "wall_seconds": r.timing.wall_seconds,
                "cpu_seconds": r.timing.cpu_seconds,
                "peak_memory_bytes": r.timing.peak_memory_bytes,
                "cached": r.timing.cached,
                "params": _json(r.params),
                "output": None if r.output is None else _json(r.output),
            }
        )
    return rows


RESULTS = Table(
    "results",
    "results.parquet",
    "One row per check that was planned, whatever its status.",
    (
        _RUN_ID,
        Column("check_id", _STRING, "The check.", False),
        Column("check_version", _STRING, "The check's version.", False),
        Column(
            "status",
            _STRING,
            "How the check ended: `ok`, `skipped`, `error` and so on.",
            False,
        ),
        Column("findings", _INT, "Findings the check raised.", False),
        Column(
            "note_code",
            _STRING,
            "The code of the first note, which says why a check did not finish.",
        ),
        Column("note_message", _STRING, "The message of that note."),
        Column(
            "wall_seconds", _FLOAT, "Wall time the executor spent on the check.", False
        ),
        Column("cpu_seconds", _FLOAT, "Approximate CPU time."),
        Column("peak_memory_bytes", _INT, "Approximate peak resident memory."),
        Column("cached", _BOOL, "True when the result came from the cache.", False),
        Column("params", _STRING, "The resolved parameters as a JSON object.", False),
        Column(
            "output",
            _STRING,
            "The check's own output as a JSON object; null when it has none.",
        ),
    ),
    _result_rows,
)

TABLES: dict[str, Table] = {t.name: t for t in (FINDINGS, PROFILE, RESULTS)}


def tables_schema() -> dict[str, Any]:
    """The column listing of every table, as the checked-in ``tables.schema.json``."""
    return {
        "schema_version": TABLES_SCHEMA_VERSION,
        "tables": {
            table.name: {
                "file": table.filename,
                "description": table.description,
                "columns": [
                    {
                        "name": c.name,
                        "type": str(c.type),
                        "nullable": c.nullable,
                        "description": c.description,
                    }
                    for c in table.columns
                ],
            }
            for table in TABLES.values()
        },
    }
