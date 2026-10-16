"""Turn a finished DSBF run into a stable snapshot that is practical to review.

A raw ``report.json`` is megabytes of floats, timings and paths. The snapshot keeps what
a behaviour change should be judged on: each task's status and summary, the guidance it
raised, its warnings, and its result data. It drops anything that differs between two
identical runs, rounds floats, and replaces results too large to read in a diff with
their length and a digest, so any change to them still fails the comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

SNAPSHOT_FORMAT = 1
SIG_FIGS = 6

# A collection whose compact JSON is longer than this is searched, not kept as is.
INLINE_CHARS = 2000
# A dict or list with more entries than this is replaced by its length and a digest.
MAX_ENTRIES = 40

# Tasks whose whole result is a timing or memory measurement.
VOLATILE_TASKS = frozenset({"identify_bottleneck_tasks", "log_resource_usage"})
# Keys that hold timings, memory or timestamps wherever they appear.
_VOLATILE_KEY = re.compile(
    r"memory|duration|elapsed|timestamp|^(start|end)_time$", re.IGNORECASE
)

_NON_FINITE = {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}


def round_sig(value: float, digits: int = SIG_FIGS) -> float | str:
    """Round to ``digits`` significant figures; spell out NaN and infinities."""
    if math.isnan(value):
        return _NON_FINITE["nan"]
    if math.isinf(value):
        return _NON_FINITE["inf"] if value > 0 else _NON_FINITE["-inf"]
    if value == 0:
        return 0.0
    return float(f"{value:.{digits - 1}e}")


def scrub_text(text: str, roots: list[str]) -> str:
    """Replace every run-specific directory in ``text`` with a placeholder."""
    for root in sorted(roots, key=len, reverse=True):
        text = text.replace(root, "<run>")
    return text


def canonical(value: Any) -> Any:
    """Return ``value`` with floats rounded, volatile keys dropped and keys sorted."""
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return round_sig(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {
            str(k): canonical(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            if not _VOLATILE_KEY.search(str(k))
        }
    if isinstance(value, list | tuple):
        return [canonical(v) for v in value]
    return str(value)


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha1(compact_json(value).encode("utf-8")).hexdigest()[:12]


def reduce_value(value: Any) -> Any:
    """Keep small structures whole and summarise large ones.

    ``value`` must already be canonical. A structure that fits within ``INLINE_CHARS``
    is kept as is. A larger dict is searched value by value, unless it has more than
    ``MAX_ENTRIES`` keys; a larger list or over-wide dict becomes its length and a
    digest.
    """
    if not isinstance(value, dict | list):
        return value
    if len(compact_json(value)) <= INLINE_CHARS:
        return value
    if len(value) > MAX_ENTRIES:
        return {"_len": len(value), "_digest": digest(value)}
    if isinstance(value, dict):
        return {k: reduce_value(v) for k, v in value.items()}
    return {"_len": len(value), "_digest": digest(value)}


def flatten_guidance(guidance: Any) -> list[dict[str, str | None]]:
    """List guidance items as ``path``, ``phase``, ``column``, ``level``, ``title``.

    Guidance is nested by column and phase in a shape that varies by task, so this
    finds items by their ``level`` and ``title`` keys rather than by position. Bodies
    are left out: titles carry the numbers that matter and the bodies are long prose.
    """
    items: list[dict[str, str | None]] = []

    def walk(node: Any, path: list[str]) -> None:
        if isinstance(node, dict):
            if "level" in node and "title" in node:
                items.append(
                    {
                        "path": "/".join(path),
                        "phase": node.get("phase"),
                        "column": node.get("column"),
                        "level": node.get("level"),
                        "title": node.get("title"),
                    }
                )
                return
            for key, child in node.items():
                walk(child, [*path, str(key)])
        elif isinstance(node, list):
            for child in node:
                walk(child, path)

    walk(guidance, [])
    return sorted(items, key=compact_json)


def normalise_task(
    name: str, result: dict[str, Any], roots: list[str]
) -> dict[str, Any]:
    """Reduce one task result from ``report.json`` to its snapshot entry."""
    entry: dict[str, Any] = {"status": result.get("status")}
    if name in VOLATILE_TASKS:
        return entry

    def clean(value: Any) -> Any:
        scrubbed = json.loads(scrub_text(compact_json(value), roots))
        return reduce_value(canonical(scrubbed))

    entry["summary"] = clean(result.get("summary"))
    entry["data"] = clean(result.get("data"))
    entry["guidance"] = json.loads(
        scrub_text(compact_json(flatten_guidance(result.get("guidance"))), roots)
    )
    entry["reliability_warnings"] = clean(result.get("reliability_warnings"))
    entry["recommendations"] = clean(result.get("recommendations"))
    entry["ml_impact_score"] = clean(result.get("ml_impact_score"))
    error = result.get("error_metadata")
    entry["error"] = clean(error) if error else None
    return entry


def build_snapshot(
    dataset: str,
    report: dict[str, Any],
    run_info: dict[str, Any],
    roots: list[str],
) -> dict[str, Any]:
    """Assemble the snapshot for one dataset from its report and run information."""
    results = report["results"]
    return {
        "format": SNAPSHOT_FORMAT,
        "dataset": dataset,
        "run": canonical(json.loads(scrub_text(compact_json(run_info), roots))),
        "top_ml_risks": reduce_value(
            canonical(
                json.loads(scrub_text(compact_json(report["top_ml_risks"]), roots))
            )
        ),
        "tasks": {
            name: normalise_task(name, results[name], roots) for name in sorted(results)
        },
    }


def render_snapshot(snapshot: dict[str, Any]) -> str:
    """Serialise with fixed formatting so identical content is byte-identical."""
    return json.dumps(snapshot, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
