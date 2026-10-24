"""Read the measured rates against the committed baseline, as Markdown."""

from __future__ import annotations

from typing import Any


def _cell(entry: dict[str, Any] | None, rate: str) -> str:
    if entry is None or rate not in entry:
        return ""
    return f"{entry[rate]:.2f}"


def _metrics(entry: dict[str, Any]) -> str:
    return ", ".join(
        f"{name} {metric['mean']:.2f}"
        for name, metric in sorted(entry.get("metrics", {}).items())
    )


def render_summary(results: dict[str, Any], baseline: dict[str, Any]) -> str:
    """A table of every check's rates with the baseline beside them, for the CI job."""
    lines = [
        "| Check | Version | Detection | Baseline | False alarms | Baseline "
        "| Trials | Not scored | Probes |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for check_id, entry in results["checks"].items():
        was = baseline.get("checks", {}).get(check_id)
        unscored = ", ".join(f"{k} {v}" for k, v in entry["unscored"].items())
        lines.append(
            f"| `{check_id}` | {entry['version']} "
            f"| {_cell(entry, 'detection_rate')} | {_cell(was, 'detection_rate')} "
            f"| {_cell(entry, 'false_alarm_rate')} | {_cell(was, 'false_alarm_rate')} "
            f"| {entry['trials']} | {unscored} | {_metrics(entry)} |"
        )
    return "\n".join(lines) + "\n"
