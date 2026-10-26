# dsbf/model/notebook.py
"""The HTML a notebook shows for a run. Every string from the data is escaped."""

from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dsbf.model.run import Run

__all__ = ["render_run_html"]

_TOP_FINDINGS = 10
_TABLE = "border-collapse:collapse"
_CELL = "text-align:left;padding:2px 10px"


def render_run_html(run: "Run") -> str:
    """The health summary and the most severe findings of ``run`` as one fragment."""
    findings = run.findings
    shown = findings[:_TOP_FINDINGS]
    rows = "".join(
        "<tr>"
        + "".join(f"<td style='{_CELL}'>{escape(cell)}</td>" for cell in cells)
        + "</tr>"
        for cells in (
            (
                f.severity.value,
                f.dimension.value,
                ", ".join(f.subject.columns) or f.subject.kind,
                f.title,
            )
            for f in shown
        )
    )
    more = (
        f"<p>{len(findings) - len(shown):,} more findings not shown.</p>"
        if len(findings) > len(shown)
        else ""
    )
    body = (
        f"<table style='{_TABLE}'><tr>"
        + "".join(
            f"<th style='{_CELL}'>{name}</th>"
            for name in ("severity", "dimension", "subject", "finding")
        )
        + f"</tr>{rows}</table>{more}"
        if shown
        else "<p>No findings.</p>"
    )
    return (
        f"<div><pre style='margin:0'>{escape(run.render_health())}</pre>"
        f"<h4 style='margin:8px 0 4px'>Top findings</h4>{body}</div>"
    )
