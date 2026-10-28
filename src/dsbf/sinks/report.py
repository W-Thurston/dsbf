# dsbf/sinks/report.py
"""The HTML sink: one self-contained page for a run, with inline SVG charts.

The page needs no server and fetches nothing. Every string that comes from the data
or from a finding goes through ``_e``, the only way text enters the markup.
"""

from html import escape
from pathlib import Path

from dsbf.model import Dimension, Run, Severity
from dsbf.model.profile import (
    BooleanProfile,
    CategoricalProfile,
    ColumnProfile,
    DatetimeProfile,
    Histogram,
    NumericProfile,
    TextProfile,
    ValueCount,
)
from dsbf.sinks.registry import Sink, register_sink

REPORT_FILE = "report.html"

_FINDINGS_PER_DIMENSION = 50
_TOP_VALUES_SHOWN = 10
_LABEL_CHARS = 28

_STYLE = """
:root{--bg:#fff;--fg:#1f2328;--muted:#656d76;--line:#d0d7de;--card:#f6f8fa;
--bar:#2f6fb5;--info:#656d76;--notice:#2f6fb5;--warn:#b56a00;--error:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#8b949e;
--line:#30363d;--card:#161b22;--bar:#58a6ff;--info:#8b949e;--notice:#58a6ff;
--warn:#e0a030;--error:#f0776b}}
*{box-sizing:border-box}
body{margin:0 auto;max-width:960px;padding:24px 16px;background:var(--bg);
color:var(--fg);font:15px/1.5 system-ui,sans-serif}
h1{font-size:1.6em;margin:0 0 4px}h2{font-size:1.2em;margin:32px 0 8px;
border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:1em;margin:0}
.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:3px 10px 3px 0;vertical-align:top}
th{font-weight:600;color:var(--muted)}td.n,th.n{text-align:right}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:12px;margin:8px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
details{margin:6px 0}summary{cursor:pointer;font-weight:600}
.sev{font-weight:600}.sev-info{color:var(--info)}.sev-notice{color:var(--notice)}
.sev-warn{color:var(--warn)}.sev-error{color:var(--error)}
pre{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}
svg{display:block;max-width:100%;height:auto}svg text{fill:var(--muted);font-size:10px}
svg rect.bar{fill:var(--bar)}
"""


def _e(value: object) -> str:
    """Escape any value for use in markup or an attribute; the only text path."""
    return escape(str(value), quote=True)


def _num(value: float | None) -> str:
    if value is None:
        return "–"
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.4g}"


def _share(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "–"


def _clip(text: str) -> str:
    return text if len(text) <= _LABEL_CHARS else text[: _LABEL_CHARS - 1] + "…"


def _histogram_svg(histogram: Histogram) -> str:
    width, height, pad = 280, 90, 14
    peak = max(histogram.counts) or 1
    step = width / len(histogram.counts)
    bars = "".join(
        f"<rect class='bar' x='{i * step:.2f}' "
        f"y='{height - count / peak * height:.2f}' "
        f"width='{max(step - 0.5, 0.5):.2f}' height='{count / peak * height:.2f}'>"
        f"<title>{_e(_num(histogram.edges[i]))} to "
        f"{_e(_num(histogram.edges[i + 1]))}: {count:,}</title></rect>"
        for i, count in enumerate(histogram.counts)
    )
    low, high = _e(_num(histogram.edges[0])), _e(_num(histogram.edges[-1]))
    return (
        f"<svg viewBox='0 0 {width} {height + pad}' role='img' "
        f"aria-label='Histogram from {low} to {high}'>{bars}"
        f"<text x='0' y='{height + pad - 2}'>{low}</text>"
        f"<text x='{width}' y='{height + pad - 2}' text-anchor='end'>{high}</text>"
        "</svg>"
    )


def _bars_svg(rows: list[tuple[str, int]], total: int, label: str) -> str:
    width, row, label_w = 280, 16, 110
    peak = max((count for _, count in rows), default=1) or 1
    body = "".join(
        f"<text x='0' y='{i * row + 11}'>{_e(_clip(name))}</text>"
        f"<rect class='bar' x='{label_w}' y='{i * row + 2}' "
        f"width='{max(count / peak * (width - label_w - 40), 1):.2f}' height='11'>"
        f"<title>{_e(name)}: {count:,}</title></rect>"
        f"<text x='{label_w + count / peak * (width - label_w - 40) + 4:.2f}' "
        f"y='{i * row + 11}'>{_share(count, total)}</text>"
        for i, (name, count) in enumerate(rows)
    )
    return (
        f"<svg viewBox='0 0 {width} {len(rows) * row}' role='img' "
        f"aria-label='{_e(label)}'>{body}</svg>"
    )


def _top_values(values: tuple[ValueCount, ...], non_null: int) -> str:
    shown = values[:_TOP_VALUES_SHOWN]
    if not shown:
        return ""
    rows = [(str(v.value), v.count) for v in shown]
    return _bars_svg(rows, non_null, "Most common values")


def _facts(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><th>{_e(name)}</th><td class='n'>{_e(value)}</td></tr>"
        for name, value in rows
    )
    return f"<table>{body}</table>"


def _column_detail(column: ColumnProfile) -> tuple[list[tuple[str, str]], str]:
    """The extra facts and the chart for one column, by its semantic type."""
    non_null = column.counts.non_null
    if isinstance(column, NumericProfile):
        stats = column.stats
        facts = (
            [
                ("min", _num(stats.min)),
                ("mean", _num(stats.mean)),
                ("max", _num(stats.max)),
                ("std", _num(stats.std)),
            ]
            if stats
            else []
        )
        facts.append(("zeros", f"{column.zeros:,}"))
        return facts, _histogram_svg(column.histogram) if column.histogram else ""
    if isinstance(column, CategoricalProfile | TextProfile):
        facts = []
        if column.lengths:
            facts.append(("length", f"{column.lengths.min}–{column.lengths.max}"))
        if column.empty_strings:
            facts.append(("empty strings", f"{column.empty_strings:,}"))
        return facts, _top_values(column.top_values, non_null)
    if isinstance(column, BooleanProfile):
        rows = [("true", column.true_count), ("false", column.false_count)]
        return [], _bars_svg(rows, non_null, "True and false counts")
    if isinstance(column, DatetimeProfile):
        facts = []
        if column.min and column.max:
            facts += [("from", str(column.min)), ("to", str(column.max))]
        if column.inferred_frequency:
            facts.append(("frequency", column.inferred_frequency))
        return facts, ""
    return [], ""


def _column_card(column: ColumnProfile) -> str:
    counts = column.counts
    facts = [
        ("rows", f"{counts.rows:,}"),
        ("nulls", f"{counts.nulls:,} ({_share(counts.nulls, counts.rows)})"),
        ("distinct", f"{'' if counts.distinct_exact else '≈'}{counts.distinct:,}"),
    ]
    extra, chart = _column_detail(column)
    return (
        "<div class='card'>"
        f"<h3>{_e(column.name)}</h3>"
        f"<div class='muted'>{_e(column.semantic_type)} · "
        f"{_e(column.physical_dtype)}</div>"
        f"{_facts(facts + extra)}{chart}</div>"
    )


def _health(run: Run) -> str:
    counts = run.health.counts
    summary = " · ".join(
        f"{getattr(counts, name):,} {name.replace('_', ' ')}"
        for name in type(counts).model_fields
    )
    issues = "".join(
        f"<tr><td>{_e(issue.check_id)}</td><td>{_e(issue.status.value)}</td>"
        f"<td>{_e(issue.message)}"
        + (f"<div class='muted'>{_e(issue.hint)}</div>" if issue.hint else "")
        + "</td></tr>"
        for issue in run.health.issues
    )
    notes = "".join(f"<li>{_e(note.message)}</li>" for note in run.notes)
    return (
        f"<p>{_e(summary)}</p>"
        + (
            f"<table><tr><th>check</th><th>status</th><th>why</th></tr>{issues}</table>"
            if issues
            else ""
        )
        + (f"<ul>{notes}</ul>" if notes else "")
    )


def _policy(run: Run) -> str:
    outcome = run.policy
    if outcome is None:
        return ""
    verdict = "failed" if outcome.failed else "passed"
    rows = "".join(
        f"<tr><td>{_e(result.rule.name)}</td>"
        f"<td class='n'>{len(result.finding_ids) or len(result.check_ids):,}</td>"
        f"<td>{_e(', '.join(a.kind.value for a in result.rule.then))}</td></tr>"
        for result in outcome.rules
        if result.triggered
    )
    return f"<h2>Policy</h2><p><strong>Policy {verdict}.</strong></p>" + (
        f"<table><tr><th>rule</th><th class='n'>matched</th><th>actions</th></tr>"
        f"{rows}</table>"
        if rows
        else "<p class='muted'>No rule matched.</p>"
    )


def _severity(value: Severity) -> str:
    return f"<span class='sev sev-{_e(value.value)}'>{_e(value.value)}</span>"


def _findings(run: Run) -> str:
    findings = run.findings
    if not findings:
        return "<p>No findings.</p>"
    levels = list(Severity)
    header = "".join(f"<th class='n'>{_e(level.value)}</th>" for level in levels)
    rows, sections = [], []
    for dimension in Dimension:
        group = [f for f in findings if f.dimension == dimension]
        if not group:
            continue
        cells = "".join(
            f"<td class='n'>{sum(f.severity == level for f in group):,}</td>"
            for level in levels
        )
        rows.append(f"<tr><td>{_e(dimension.value)}</td>{cells}</tr>")
        items = "".join(
            f"<tr><td>{_severity(f.severity)}</td>"
            f"<td>{_e(', '.join(f.subject.columns) or f.subject.kind)}</td>"
            f"<td>{_e(f.title)}<div class='muted'>{_e(f.message)}</div></td></tr>"
            for f in group[:_FINDINGS_PER_DIMENSION]
        )
        more = len(group) - _FINDINGS_PER_DIMENSION
        tail = f"<p class='muted'>{more:,} more in run.json.</p>" if more > 0 else ""
        sections.append(
            f"<details{' open' if group[0].severity >= Severity.WARN else ''}>"
            f"<summary>{_e(dimension.value)} ({len(group):,})</summary>"
            f"<table>{items}</table>{tail}</details>"
        )
    return (
        f"<table><tr><th>dimension</th>{header}</tr>{''.join(rows)}</table>"
        + "".join(sections)
    )


def render_report(run: Run) -> str:
    """The whole report for ``run`` as one HTML document."""
    dataset = run.dataset
    sampling = dataset.sampling
    scan = f"sample of {sampling.sampled_rows:,} rows" if sampling else "full scan"
    cards = "".join(_column_card(column) for column in run.profile.columns)
    title = f"DSBF report: {dataset.name}"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_e(title)}</title><style>{_STYLE}</style></head><body>"
        f"<h1>{_e(dataset.name)}</h1>"
        f"<div class='muted'>{dataset.n_rows:,} rows × {dataset.n_columns:,} columns · "
        f"{_e(scan)} · run {_e(run.run_id)} · "
        f"{_e(run.finished_at.isoformat(timespec='seconds'))}</div>"
        f"<h2>Health</h2>{_health(run)}"
        f"{_policy(run)}"
        f"<h2>Findings</h2>{_findings(run)}"
        f"<h2>Columns</h2><div class='grid'>{cards}</div>"
        "</body></html>"
    )


def _write(run: Run, path: Path) -> None:
    path.write_text(render_report(run), encoding="utf-8")


register_sink(Sink("html", REPORT_FILE, _write))
