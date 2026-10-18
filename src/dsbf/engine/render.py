# dsbf/engine/render.py
"""A plan as plain text, for ``dsbf plan`` and for logs."""

from dsbf.engine.expectation import Expected
from dsbf.engine.plan import Plan, PlanEntry

_MARKS = {
    Expected.WILL_RUN: "✓",
    Expected.NOT_APPLICABLE: "–",
    Expected.UNAVAILABLE: "⦸",
    Expected.SKIPPED: "↷",
}
_MIN_ID_WIDTH = 22
_STATUS_WIDTH = len("not applicable") + 2


def format_plan(plan: Plan, *, dataset: str | None = None) -> str:
    """The plan by level, one line per check, then its notes.

    ``dataset`` names what the plan was made for, such as ``titanic (891 × 15)``.
    """
    header = "Plan" if dataset is None else f"Plan for {dataset}"
    lines = [f"{header}: {_count(plan)}", f"  {_summary(plan)}"]
    id_width = max([_MIN_ID_WIDTH, *(len(entry.id) + 2 for entry in plan)])
    for number, level in enumerate(plan.levels):
        lines += ["", f"level {number}"]
        lines += [_entry_line(entry, id_width) for entry in level]
    for note in plan.notes:
        hint = f" Hint: {note.hint}" if note.hint else ""
        lines += ["", f"⚠ {note.message}{hint}"]
    return "\n".join(lines)


def _count(plan: Plan) -> str:
    required = sum(not entry.selected for entry in plan)
    noun = "check" if len(plan) == 1 else "checks"
    extra = f", {required} pulled in as requirements" if required else ""
    return f"{len(plan)} {noun}{extra}"


def _summary(plan: Plan) -> str:
    parts = [
        f"{status.value.replace('_', ' ')} "
        f"{sum(entry.expected.status is status for entry in plan)}"
        for status in Expected
        if any(entry.expected.status is status for entry in plan)
    ]
    summary = " · ".join(parts) or "nothing to run"
    if plan and not plan.profiled:
        summary += " · columns not assessed (no profile)"
    return summary


def _entry_line(entry: PlanEntry, id_width: int) -> str:
    expected = entry.expected
    status = expected.status.value.replace("_", " ")
    line = (
        f"  {_MARKS[expected.status]} {status:<{_STATUS_WIDTH}}{entry.id:<{id_width}}"
    )
    text = [_reads(entry)] if expected.status is Expected.WILL_RUN else []
    if expected.reason:
        hint = f" Hint: {expected.hint}" if expected.hint else ""
        text.append(expected.reason + hint)
    text += _relations(entry)
    return (line + " · ".join(part for part in text if part)).rstrip()


def _reads(entry: PlanEntry) -> str:
    columns = entry.expected.columns
    if columns is None:
        return ""
    return f"reads {len(columns)} column{'' if len(columns) == 1 else 's'}"


def _relations(entry: PlanEntry) -> list[str]:
    parts = []
    if entry.requires and entry.expected.status is not Expected.SKIPPED:
        parts.append(f"requires {', '.join(entry.requires)}")
    if entry.after:
        parts.append(f"after {', '.join(entry.after)}")
    if entry.required_by:
        marker = "" if entry.selected else "added: "
        parts.append(f"{marker}required by {', '.join(entry.required_by)}")
    changed = _changed_params(entry)
    if changed:
        parts.append(", ".join(f"{key}={value!r}" for key, value in changed.items()))
    return parts


def _changed_params(entry: PlanEntry) -> dict[str, object]:
    defaults = entry.check.spec.params().model_dump(mode="json")
    return {
        key: value
        for key, value in entry.params.model_dump(mode="json").items()
        if value != defaults.get(key)
    }
