# dsbf/comparison.py
"""Compare two runs: which findings are new, resolved or changed."""

import math

from dsbf.model import Finding, Run
from dsbf.model.diff import FindingChange, MetricChange, RunDiff, StatusChange

DEFAULT_TOLERANCE = 0.05


def compare(base: Run, head: Run, *, tolerance: float = DEFAULT_TOLERANCE) -> RunDiff:
    """What ``head`` shows that ``base`` did not.

    Findings match by ``Finding.id``. A matched finding is changed when its
    severity differs or a numeric metric (p-values aside) moved by more than
    ``tolerance`` of the base value (any move from or to zero counts); other
    metrics change when they differ. Check statuses are compared by check id.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite number, at least 0")
    before = {f.id: f for f in base.findings}
    after = {f.id: f for f in head.findings}
    changed = [
        change
        for id_, finding in after.items()
        if id_ in before and (change := _change(before[id_], finding, tolerance))
    ]
    key = base.dataset.key
    return RunDiff(
        base_run_id=str(base.run_id),
        head_run_id=str(head.run_id),
        same_dataset=key is not None and key == head.dataset.key,
        tolerance=tolerance,
        new=tuple(f for id_, f in after.items() if id_ not in before),
        resolved=tuple(f for id_, f in before.items() if id_ not in after),
        changed=tuple(changed),
        status_changes=_status_changes(base, head),
    )


def _status_changes(base: Run, head: Run) -> tuple[StatusChange, ...]:
    changes = []
    for check_id in dict.fromkeys([*base.results, *head.results]):
        old, new = base.results.get(check_id), head.results.get(check_id)
        old_status = old.status if old else None
        new_status = new.status if new else None
        if old_status != new_status:
            changes.append(
                StatusChange(check_id=check_id, base=old_status, head=new_status)
            )
    return tuple(changes)


def _change(base: Finding, head: Finding, tolerance: float) -> FindingChange | None:
    moved = tuple(
        MetricChange(name=name, base=_scalar(old), head=_scalar(new))
        for name, old in base.evidence.metrics.items()
        if name in head.evidence.metrics
        and not _is_p_value(name)
        and _differs(old, (new := head.evidence.metrics[name]), tolerance)
    )
    if base.severity == head.severity and not moved:
        return None
    return FindingChange(
        id=head.id,
        code=head.code,
        check_id=head.check_id,
        title=head.title,
        base_severity=base.severity,
        head_severity=head.severity,
        metrics=moved,
    )


def _is_p_value(name: str) -> bool:
    """A p-value moves by orders of magnitude between near-identical runs, so a
    relative change in one says nothing."""
    return (
        name.startswith("p_")
        or "p_value" in name
        or "p_adjusted" in name
        or name.endswith("_p")
    )


def _scalar(value: object) -> bool | int | float | str | None:
    """Collections are shown by their text; the change says only that they differ."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _differs(old: object, new: object, tolerance: float) -> bool:
    numbers = (
        isinstance(old, int | float)
        and isinstance(new, int | float)
        and not isinstance(old, bool)
        and not isinstance(new, bool)
    )
    if not numbers:
        return old != new
    if old == new:
        return False
    if old == 0 or new == 0:
        return True
    return abs(new - old) > tolerance * abs(old)  # type: ignore[operator]
