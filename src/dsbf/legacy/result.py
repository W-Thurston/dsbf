# dsbf/legacy/result.py
"""How a legacy ``TaskResult`` is stored in, and read back from, a check's output."""

from dataclasses import fields
from datetime import date
from typing import Any

import numpy as np

from dsbf.eda.task_result import TaskResult
from dsbf.engine import CheckContext, CheckOutput, CheckOutputModel
from dsbf.legacy.findings import findings_from
from dsbf.model import CheckStatus, Severity


class LegacyTaskOutput(CheckOutputModel):
    """A legacy task's whole result, as ``TaskResult.to_dict`` writes it."""

    result: dict[str, Any]


def plain(value: Any) -> Any:
    """``value`` as JSON would write it: keys become strings, arrays lists.

    Legacy results hold whatever the task left in them (int keys, arrays, paths,
    timedeltas). The legacy report writer turned those into JSON as it went; this
    does the same ahead of the contract's strict check, so a result reads back the
    way ``report.json`` showed it. NaN and infinity stay for the contract to null.
    """
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(key): plain(member) for key, member in value.items()}
    if isinstance(value, list | tuple):
        return [plain(member) for member in value]
    if isinstance(value, set | frozenset):
        return sorted((plain(member) for member in value), key=repr)
    if value is None or isinstance(value, bool | int | float | str | date):
        return value
    return str(value)  # a path, a timedelta, anything else a task left behind


def to_output(ctx: CheckContext, result: TaskResult) -> CheckOutput:
    """The check's return value for ``result``, whichever way the task ended.

    A task that returns ``failed`` or ``skipped`` instead of raising ends the check
    ``error`` or ``skipped`` with a note taken from its result; anything else is
    ``ok``, the output is the result itself and each guidance blurb becomes a finding.
    """
    message = str(result.summary.get("message") or "")
    if result.status == "failed":
        meta = result.error_metadata or {}
        return ctx.stop(
            CheckStatus.ERROR,
            "task_failed",
            message or f"{result.name} failed.",
            severity=Severity.ERROR,
            hint=meta.get("suggested_action"),
            exception_type=meta.get("error_type"),
            detail=meta.get("trace_summary"),
        )
    if result.status == "skipped":
        return ctx.stop(
            CheckStatus.SKIPPED,
            "task_skipped",
            message or f"{result.name} was skipped.",
            severity=Severity.INFO,
        )
    stored = plain(result.to_dict())
    findings_from(ctx, stored["guidance"])
    return ctx.output(LegacyTaskOutput(result=stored))


def from_output(output: dict[str, Any]) -> TaskResult:
    """The ``TaskResult`` a stored output holds, as a downstream task reads it."""
    stored = output["result"]
    known = {f.name for f in fields(TaskResult)}
    return TaskResult(**{k: v for k, v in stored.items() if k in known})
