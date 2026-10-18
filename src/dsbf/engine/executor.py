# dsbf/engine/executor.py
"""Execute a plan: every check ends in an explicit status, and the run completes."""

import time
import traceback
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from functools import partial
from uuid import uuid4

from dsbf.data import Dataset
from dsbf.engine.errors import CheckContractError
from dsbf.engine.events import (
    CheckFinished,
    CheckStarted,
    Dispatcher,
    Hook,
    NoteRecorded,
)
from dsbf.engine.expectation import Expected
from dsbf.engine.limits import Limits
from dsbf.engine.meter import DEFAULT_SAMPLE_SECONDS, Meter, Window
from dsbf.engine.plan import Plan, PlanEntry
from dsbf.engine.worker import attempt_check
from dsbf.model import (
    CheckResult,
    CheckStatus,
    DatasetManifest,
    DatasetProfile,
    Note,
    PlanManifest,
    PlannedCheck,
    Run,
    Severity,
    Timing,
)

# A hard dependent runs on these. Anything else (a failure, a timeout, a skip, a
# missing extra) blocks it; "not applicable" only says the upstream had nothing to read.
_SATISFIES_DEPENDENTS = (CheckStatus.OK, CheckStatus.NOT_APPLICABLE)
_BUDGET_HINT = "Raise the run budget, or select a lighter depth or fewer checks."


def execute(
    plan: Plan,
    dataset: Dataset,
    profile: DatasetProfile,
    *,
    dsbf_version: str,
    limits: Limits = Limits(),
    hooks: Iterable[Hook] = (),
    memory_sample_seconds: float | None = DEFAULT_SAMPLE_SECONDS,
) -> Run:
    """Run every check in ``plan`` against ``dataset`` and return the ``Run``.

    Checks run one after another, level by level, each on its own thread. Nothing a
    check does stops the run: an exception becomes an ``error`` result with a note, a
    check that outlasts ``limits.check_timeout_seconds`` is ``timeout``, a check whose
    hard requirement did not complete is ``skipped`` (a soft ``after`` upstream that
    failed changes nothing), and a check missing an optional dependency is
    ``unavailable``. Once ``limits.run_budget_seconds`` is used up the running check
    times out and the rest are ``skipped``. Every planned check has a result, and
    ``Run.health`` says why any did not complete. ``KeyboardInterrupt`` and
    ``SystemExit`` are not caught.

    A thread cannot be killed: a timed-out check that never calls back into its
    context keeps running in the background until it returns (see
    ``CheckContext.checkpoint``). Each result's ``timing`` is approximate: the wall
    time is the executor's, the CPU time is the whole process's, and the memory is
    the process's resident peak found by sampling every ``memory_sample_seconds``
    (``None`` skips memory).

    Each ``hook`` receives a ``CheckStarted``, a ``NoteRecorded`` for every note and
    a ``CheckFinished``, in that order, from the calling thread. A check that is not
    called has no ``CheckStarted``. A hook that raises is logged and dropped.

    ``profile`` must be the profile of ``dataset``.

    Raises:
        ValueError: ``profile`` is not the profile of ``dataset``, or
            ``memory_sample_seconds`` is not positive.
    """
    if [column.name for column in profile.columns] != dataset.columns:
        raise ValueError("the profile and the dataset have different columns")
    if profile.n_rows != dataset.n_rows:
        raise ValueError(
            f"the profile counts {profile.n_rows} rows, the dataset {dataset.n_rows}"
        )
    events = Dispatcher(hooks)
    total = len(plan.entries)
    started_at = datetime.now(UTC)
    clock = time.monotonic()
    results: dict[str, CheckResult] = {}
    with Meter(memory_sample_seconds) as meter:
        for position, entry in enumerate(
            (entry for level in plan.levels for entry in level), start=1
        ):
            remaining = (
                None
                if limits.run_budget_seconds is None
                else limits.run_budget_seconds - (time.monotonic() - clock)
            )
            result = _execute_entry(
                entry,
                dataset,
                profile,
                results,
                limits,
                remaining,
                meter,
                started=partial(events.emit, CheckStarted(entry.id, position, total)),
            )
            results[entry.id] = result
            for note in result.notes:
                events.emit(NoteRecorded(entry.id, note, position, total))
            events.emit(CheckFinished(entry.id, result, position, total))
    return Run.build(
        results=results.values(),
        run_id=uuid4(),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        dataset=_dataset_manifest(dataset),
        plan=_plan_manifest(plan, dsbf_version),
        profile=profile,
    )


def _execute_entry(
    entry: PlanEntry,
    dataset: Dataset,
    profile: DatasetProfile,
    results: dict[str, CheckResult],
    limits: Limits,
    budget_left: float | None,
    meter: Meter,
    started: Callable[[], None],
) -> CheckResult:
    expected = entry.expected
    if expected.status is Expected.UNAVAILABLE:
        return _without_running(
            entry,
            CheckStatus.UNAVAILABLE,
            Note(
                code="extra_missing",
                message=expected.reason or f"{entry.id} cannot run here.",
                hint=expected.hint,
            ),
        )
    blockers = {
        needed: results[needed].status
        for needed in entry.requires
        if results[needed].status not in _SATISFIES_DEPENDENTS
    }
    if blockers:
        listed = ", ".join(f"{name!r} ({status})" for name, status in blockers.items())
        return _without_running(
            entry,
            CheckStatus.SKIPPED,
            Note(
                code="dependency_failed",
                message=f"Requires {listed}, which did not complete.",
                hint="Fix the check it requires, or exclude both.",
                blocked_by=tuple(blockers),
            ),
        )
    if budget_left is not None and budget_left <= 0:
        return _without_running(
            entry,
            CheckStatus.SKIPPED,
            Note(
                code="run_budget_exhausted",
                message=(
                    f"The run budget of {limits.run_budget_seconds:g}s was used up "
                    "before this check started."
                ),
                hint=_BUDGET_HINT,
            ),
        )

    limit = limits.check_timeout_seconds
    by_budget = budget_left is not None and (limit is None or budget_left < limit)
    if by_budget:
        limit = budget_left
    started()
    with meter.window() as window:
        attempt = attempt_check(
            entry,
            dataset,
            profile,
            results,
            limit=limit,
            grace=limits.cancel_grace_seconds,
        )
    if attempt.timed_out:
        assert limit is not None
        return _timeout_result(entry, window, limits, limit, by_budget, attempt.stopped)
    if attempt.error is not None:
        return _error_result(entry, attempt.error, window)
    assert attempt.result is not None
    return attempt.result.model_copy(update={"timing": window.timing})


def _without_running(entry: PlanEntry, status: CheckStatus, note: Note) -> CheckResult:
    return CheckResult(
        check_id=entry.id,
        check_version=entry.check.version,
        status=status,
        notes=(note,),
        timing=Timing(wall_seconds=0.0),
        params=entry.params.model_dump(mode="json"),
    )


def _error_result(entry: PlanEntry, error: Exception, window: Window) -> CheckResult:
    """Turn an exception into an ``error`` result; the findings so far are dropped."""
    what = str(error) or type(error).__name__
    if isinstance(error, CheckContractError):
        hint = (
            f"{entry.id} used its context in a way its declaration does not allow, "
            "which is a bug in the check; report it to the check's author."
        )
    else:
        hint = (
            f"Run {entry.id} on its own to reproduce it, and report the traceback "
            "in this note to the check's author if the data is valid."
        )
    note = Note(
        code="check_failed",
        message=f"Raised {type(error).__name__}: {what}",
        severity=Severity.ERROR,
        hint=hint,
        exception_type=type(error).__name__,
        detail="".join(traceback.format_exception(error)),
    )
    return CheckResult(
        check_id=entry.id,
        check_version=entry.check.version,
        status=CheckStatus.ERROR,
        notes=(note,),
        timing=window.timing,
        params=entry.params.model_dump(mode="json"),
    )


def _timeout_result(
    entry: PlanEntry,
    window: Window,
    limits: Limits,
    limit: float,
    by_budget: bool,
    stopped: bool,
) -> CheckResult:
    """A check ran out of time; the findings so far are dropped."""
    if by_budget:
        code = "run_budget_exhausted"
        message = (
            f"The run budget of {limits.run_budget_seconds:g}s was used up while "
            f"{entry.id} was running."
        )
        hint = _BUDGET_HINT
    else:
        code = "check_timeout"
        message = f"Did not finish within {limit:g}s."
        hint = (
            f"Raise the check timeout, run {entry.id} on a sample, or exclude it "
            "from the plan."
        )
    detail = (
        "The check noticed the cancellation and stopped."
        if stopped
        else "The check was still running when the executor stopped waiting; it "
        "cannot be interrupted and keeps using CPU and memory until it returns, "
        "which can inflate the timings of the checks after it."
    )
    note = Note(
        code=code,
        message=message,
        severity=Severity.ERROR,
        hint=hint,
        detail=detail,
    )
    return CheckResult(
        check_id=entry.id,
        check_version=entry.check.version,
        status=CheckStatus.TIMEOUT,
        notes=(note,),
        timing=window.timing,
        params=entry.params.model_dump(mode="json"),
    )


def _dataset_manifest(dataset: Dataset) -> DatasetManifest:
    return DatasetManifest(
        name=dataset.name,
        source=dataset.source,
        fingerprint=dataset.fingerprint,
        n_rows=dataset.source_rows,
        n_columns=dataset.n_columns,
        sampling=dataset.sampling,
    )


def _plan_manifest(plan: Plan, dsbf_version: str) -> PlanManifest:
    return PlanManifest(
        dsbf_version=dsbf_version,
        depth=plan.selection.depth.value,
        checks=tuple(
            PlannedCheck(
                check_id=entry.id,
                check_version=entry.check.version,
                params=entry.params.model_dump(mode="json"),
                depends_on=(*entry.requires, *entry.after),
            )
            for entry in plan.entries
        ),
    )
