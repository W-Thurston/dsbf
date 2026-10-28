# dsbf/engine/executor.py
"""Execute a plan: every check ends in an explicit status, and the run completes."""

import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from dsbf.data import Dataset
from dsbf.engine.artifacts import make_ref
from dsbf.engine.cache import CacheSession, ResultCache
from dsbf.engine.capture import capture_warnings
from dsbf.engine.events import (
    CheckFinished,
    CheckStarted,
    Dispatcher,
    Hook,
    NoteRecorded,
)
from dsbf.engine.expectation import Expected
from dsbf.engine.extras import preload_extras
from dsbf.engine.limits import CheckOverride, Limits
from dsbf.engine.meter import DEFAULT_SAMPLE_SECONDS, Meter, Window
from dsbf.engine.plan import Plan, PlanEntry
from dsbf.engine.reference import Reference
from dsbf.engine.scheduler import Launch, Scheduler, default_workers
from dsbf.engine.worker import Attempt, Failure
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
from dsbf.model.artifact import ArtifactRef, Payload

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
    workers: int | None = None,
    cache: ResultCache | None = None,
    reference: Reference | None = None,
    check_overrides: Mapping[str, CheckOverride] | None = None,
    dataset_key: str | None = None,
) -> Run:
    """Run every check in ``plan`` against ``dataset`` and return the ``Run``.

    Checks run level by level; the checks of a level, which depend on nothing in
    each other, run at the same time, up to ``workers`` at once (``None`` is one per
    core, at most 8). A check runs on a thread of its own, or in a worker process if
    its spec says ``runs_in="process"``. Nothing a check does stops the run: an
    exception becomes an ``error`` result with a note, a check that outlasts
    ``limits.check_timeout_seconds`` is ``timeout``, a check whose hard requirement
    did not complete is ``skipped`` (a soft ``after`` upstream that failed changes
    nothing), and a check missing an optional dependency is ``unavailable``. Once
    ``limits.run_budget_seconds`` is used up the running checks time out and the
    rest are ``skipped``. Every planned check has a result, in plan order, and
    ``Run.health`` says why any did not complete. ``KeyboardInterrupt`` and
    ``SystemExit`` are not caught. Apart from ``timing``, the run does not depend
    on ``workers``. ``check_overrides`` gives a check its own timeout or ``runs_in``,
    by id, in place of the limit and the spec's.

    A Python warning a check raises does not reach ``warnings.showwarning``: the
    check's result gets one ``python_warning`` note for each category and message,
    with a ``count``, whatever status it ends with. For that, while the run lasts,
    the process's warning filters show a warning every time it is raised instead of
    once (``ignore`` and ``error`` filters hold) and are restored afterwards (see
    ``dsbf.engine.capture``). The change is process-wide: the caller's other
    threads see each repeat of their own warnings meanwhile, and a check or library
    that calls ``warnings.catch_warnings`` changes the filters for every thread.
    A warning from a thread a check starts itself is not routed to the check.

    A thread cannot be killed: a timed-out check on a thread that never calls back
    into its context keeps running in the background until it returns (see
    ``CheckContext.checkpoint``). A worker process is terminated instead. Each
    result's ``timing`` is approximate: the wall time is the executor's. On a thread
    the CPU time and memory are the whole process's, found by sampling every
    ``memory_sample_seconds`` (``None`` skips memory), and ``shared_with`` counts the
    checks that overlapped. In a worker they are the worker's own.

    Each ``hook`` receives a ``CheckStarted``, a ``NoteRecorded`` for every note and
    a ``CheckFinished``, in that order for each check, from the calling thread, so
    a hook needs no locking. Events of different checks interleave in the order the
    checks end, which is plan order with one worker. A check that is not called has
    no ``CheckStarted``. A hook that raises is logged and dropped.

    With a ``cache``, a check whose inputs are unchanged since it last completed is
    not run: its stored result is returned, with ``timing.cached`` set, no
    ``CheckStarted`` and its notes and ``CheckFinished`` as usual. The inputs are the
    data, the profile, the check's id, version and parameters, and the results it may
    read, so changing one check's parameters re-runs it and the checks that read its
    changed result. Only ``ok`` results are stored, and a hit is served even after the
    run budget is used up. A cache that fails is logged once and ignored.

    ``profile`` must be the profile of ``dataset``. ``reference`` is the dataset the
    checks that require one compare with; no other check can read it, and the
    cache key of one that does covers it.

    Raises:
        ValueError: ``profile`` is not the profile of ``dataset``,
            ``memory_sample_seconds`` is not positive, or ``workers`` is below 1.
    """
    if [column.name for column in profile.columns] != dataset.columns:
        raise ValueError("the profile and the dataset have different columns")
    if profile.n_rows != dataset.n_rows:
        raise ValueError(
            f"the profile counts {profile.n_rows} rows, the dataset {dataset.n_rows}"
        )
    if workers is None:
        workers = default_workers()
    elif workers < 1:
        raise ValueError(f"workers must be at least 1, not {workers!r}")
    preload_extras(
        extra
        for entry in plan.entries
        if entry.expected.status is not Expected.UNAVAILABLE
        for extra in entry.check.spec.extras
    )
    overrides = check_overrides or {}
    events = Dispatcher(hooks)
    total = len(plan.entries)
    positions = {
        entry.id: position
        for position, entry in enumerate(
            (entry for level in plan.levels for entry in level), start=1
        )
    }
    session = (
        None if cache is None else CacheSession(cache, dataset, profile, reference)
    )
    started_at = datetime.now(UTC)
    clock = time.monotonic()
    results: dict[str, CheckResult] = {}

    def budget_left() -> float | None:
        if limits.run_budget_seconds is None:
            return None
        return limits.run_budget_seconds - (time.monotonic() - clock)

    with (
        capture_warnings(),
        Meter(memory_sample_seconds) as meter,
        Scheduler(
            workers=workers,
            dataset=dataset,
            profile=profile,
            grace=limits.cancel_grace_seconds,
            meter=meter,
            reference=reference,
        ) as scheduler,
    ):
        for level in plan.levels:
            done: dict[str, CheckResult] = {}
            for entry, result in scheduler.run_level(
                level,
                results,
                prepare=lambda entry: _prepare(
                    entry, results, limits, budget_left(), session, overrides
                ),
                started=lambda entry: events.emit(
                    CheckStarted(entry.id, positions[entry.id], total)
                ),
                finish=lambda launch, attempt, window: _finish(
                    launch, attempt, window, limits, session
                ),
            ):
                done[entry.id] = result
                position = positions[entry.id]
                for note in result.notes:
                    events.emit(NoteRecorded(entry.id, note, position, total))
                events.emit(CheckFinished(entry.id, result, position, total))
            for entry in level:  # plan order, whatever order they ended in
                results[entry.id] = done[entry.id]
    run = Run.build(
        results=results.values(),
        run_id=uuid4(),
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=time.monotonic() - clock),
        dataset=_dataset_manifest(dataset, dataset_key),
        plan=_plan_manifest(plan, dsbf_version),
        profile=profile,
        notes=(*plan.notes, *_cache_notes(session)),
    )
    run._frame = dataset.frame  # noqa: SLF001 (the run holds what it analysed)
    return run


def _prepare(
    entry: PlanEntry,
    results: dict[str, CheckResult],
    limits: Limits,
    budget_left: float | None,
    session: CacheSession | None,
    overrides: Mapping[str, CheckOverride],
) -> CheckResult | Launch:
    """The result of a check that will not run, or how to run the one that will."""
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
    if session is not None:
        looked_up = time.perf_counter()
        started_at = datetime.now(UTC)
        stored = session.lookup(entry, results)
        if stored is not None:
            return stored.model_copy(
                update={
                    "timing": Timing(
                        wall_seconds=time.perf_counter() - looked_up,
                        started_at=started_at,
                        cached=True,
                    )
                }
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
    override = overrides.get(entry.id, CheckOverride())
    limit = override.timeout_seconds or limits.check_timeout_seconds
    by_budget = budget_left is not None and (limit is None or budget_left < limit)
    if by_budget:
        limit = budget_left
    return Launch(entry, limit, by_budget, override.runs_in or entry.check.spec.runs_in)


def _finish(
    launch: Launch,
    attempt: Attempt,
    window: Window,
    limits: Limits,
    session: CacheSession | None,
) -> CheckResult:
    """The result of a check that ran, from how its turn ended."""
    entry = launch.entry
    if attempt.timed_out:
        assert launch.limit is not None
        result = _timeout_result(entry, window, limits, launch, attempt)
    elif attempt.failure is not None:
        result = _error_result(entry, attempt.failure, window)
    else:
        assert attempt.result is not None
        result = attempt.result.model_copy(update={"timing": window.timing})
    if attempt.notes:
        result = result.model_copy(update={"notes": (*result.notes, *attempt.notes)})
    if session is not None and attempt.result is not None:
        session.store(entry, result)
    return result


def _without_running(entry: PlanEntry, status: CheckStatus, note: Note) -> CheckResult:
    return CheckResult(
        check_id=entry.id,
        check_version=entry.check.version,
        status=status,
        notes=(note,),
        timing=Timing(wall_seconds=0.0),
        params=entry.params.model_dump(mode="json"),
    )


def _error_result(entry: PlanEntry, failure: Failure, window: Window) -> CheckResult:
    """Turn an exception into an ``error`` result; the findings so far are dropped."""
    if failure.type_name == "WorkerDied":
        hint = (
            f"{entry.id} ended its worker process without a result; look for a crash "
            "in a compiled extension, or for the check calling os._exit or running "
            "out of memory."
        )
        code = "worker_died"
        message = failure.message
    else:
        code = "check_failed"
        message = f"Raised {failure.type_name}: {failure.message}"
        if failure.contract:
            hint = (
                f"{entry.id} used its context in a way its declaration does not "
                "allow, which is a bug in the check; report it to the check's author."
            )
        else:
            hint = (
                f"Run {entry.id} on its own to reproduce it, and report the traceback "
                f"(the {entry.id}/traceback artifact) to the check's author if the "
                "data is valid."
            )
    note = Note(
        code=code,
        message=message,
        severity=Severity.ERROR,
        hint=hint,
        exception_type=None if code == "worker_died" else failure.type_name,
        detail=failure.summary or None,
    )
    artifacts: list[tuple[ArtifactRef, Payload]] = []
    if failure.detail:
        artifacts.append(
            make_ref(
                f"{entry.id}/traceback",
                failure.detail,
                "The formatted traceback of the exception.",
            )
        )
    result = CheckResult(
        check_id=entry.id,
        check_version=entry.check.version,
        status=CheckStatus.ERROR,
        notes=(note,),
        artifacts=tuple(ref for ref, _ in artifacts),
        timing=window.timing,
        params=entry.params.model_dump(mode="json"),
    )
    return result.with_payloads({ref.ref: data for ref, data in artifacts})


def _timeout_result(
    entry: PlanEntry,
    window: Window,
    limits: Limits,
    launch: Launch,
    attempt: Attempt,
) -> CheckResult:
    """A check ran out of time; the findings so far are dropped."""
    limit = launch.limit
    assert limit is not None
    if launch.by_budget:
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
    if attempt.killed:
        detail = "The check ran in a worker process, which was killed."
    elif attempt.stopped:
        detail = "The check noticed the cancellation and stopped."
    else:
        detail = (
            "The check was still running when the executor stopped waiting; it "
            "cannot be interrupted and keeps using CPU and memory until it returns, "
            "which can inflate the timings of the checks after it. Declare it "
            'runs_in="process" to have it killed.'
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


def _cache_notes(session: CacheSession | None) -> tuple[Note, ...]:
    if session is None or session.failure is None:
        return ()
    return (
        Note(
            code="cache_unavailable",
            message="The result cache failed, so the checks after that ran uncached.",
            severity=Severity.WARN,
            hint="Check that the cache directory is writable, or run without a cache.",
            detail=session.failure,
        ),
    )


def _dataset_manifest(dataset: Dataset, key: str | None) -> DatasetManifest:
    return DatasetManifest(
        name=dataset.name,
        source=dataset.source,
        key=key or dataset.source.location or "",
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
