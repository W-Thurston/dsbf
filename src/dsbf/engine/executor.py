# dsbf/engine/executor.py
"""Execute a plan: every check ends in an explicit status, and the run completes."""

import time
import traceback
from datetime import UTC, datetime
from uuid import uuid4

from dsbf.data import Dataset
from dsbf.engine.context import CheckContext
from dsbf.engine.errors import CheckContractError
from dsbf.engine.expectation import Expected
from dsbf.engine.plan import Plan, PlanEntry
from dsbf.engine.runner import run_check
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


def execute(
    plan: Plan,
    dataset: Dataset,
    profile: DatasetProfile,
    *,
    dsbf_version: str,
) -> Run:
    """Run every check in ``plan`` against ``dataset`` and return the ``Run``.

    Checks run one after another, level by level. Nothing a check does stops the run:
    an exception becomes an ``error`` result with a note, a check whose hard
    requirement did not complete is ``skipped`` (a soft ``after`` upstream that failed
    changes nothing), and a check missing an optional dependency is ``unavailable``.
    Every planned check has a result, and ``Run.health`` says why any did not
    complete. ``KeyboardInterrupt`` and ``SystemExit`` are not caught.

    ``profile`` must be the profile of ``dataset``.

    Raises:
        ValueError: ``profile`` is not the profile of ``dataset``.
    """
    if [column.name for column in profile.columns] != dataset.columns:
        raise ValueError("the profile and the dataset have different columns")
    if profile.n_rows != dataset.n_rows:
        raise ValueError(
            f"the profile counts {profile.n_rows} rows, the dataset {dataset.n_rows}"
        )
    started_at = datetime.now(UTC)
    results: dict[str, CheckResult] = {}
    for level in plan.levels:
        for entry in level:
            results[entry.id] = _execute_entry(entry, dataset, profile, results)
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

    started_at = datetime.now(UTC)
    clock = time.perf_counter()
    try:
        context = CheckContext(
            entry.check.spec,
            dataset,
            profile,
            entry.params,
            upstream={
                name: results[name]
                for name in (*entry.requires, *entry.after)
                if name in results
            },
        )
        return run_check(entry.check, context)
    except Exception as error:
        return _error_result(entry, error, started_at, time.perf_counter() - clock)


def _without_running(entry: PlanEntry, status: CheckStatus, note: Note) -> CheckResult:
    return CheckResult(
        check_id=entry.id,
        check_version=entry.check.version,
        status=status,
        notes=(note,),
        timing=Timing(wall_seconds=0.0),
        params=entry.params.model_dump(mode="json"),
    )


def _error_result(
    entry: PlanEntry, error: Exception, started_at: datetime, wall_seconds: float
) -> CheckResult:
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
        timing=Timing(wall_seconds=wall_seconds, started_at=started_at),
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
