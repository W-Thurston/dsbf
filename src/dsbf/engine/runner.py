# dsbf/engine/runner.py
"""Run one check against a context and turn what it does into a CheckResult."""

import time
from datetime import UTC, datetime

from dsbf.engine.check import Check
from dsbf.engine.context import CheckContext, CheckOutput
from dsbf.engine.errors import CheckContractError
from dsbf.model import CheckResult, CheckStatus, Note, Timing


def run_check(check: Check, context: CheckContext) -> CheckResult:
    """Call the check and assemble its result.

    A check with no eligible column is ``not_applicable`` and isn't called. Any
    exception from the check propagates: turning it into an ``error`` result is the
    executor's job, so this stays the one place a result is built from a run.

    Raises:
        CheckContractError: ``context`` belongs to another check, or the check
            returned something other than ``ctx.output(...)`` or ``None``.
    """
    spec = check.spec
    if context.spec != spec:
        raise CheckContractError(
            f"context was built for {context.spec.id}, not {spec.id}"
        )
    started_at = datetime.now(UTC)
    clock = time.perf_counter()
    notes: tuple[Note, ...] = ()
    output: CheckOutput | None = None
    status = CheckStatus.OK
    if spec.applies_to is not None and not context.columns:
        status = CheckStatus.NOT_APPLICABLE
        notes = (
            Note(
                code="no_eligible_columns",
                message=(
                    f"No column is of a type {spec.id} applies to "
                    f"({spec.applies_to.describe()})."
                ),
            ),
        )
    else:
        output = check(context, context.params)
        if output is not None and not isinstance(output, CheckOutput):
            raise CheckContractError(
                f"{spec.id} returned {type(output).__name__}; return ctx.output(...) "
                "or None"
            )
        notes = context.notes
    return CheckResult(
        check_id=spec.id,
        check_version=spec.version,
        status=status,
        notes=notes,
        findings=context.findings,
        output=output.data if output else None,
        timing=Timing(wall_seconds=time.perf_counter() - clock, started_at=started_at),
        params=context.params.model_dump(mode="json"),
    )
