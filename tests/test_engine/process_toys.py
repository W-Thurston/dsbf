"""Checks that run in a worker process, so they live at the top level of a module.

A worker imports a check by name, which a function defined inside a test cannot be.
"""

import os
import time

from pydantic import Field

from dsbf.engine import CheckContext, SemanticTypes, check
from dsbf.model import (
    CheckParams,
    Confidence,
    Dimension,
    Evidence,
    Severity,
    Subject,
)


class PathParams(CheckParams):
    path: str = Field("", description="A file the check writes its process id to.")


def _report(ctx: CheckContext, code: str, **metrics: float) -> None:
    ctx.finding(
        code=code,
        subject=Subject.dataset(),
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        title=code,
        message=code,
        evidence=Evidence(
            metrics=metrics,
            n_used=ctx.profile.n_rows,
            n_total=ctx.profile.n_rows,
            method="python_loop",
        ),
    )


@check(
    id="py_total",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.DISTRIBUTION],
    runs_in="process",
)
def py_total(ctx: CheckContext, params: CheckParams):
    """Sum every numeric column in a Python loop."""
    total = 0.0
    for column in ctx.columns:
        for value in ctx.values(column).tolist():
            total += value
    _report(ctx, "total", total=total)
    return None


@check(
    id="py_pid",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.VALIDITY],
    runs_in="process",
)
def py_pid(ctx: CheckContext, params: CheckParams):
    """Report the process id the check runs in."""
    _report(ctx, "pid", pid=float(os.getpid()))
    return None


@check(
    id="py_rows",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.VALIDITY],
    runs_in="process",
)
def py_rows(ctx: CheckContext, params: CheckParams):
    """Report how many rows the worker saw."""
    _report(ctx, "rows", rows=float(ctx.profile.n_rows))
    return None


@check(
    id="py_sleeper",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.VALIDITY],
    params=PathParams,
    runs_in="process",
)
def py_sleeper(ctx: CheckContext, params: PathParams):
    """Record the process id, then sleep far longer than any test waits."""
    if params.path:
        with open(params.path, "w") as handle:
            handle.write(str(os.getpid()))
    time.sleep(60)
    return None


@check(
    id="py_busy",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.VALIDITY],
    runs_in="process",
)
def py_busy(ctx: CheckContext, params: CheckParams):
    """Burn 0.3 s of CPU in Python."""
    end = time.process_time() + 0.3
    while time.process_time() < end:
        pass
    return None


@check(
    id="py_raises",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.VALIDITY],
    runs_in="process",
)
def py_raises(ctx: CheckContext, params: CheckParams):
    """Fail the way a bug does."""
    _report(ctx, "before_failing", n=1)
    raise ValueError("boom in worker")


@check(
    id="py_exits",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.VALIDITY],
    runs_in="process",
)
def py_exits(ctx: CheckContext, params: CheckParams):
    """End the process without a result, as a crash in compiled code would."""
    os._exit(7)


@check(
    id="py_exits_cleanly",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.VALIDITY],
    runs_in="process",
)
def py_exits_cleanly(ctx: CheckContext, params: CheckParams):
    """Call sys.exit, which a worker cannot hand back to the caller."""
    raise SystemExit(0)


@check(
    id="py_downstream",
    version="1.0.0",
    kind="synthesis",
    applies_to=None,
    dimensions=[Dimension.USABILITY],
    after=["py_total"],
    runs_in="process",
)
def py_downstream(ctx: CheckContext, params: CheckParams):
    """Read the result of a check that ran before it, in another process."""
    upstream = ctx.result("py_total")
    total = -1.0
    if upstream:
        total = float(upstream.findings[0].evidence.metrics["total"])  # type: ignore[arg-type]
    _report(ctx, "seen_total", total=total)
    return None
