# dsbf/checks/temporal_gaps.py
"""Datetime columns with missing periods or no regular spacing."""

from datetime import datetime

from pydantic import Field

from dsbf.engine import (
    CheckContext,
    CheckOutput,
    CheckOutputModel,
    Requires,
    SemanticTypes,
    check,
)
from dsbf.model import (
    CheckParams,
    CheckStatus,
    Confidence,
    Contract,
    Cost,
    DatetimeProfile,
    Dimension,
    Evidence,
    Severity,
    Subject,
)

_SECONDS_PER_DAY = 86_400


class TemporalGapsParams(CheckParams):
    min_instants: int = Field(
        10,
        ge=3,
        description="Distinct values a column needs before its spacing is judged.",
    )
    warn_gaps: int = Field(
        5,
        ge=1,
        description="Gaps from which the finding is a warning rather than information.",
    )


class ColumnGaps(Contract):
    distinct: int = Field(ge=0, description="Distinct values of the column.")
    first: datetime | None
    last: datetime | None
    frequency: str | None = Field(
        description="The dominant spacing, as an offset alias; None when no step "
        "covers half of the intervals."
    )
    n_gaps: int | None = Field(
        ge=0,
        description="Steps between distinct values longer than 1.5 times the "
        "dominant step; None without a dominant step.",
    )
    largest_gap_seconds: float | None = Field(ge=0)


class TemporalGapsOutput(CheckOutputModel):
    columns: dict[str, ColumnGaps] = Field(
        description="Every datetime column that was judged, by name."
    )
    skipped: dict[str, str] = Field(description="Columns not judged, with the reason.")


@check(
    id="temporal_gaps",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("datetime"),
    dimensions=[Dimension.TEMPORAL],
    params=TemporalGapsParams,
    output=TemporalGapsOutput,
    requires=Requires(profile=["counts", "datetime"]),
    cost=Cost.CHEAP,
    codes=["gaps", "irregular_spacing"],
    packs=["core"],
    tags=["datetime", "gaps", "time-series"],
)
def temporal_gaps(ctx: CheckContext, params: TemporalGapsParams) -> CheckOutput:
    """Flag missing periods and irregular spacing in datetime columns.

    The profile took the dominant step between consecutive distinct values (when it
    is at least half of them) and counted the steps longer than 1.5 times it. A
    column with no dominant step is irregular: its gaps are not periods missing from
    a regular series.
    """
    columns: dict[str, ColumnGaps] = {}
    skipped: dict[str, str] = {}
    for column in ctx.profile.columns:
        assert isinstance(column, DatetimeProfile)
        if column.counts.distinct < params.min_instants:
            skipped[column.name] = f"fewer than {params.min_instants} distinct values"
            continue
        columns[column.name] = ColumnGaps(
            distinct=column.counts.distinct,
            first=column.min,
            last=column.max,
            frequency=column.inferred_frequency,
            n_gaps=column.n_gaps,
            largest_gap_seconds=column.largest_gap_seconds,
        )
        if column.inferred_frequency is None:
            _irregular(ctx, column)
        elif column.n_gaps:
            _gaps(ctx, column, params)
    if not columns:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_datetime_column",
            f"No datetime column has {params.min_instants} or more distinct values.",
            severity=Severity.INFO,
        )
    return ctx.output(TemporalGapsOutput(columns=columns, skipped=skipped))


def _metrics(column: DatetimeProfile) -> dict[str, object]:
    largest = column.largest_gap_seconds
    return {
        "frequency": column.inferred_frequency,
        "n_gaps": column.n_gaps,
        "largest_gap_seconds": largest,
        "largest_gap_days": None if largest is None else largest / _SECONDS_PER_DAY,
        "first": column.min,
        "last": column.max,
        "distinct": column.counts.distinct,
    }


def _evidence(column: DatetimeProfile) -> Evidence:
    return Evidence(
        metrics=_metrics(column),  # type: ignore[arg-type]
        n_used=column.counts.non_null,
        n_total=column.counts.rows,
        method="dominant_step_gaps",
    )


def _gaps(
    ctx: CheckContext, column: DatetimeProfile, params: TemporalGapsParams
) -> None:
    n_gaps = column.n_gaps or 0
    largest = (column.largest_gap_seconds or 0) / _SECONDS_PER_DAY
    ctx.finding(
        code="gaps",
        subject=Subject.column(column.name),
        severity=Severity.WARN if n_gaps >= params.warn_gaps else Severity.INFO,
        confidence=Confidence.HIGH,
        title=f"{n_gaps:,} gap(s) in a {column.inferred_frequency} series",
        message=(
            f"{column.name!r} is spaced {column.inferred_frequency!r} for most of "
            f"its {column.counts.distinct:,} distinct values, but {n_gaps:,} steps "
            f"are more than 1.5 times that long, the longest {largest:,.1f} days. "
            "Periods missing from a series bias autocorrelation and decomposition, "
            "let a forward fill carry stale values across them, and make a rolling "
            "window span unlike periods. Decide whether each gap is a collection "
            "failure or an expected break (weekends, holidays)."
        ),
        evidence=_evidence(column),
    )


def _irregular(ctx: CheckContext, column: DatetimeProfile) -> None:
    largest = (column.largest_gap_seconds or 0) / _SECONDS_PER_DAY
    ctx.finding(
        code="irregular_spacing",
        subject=Subject.column(column.name),
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        title="No regular spacing",
        message=(
            f"No step between consecutive distinct values of {column.name!r} covers "
            f"half of them, so it is not a regular series (longest step "
            f"{largest:,.1f} days). That is expected for event data. A time-series "
            "check needs a frequency: set one to resample to it."
        ),
        evidence=_evidence(column),
    )
