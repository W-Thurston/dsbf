# dsbf/checks/datetime_features.py
"""Datetime columns, and the features worth extracting from them for a model."""

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

# Features that encode a cycle, whose ends are neighbours (December and January).
CYCLIC = ("month_sin_cos", "day_of_week_sin_cos", "hour_sin_cos")


class DatetimeFeaturesParams(CheckParams):
    pass


class ColumnFeatures(Contract):
    features: tuple[str, ...] = Field(
        description="Features worth extracting, in the order they are listed."
    )
    range_days: float = Field(ge=0)
    time_of_day: bool
    distinct_years: int = Field(ge=1)
    distinct_months: int = Field(ge=1, le=12)
    distinct_weekdays: int = Field(ge=1, le=7)
    distinct_hours: int = Field(ge=1, le=24)


class DatetimeFeaturesOutput(CheckOutputModel):
    columns: dict[str, ColumnFeatures] = Field(
        description="Every datetime column with two or more instants, by name."
    )


@check(
    id="datetime_features",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("datetime"),
    dimensions=[Dimension.USABILITY],
    params=DatetimeFeaturesParams,
    output=DatetimeFeaturesOutput,
    requires=Requires(profile=["counts", "datetime"]),
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["datetime", "feature-engineering", "ml-readiness"],
)
def datetime_features(ctx: CheckContext, params: DatetimeFeaturesParams) -> CheckOutput:
    """List the calendar features that vary in each datetime column.

    A model cannot use a raw timestamp; its signal is in parts of it. A part is
    listed only if the column varies in it: no hour features for midnight dates, no
    month features inside one month.
    """
    columns: dict[str, ColumnFeatures] = {}
    for column in ctx.profile.columns:
        assert isinstance(column, DatetimeProfile)
        found = _describe(column)
        if found is None:
            continue
        columns[column.name] = found
        _report(ctx, column, found)
    if not columns:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_datetime_column",
            "No datetime column has two or more instants.",
            severity=Severity.INFO,
        )
    return ctx.output(DatetimeFeaturesOutput(columns=columns))


def _describe(column: DatetimeProfile) -> ColumnFeatures | None:
    if (
        column.min is None
        or column.max is None
        or column.distinct_years is None
        or column.distinct_months is None
        or column.distinct_weekdays is None
        or column.distinct_hours is None
        or column.time_of_day is None
        or column.counts.distinct < 2  # noqa: PLR2004
    ):
        return None
    days = (column.max - column.min).total_seconds() / _SECONDS_PER_DAY
    features = ["days_since_epoch"]
    if column.distinct_years > 1:
        features.append("year")
    if column.distinct_months > 1:
        features += ["month", "quarter", "week_of_year", "month_sin_cos"]
    if column.distinct_weekdays > 1:
        features += ["day_of_week", "is_weekend", "day_of_week_sin_cos"]
    if days > 28:  # noqa: PLR2004
        features += ["day_of_month", "is_month_start", "is_month_end"]
    if column.time_of_day and column.distinct_hours > 1:
        features += ["hour", "hour_sin_cos"]
    return ColumnFeatures(
        features=tuple(features),
        range_days=days,
        time_of_day=column.time_of_day,
        distinct_years=column.distinct_years,
        distinct_months=column.distinct_months,
        distinct_weekdays=column.distinct_weekdays,
        distinct_hours=column.distinct_hours,
    )


def _report(ctx: CheckContext, column: DatetimeProfile, found: ColumnFeatures) -> None:
    cyclic = [f for f in found.features if f in CYCLIC]
    ctx.finding(
        code="extractable_features",
        subject=Subject.column(column.name),
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        title=f"{len(found.features)} feature(s) to extract",
        message=(
            f"{column.name!r} spans {found.range_days:,.0f} days "
            f"({column.min:%Y-%m-%d} to {column.max:%Y-%m-%d}) in "
            f"{column.counts.distinct:,} distinct values. A model cannot use a raw "
            f"timestamp; the parts that vary here are worth extracting: "
            f"{', '.join(found.features)}."
        ),
        evidence=Evidence(
            metrics={
                "features": found.features,
                "n_features": len(found.features),
                "n_cyclic": len(cyclic),
                "cyclic_features": tuple(cyclic),
                "range_days": found.range_days,
                "time_of_day": found.time_of_day,
                "distinct_years": found.distinct_years,
                "distinct_months": found.distinct_months,
                "distinct_weekdays": found.distinct_weekdays,
                "distinct_hours": found.distinct_hours,
            },
            n_used=column.counts.non_null,
            n_total=column.counts.rows,
            method="calendar_variation",
        ),
    )
