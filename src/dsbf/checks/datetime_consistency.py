# dsbf/checks/datetime_consistency.py
"""Datetime columns in which some values do not read as a date or time."""

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


class DatetimeConsistencyParams(CheckParams):
    warn_share: float = Field(
        0.05,
        gt=0,
        le=1,
        description="Share of a column's non-null values that do not read, from "
        "which the finding is a warning rather than information.",
    )


class ColumnConsistency(Contract):
    non_null: int = Field(ge=0)
    unparsed: int = Field(ge=0, description="Non-null values that did not read.")
    share: float = Field(ge=0, le=1, description="`unparsed` over `non_null`.")
    format: str | None = Field(description="The format the text was read with.")
    epoch_unit: str | None = Field(description="The unit of Unix timestamps.")


class DatetimeConsistencyOutput(CheckOutputModel):
    columns: dict[str, ColumnConsistency] = Field(
        description="Every datetime column, by name."
    )


@check(
    id="datetime_consistency",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("datetime"),
    dimensions=[Dimension.VALIDITY],
    params=DatetimeConsistencyParams,
    output=DatetimeConsistencyOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.CHEAP,
    codes=["unparsed_values"],
    packs=["core"],
    tags=["datetime", "validation"],
)
def datetime_consistency(
    ctx: CheckContext, params: DatetimeConsistencyParams
) -> CheckOutput:
    """Report the values of a datetime column that its inferred format cannot read.

    The profile typed a text column datetime only if one format reads over 90% of its
    values, and counted the others as unparsed, so a share of 10% or more cannot occur
    here; a column that is mostly unreadable is not a datetime column. A date or
    datetime column has none.
    """
    columns: dict[str, ColumnConsistency] = {}
    for column in ctx.profile.columns:
        assert isinstance(column, DatetimeProfile)
        non_null = column.counts.non_null
        share = column.unparsed / non_null if non_null else 0.0
        columns[column.name] = ColumnConsistency(
            non_null=non_null,
            unparsed=column.unparsed,
            share=share,
            format=column.format,
            epoch_unit=column.epoch_unit,
        )
        if column.unparsed:
            _report(ctx, column, share, params)
    if not columns:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_datetime_column",
            "The data has no datetime column.",
            severity=Severity.INFO,
        )
    return ctx.output(DatetimeConsistencyOutput(columns=columns))


def _report(
    ctx: CheckContext,
    column: DatetimeProfile,
    share: float,
    params: DatetimeConsistencyParams,
) -> None:
    read_as = (
        f"the format {column.format!r}"
        if column.format
        else f"Unix timestamps in {column.epoch_unit}"
        if column.epoch_unit
        else "a date or time"
    )
    ctx.finding(
        code="unparsed_values",
        subject=Subject.column(column.name),
        severity=Severity.WARN if share >= params.warn_share else Severity.INFO,
        confidence=Confidence.HIGH,
        title=f"{column.unparsed:,} values do not read as dates ({share:.1%})",
        message=(
            f"{column.unparsed:,} of the {column.counts.non_null:,} non-null values "
            f"of {column.name!r} ({share:.1%}) do not read as {read_as}, though the "
            "rest do. They are mixed formats, placeholders or damaged values; each "
            "becomes missing when the column is converted, so a date filter or a "
            "time index silently drops those rows."
        ),
        evidence=Evidence(
            metrics={
                "unparsed": column.unparsed,
                "share": share,
                "warn_share": params.warn_share,
                "format": column.format,
                "epoch_unit": column.epoch_unit,
            },
            n_used=column.counts.non_null,
            n_total=column.counts.rows,
            method="parse_with_inferred_format",
        ),
    )
