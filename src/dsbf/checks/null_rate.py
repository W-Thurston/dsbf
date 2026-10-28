# dsbf/checks/null_rate.py
"""Columns with a large share of missing values."""

from pydantic import Field, model_validator

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
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    Severity,
    Subject,
)


class NullRateParams(CheckParams):
    notice_share: float = Field(
        0.05, ge=0, le=1, description="Null share from which a column is reported."
    )
    warn_share: float = Field(
        0.20, ge=0, le=1, description="Null share from which the finding is a warning."
    )
    error_share: float = Field(
        0.50,
        ge=0,
        le=1,
        description="Null share from which the finding is an error and the column "
        "is listed as mostly missing.",
    )

    @model_validator(mode="after")
    def _ordered(self) -> "NullRateParams":
        if not self.notice_share <= self.warn_share <= self.error_share:
            raise ValueError("notice_share <= warn_share <= error_share")
        return self


class ColumnNulls(Contract):
    null_count: int = Field(ge=0)
    rows: int = Field(ge=0)
    null_share: float = Field(ge=0, le=1)
    mostly_missing: bool = Field(description="The share is at least `error_share`.")


class NullRateOutput(CheckOutputModel):
    columns: dict[str, ColumnNulls] = Field(description="Every column, by name.")


@check(
    id="null_rate",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.COMPLETENESS],
    params=NullRateParams,
    output=NullRateOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.CHEAP,
    codes=["high_null_rate"],
    packs=["core"],
    tags=["missingness"],
)
def null_rate(ctx: CheckContext, params: NullRateParams) -> CheckOutput:
    """Report the share of missing values in each column."""
    columns: dict[str, ColumnNulls] = {}
    for column in ctx.profile.columns:
        counts = column.counts
        share = counts.nulls / counts.rows if counts.rows else 0.0
        columns[column.name] = ColumnNulls(
            null_count=counts.nulls,
            rows=counts.rows,
            null_share=share,
            mostly_missing=share >= params.error_share,
        )
        if counts.nulls == 0 or share < params.notice_share:
            continue
        if share >= params.error_share:
            severity, label = Severity.ERROR, "Severe"
        elif share >= params.warn_share:
            severity, label = Severity.WARN, "High"
        else:
            severity, label = Severity.INFO, "Notable"
        ctx.finding(
            code="high_null_rate",
            subject=Subject.column(column.name),
            severity=severity,
            confidence=Confidence.HIGH,
            title=f"{label} missingness ({share:.1%})",
            message=(
                f"{column.name!r} is missing in {counts.nulls:,} of {counts.rows:,} "
                f"rows ({share:.1%}). Find out why before filling or dropping it: a "
                "value that was never collected, one that only applies to some rows, "
                "and one lost in transit call for different treatment."
            ),
            evidence=Evidence(
                metrics={"null_count": counts.nulls, "null_share": share},
                n_used=counts.rows,
                n_total=counts.rows,
                method="null_share",
            ),
        )
    return ctx.output(NullRateOutput(columns=columns))
