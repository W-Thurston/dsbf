# dsbf/checks/zeros.py
"""Numeric columns in which zero is a large share of the values."""

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
    NumericProfile,
    Severity,
    Subject,
)


class ZerosParams(CheckParams):
    notice_share: float = Field(
        0.30, ge=0, le=1, description="Zero share from which a column is reported."
    )
    warn_share: float = Field(
        0.50, ge=0, le=1, description="Zero share from which the finding is a warning."
    )
    flag_share: float = Field(
        0.95, ge=0, le=1, description="Zero share above which a column is flagged."
    )

    @model_validator(mode="after")
    def _ordered(self) -> "ZerosParams":
        if self.notice_share > self.warn_share:
            raise ValueError("notice_share is at most warn_share")
        return self


class ColumnZeros(Contract):
    zero_count: int = Field(ge=0)
    non_null: int = Field(
        ge=1, description="Non-null values, which the share is taken over."
    )
    zero_share: float = Field(ge=0, le=1)
    flagged: bool = Field(description="The share is above `flag_share`.")


class ZerosOutput(CheckOutputModel):
    columns: dict[str, ColumnZeros] = Field(
        description="Every numeric column, by name."
    )


@check(
    id="zeros",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.DISTRIBUTION],
    params=ZerosParams,
    output=ZerosOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.CHEAP,
    codes=["high_zero_rate"],
    packs=["core"],
    tags=["zeros", "sparsity"],
)
def zeros(ctx: CheckContext, params: ZerosParams) -> CheckOutput:
    """Flag numeric columns in which zero is a large share of the values."""
    columns: dict[str, ColumnZeros] = {}
    for column in ctx.profile.columns:
        if not isinstance(column, NumericProfile):
            continue
        non_null = column.counts.non_null
        share = column.zeros / non_null
        columns[column.name] = ColumnZeros(
            zero_count=column.zeros,
            non_null=non_null,
            zero_share=share,
            flagged=share > params.flag_share,
        )
        if share < params.notice_share:
            continue
        severity = Severity.WARN if share >= params.warn_share else Severity.INFO
        ctx.finding(
            code="high_zero_rate",
            subject=Subject.column(column.name),
            severity=severity,
            confidence=Confidence.HIGH,
            title=f"{'High' if severity is Severity.WARN else 'Notable'} zero rate "
            f"({share:.1%})",
            message=(
                f"{column.name!r} is zero in {column.zeros:,} of {non_null:,} values "
                f"({share:.1%}). A zero can be a measurement, such as nothing sold, or "
                "a placeholder for a missing value; the two call for different "
                "treatment."
            ),
            evidence=Evidence(
                metrics={"zero_count": column.zeros, "zero_share": share},
                n_used=non_null,
                n_total=ctx.profile.n_rows,
                method="zero_share",
            ),
        )
    return ctx.output(ZerosOutput(columns=columns))
