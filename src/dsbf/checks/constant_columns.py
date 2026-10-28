# dsbf/checks/constant_columns.py
"""Columns that hold a single value, or none."""

from pydantic import Field

from dsbf.engine import (
    CheckContext,
    CheckOutput,
    CheckOutputModel,
    NoParams,
    Requires,
    SemanticTypes,
    check,
)
from dsbf.model import (
    Confidence,
    Cost,
    Dimension,
    Evidence,
    Severity,
    Subject,
)


class ConstantColumnsOutput(CheckOutputModel):
    constant_columns: list[str] = Field(
        description="Columns with at most one distinct non-null value, in order."
    )


@check(
    id="constant_columns",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.USABILITY],
    params=NoParams,
    output=ConstantColumnsOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.CHEAP,
    codes=["constant_column"],
    packs=["core"],
    tags=["constant", "redundancy"],
)
def constant_columns(ctx: CheckContext, params: NoParams) -> CheckOutput:
    """Flag columns in which every non-null value is the same."""
    names: list[str] = []
    for column in ctx.profile.columns:
        counts = column.counts
        if counts.rows == 0 or counts.distinct > 1:
            continue
        names.append(column.name)
        share = counts.nulls / counts.rows
        if counts.distinct == 1:
            what = "holds a single value"
            nulls = f", and is null in {share:.1%} of rows" if counts.nulls else ""
        else:
            what = "has no values at all"
            nulls = ""
        ctx.finding(
            code="constant_column",
            subject=Subject.column(column.name),
            severity=Severity.ERROR,
            confidence=(
                Confidence.HIGH if counts.distinct_exact else Confidence.MODERATE
            ),
            title="Constant column",
            message=(
                f"{column.name!r} {what}{nulls}. It cannot tell one row from another. "
                "Check that it is not a loading artefact, a column filled in by "
                "mistake, or a filter that left a single value."
            ),
            evidence=Evidence(
                metrics={"n_unique": counts.distinct, "null_share": share},
                n_used=counts.rows,
                n_total=counts.rows,
                method="distinct_count",
            ),
        )
    return ctx.output(ConstantColumnsOutput(constant_columns=names))
