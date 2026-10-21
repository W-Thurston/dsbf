# dsbf/checks/high_cardinality.py
"""Categorical columns with many distinct values."""

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
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    Severity,
    Subject,
)


class HighCardinalityParams(CheckParams):
    max_distinct: int = Field(
        50,
        ge=2,
        description="Distinct values above which a categorical column is reported.",
    )


class ColumnCardinality(Contract):
    distinct: int = Field(ge=0)
    distinct_share: float = Field(
        ge=0, le=1, description="Distinct values per non-null value."
    )
    high: bool = Field(description="The column has more than `max_distinct` values.")


class HighCardinalityOutput(CheckOutputModel):
    columns: dict[str, ColumnCardinality] = Field(
        description="Every categorical column, by name."
    )


@check(
    id="high_cardinality",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("categorical"),
    dimensions=[Dimension.USABILITY],
    params=HighCardinalityParams,
    output=HighCardinalityOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["cardinality", "encoding"],
)
def high_cardinality(ctx: CheckContext, params: HighCardinalityParams) -> CheckOutput:
    """Flag categorical columns with more distinct values than `max_distinct`."""
    columns: dict[str, ColumnCardinality] = {}
    for column in ctx.profile.columns:
        counts = column.counts
        non_null = counts.non_null
        high = counts.distinct > params.max_distinct
        share = counts.distinct / non_null if non_null else 0.0
        columns[column.name] = ColumnCardinality(
            distinct=counts.distinct, distinct_share=share, high=high
        )
        if not high:
            continue
        ctx.finding(
            code="high_cardinality",
            subject=Subject.column(column.name),
            severity=Severity.WARN,
            confidence=(
                Confidence.HIGH if counts.distinct_exact else Confidence.MODERATE
            ),
            title=f"High cardinality ({counts.distinct:,} distinct values)",
            message=(
                f"{column.name!r} has {counts.distinct:,} distinct values in "
                f"{non_null:,} non-null rows, above {params.max_distinct:,}. A "
                "frequency table of it is mostly a long tail, and one-hot encoding "
                "it would add that many sparse columns."
            ),
            evidence=Evidence(
                metrics={
                    "n_unique": counts.distinct,
                    "distinct_share": share,
                    "max_distinct": params.max_distinct,
                },
                n_used=non_null,
                n_total=counts.rows,
                method="distinct_count",
            ),
        )
    return ctx.output(HighCardinalityOutput(columns=columns))
