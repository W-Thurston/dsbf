# dsbf/checks/categorical_encoding.py
"""How to turn a categorical column into numbers for a model."""

from typing import Literal

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
    Action,
    CategoricalProfile,
    CheckParams,
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    Guidance,
    Severity,
    Subject,
)

Strategy = Literal["binary", "one_hot", "frequency"]

# The `method` of the `encode` actions this check emits; the export that turns
# guidance into a pipeline reads them.
ENCODE_METHODS: tuple[Strategy, ...] = ("binary", "one_hot", "frequency")


class CategoricalEncodingParams(CheckParams):
    max_one_hot: int = Field(
        10,
        ge=2,
        description="Most distinct values for which one-hot columns are suggested.",
    )
    min_rows_per_level: int = Field(
        20,
        ge=1,
        description="Rows per distinct value below which one-hot columns would be "
        "too sparse to estimate, so the frequency of each value is suggested.",
    )


class ColumnEncoding(Contract):
    cardinality: int = Field(ge=2, description="Distinct non-null values.")
    rows_per_level: float = Field(gt=0, description="Non-null rows per distinct value.")
    strategy: Strategy
    rarest_level_count: int | None = Field(
        description="Rows of the least frequent value; known only when the profile "
        "lists every value."
    )


class CategoricalEncodingOutput(CheckOutputModel):
    columns: dict[str, ColumnEncoding] = Field(
        description="Each categorical column with at least two values, by name."
    )


def _strategy(
    distinct: int, rows_per_level: float, params: CategoricalEncodingParams
) -> tuple[Strategy, str]:
    """The encoding for a column and why, in words."""
    if distinct == 2:
        return "binary", "it has two values, which one 0/1 column holds"
    if distinct > params.max_one_hot:
        return (
            "frequency",
            f"it has {distinct} values, more than {params.max_one_hot} one-hot "
            "columns can carry without a wide, sparse matrix",
        )
    if rows_per_level < params.min_rows_per_level:
        return (
            "frequency",
            f"each value has only {rows_per_level:.0f} rows on average, fewer than "
            f"the {params.min_rows_per_level} a one-hot column needs to be estimated",
        )
    return "one_hot", f"it has {distinct} values, few enough for one column each"


_DETAIL = {
    "binary": "Map the two values to 0 and 1.",
    "one_hot": "Make one 0/1 column per value and drop one, so the columns are not "
    "collinear.",
    "frequency": "Replace each value by its share of the rows, fitted on the "
    "training split. Frequency is not a ranking: a model can mistake it for one.",
}


@check(
    id="categorical_encoding",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("categorical"),
    dimensions=[Dimension.USABILITY],
    params=CategoricalEncodingParams,
    output=CategoricalEncodingOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["encoding", "ml"],
)
def categorical_encoding(
    ctx: CheckContext, params: CategoricalEncodingParams
) -> CheckOutput:
    """Suggest an encoding for each categorical column from its cardinality."""
    columns: dict[str, ColumnEncoding] = {}
    for column in ctx.profile.columns:
        if not isinstance(column, CategoricalProfile):
            continue
        counts = column.counts
        if counts.distinct < 2:  # `constant_columns` reports these
            continue
        rows_per_level = counts.non_null / counts.distinct
        strategy, reason = _strategy(counts.distinct, rows_per_level, params)
        listed = len(column.top_values) >= counts.distinct
        rarest = min((v.count for v in column.top_values), default=None)
        columns[column.name] = ColumnEncoding(
            cardinality=counts.distinct,
            rows_per_level=rows_per_level,
            strategy=strategy,
            rarest_level_count=rarest if listed else None,
        )
        ctx.finding(
            code="encoding_strategy",
            subject=Subject.column(column.name),
            severity=Severity.INFO,
            confidence=Confidence.HIGH if counts.distinct_exact else Confidence.LOW,
            title=f"Encode as {strategy.replace('_', '-')} ({counts.distinct} values)",
            message=(
                f"{column.name!r} has {counts.distinct:,} distinct values in "
                f"{counts.non_null:,} rows ({rows_per_level:,.0f} rows per value); "
                f"{reason}. A tree-based model can read integer codes directly; "
                "linear, distance-based and neural models need the encoding."
                + (
                    ""
                    if counts.distinct_exact
                    else " The count of values is an estimate."
                )
            ),
            evidence=Evidence(
                metrics={
                    "cardinality": counts.distinct,
                    "rows_per_level": rows_per_level,
                    "rarest_level_count": rarest if listed else None,
                    "strategy": strategy,
                },
                n_used=counts.non_null,
                n_total=counts.rows,
                method="cardinality_rows_per_level",
            ),
            guidance=Guidance(
                phase="ml",
                actions=(
                    Action(
                        action="encode",
                        columns=(column.name,),
                        method=strategy,
                        condition=reason,
                        detail=_DETAIL[strategy],
                        params={"n_levels": counts.distinct},
                    ),
                ),
            ),
        )
    return ctx.output(CategoricalEncodingOutput(columns=columns))
