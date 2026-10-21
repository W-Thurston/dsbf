# dsbf/checks/id_columns.py
"""Columns that look like row identifiers."""

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
    Cost,
    Dimension,
    Evidence,
    NumericProfile,
    Severity,
    Subject,
)

_INTEGER_DTYPES = ("Int", "UInt")


class IdColumnsParams(CheckParams):
    min_unique_share: float = Field(
        0.95,
        gt=0,
        le=1,
        description="Distinct values per non-null value from which a string or "
        "categorical column is read as an identifier.",
    )
    min_rows: int = Field(
        20,
        ge=2,
        description="Non-null values a column needs before its uniqueness means "
        "anything.",
    )


class IdColumnsOutput(CheckOutputModel):
    id_columns: dict[str, str] = Field(
        description="Each column read as an identifier, with the rule that read it: "
        "`unique_share` or `dense_integer_sequence`."
    )


def _is_dense_integer_sequence(column: NumericProfile) -> bool:
    """Whole numbers, no nulls, all different, and no gap between the smallest and
    the largest: a row number, whatever it starts at."""
    stats = column.stats
    counts = column.counts
    return (
        column.physical_dtype.startswith(_INTEGER_DTYPES)
        and stats is not None
        and stats.min is not None
        and stats.max is not None
        and counts.nulls == 0
        and counts.distinct == counts.rows
        and stats.max - stats.min == counts.rows - 1
    )


@check(
    id="id_columns",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("identifier", "categorical", "text", "numeric"),
    dimensions=[Dimension.USABILITY],
    params=IdColumnsParams,
    output=IdColumnsOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["id", "index"],
)
def id_columns(ctx: CheckContext, params: IdColumnsParams) -> CheckOutput:
    """Flag columns whose values label rows rather than describe them."""
    found: dict[str, str] = {}
    for column in ctx.profile.columns:
        counts = column.counts
        non_null = counts.non_null
        if non_null < params.min_rows:
            continue
        share = counts.distinct / non_null
        if isinstance(column, NumericProfile):
            if not _is_dense_integer_sequence(column):
                continue
            rule, confidence = "dense_integer_sequence", Confidence.MODERATE
            reason = (
                f"its {counts.rows:,} values are whole numbers with no repeat and no "
                "gap, which is how a row number looks"
            )
        else:
            if share < params.min_unique_share:
                continue
            rule = "unique_share"
            confidence = (
                Confidence.LOW
                if column.semantic_type == "text"
                else Confidence.MODERATE
            )
            reason = (
                f"{counts.distinct:,} of its {non_null:,} non-null values are "
                f"different ({share:.1%})"
            )
        found[column.name] = rule
        ctx.finding(
            code="likely_identifier",
            subject=Subject.column(column.name),
            severity=Severity.WARN,
            confidence=confidence,
            title="Likely identifier",
            message=(
                f"{column.name!r} looks like a row identifier: {reason}. An "
                "identifier labels rows and says nothing about them, and a model "
                "given one can memorise its rows. Confirm it is one, and keep it "
                "out of the features."
            ),
            evidence=Evidence(
                metrics={"n_unique": counts.distinct, "unique_share": share},
                n_used=non_null,
                n_total=counts.rows,
                method=rule,
            ),
        )
    return ctx.output(IdColumnsOutput(id_columns=found))
