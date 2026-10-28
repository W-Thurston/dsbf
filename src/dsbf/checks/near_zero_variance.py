# dsbf/checks/near_zero_variance.py
"""Numeric columns whose values barely differ."""

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
    NumericProfile,
    Severity,
    Subject,
)


class NearZeroVarianceParams(CheckParams):
    max_variance: float = Field(
        1e-4,
        ge=0,
        description="Variance (the square of the standard deviation) at or below "
        "which a column is reported. It is in the column's own units, so it does "
        "not suit a column measured in very small units.",
    )


class ColumnVariance(Contract):
    variance: float = Field(ge=0)
    low: bool = Field(description="The variance is at most `max_variance`.")


class NearZeroVarianceOutput(CheckOutputModel):
    columns: dict[str, ColumnVariance] = Field(
        description="Every numeric column with a standard deviation, by name."
    )


@check(
    id="near_zero_variance",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.USABILITY],
    params=NearZeroVarianceParams,
    output=NearZeroVarianceOutput,
    requires=Requires(profile=["counts", "numeric_stats"]),
    cost=Cost.CHEAP,
    codes=["near_zero_variance"],
    packs=["core"],
    tags=["variance", "low-information"],
)
def near_zero_variance(
    ctx: CheckContext, params: NearZeroVarianceParams
) -> CheckOutput:
    """Flag numeric columns whose variance is at or below `max_variance`."""
    columns: dict[str, ColumnVariance] = {}
    for column in ctx.profile.columns:
        if not isinstance(column, NumericProfile):
            continue
        stats = column.stats
        if stats is None or stats.std is None:
            continue
        variance = stats.std**2
        low = variance <= params.max_variance
        columns[column.name] = ColumnVariance(variance=variance, low=low)
        if not low:
            continue
        spread = stats.std / abs(stats.mean) if stats.mean not in (None, 0) else None
        ctx.finding(
            code="near_zero_variance",
            subject=Subject.column(column.name),
            severity=Severity.WARN,
            confidence=Confidence.HIGH,
            title=f"Near-zero variance ({variance:.2e})",
            message=(
                f"{column.name!r} has a variance of {variance:.2e} (standard "
                f"deviation {stats.std:.4g}), at or below {params.max_variance:.2e}. "
                "Almost all its values are the same, so it gives a model little to "
                "learn from. Check whether a sensor is stuck or the column is "
                "measured in very small units."
            ),
            evidence=Evidence(
                metrics={
                    "variance": variance,
                    "std": stats.std,
                    "relative_spread": spread,
                    "max_variance": params.max_variance,
                },
                n_used=column.counts.non_null,
                n_total=column.counts.rows,
                method="sample_variance",
            ),
        )
    return ctx.output(NearZeroVarianceOutput(columns=columns))
