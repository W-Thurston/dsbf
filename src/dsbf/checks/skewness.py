# dsbf/checks/skewness.py
"""Numeric columns whose values lean to one side."""

from typing import Literal

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

Level = Literal["symmetric", "mild", "moderate", "heavy"]


class SkewnessParams(CheckParams):
    mild: float = Field(
        0.5, gt=0, description="Absolute skewness above which a column is reported."
    )
    moderate: float = Field(
        1.0, gt=0, description="Absolute skewness above which the finding is a warning."
    )
    heavy: float = Field(
        2.0, gt=0, description="Absolute skewness above which the skew is called heavy."
    )
    gap_notice: float = Field(
        0.5,
        gt=0,
        description="Distance between the mean and the median, in standard "
        "deviations, from which the gap is reported.",
    )
    gap_warn: float = Field(
        1.0,
        gt=0,
        description="Gap, in standard deviations, from which the finding is a warning.",
    )
    gap_min_variance: float = Field(
        1e-4,
        ge=0,
        description="Variance at or below which the gap is not reported: a column "
        "that barely varies is `near_zero_variance`'s to report.",
    )

    @model_validator(mode="after")
    def _ordered(self) -> "SkewnessParams":
        if not self.mild <= self.moderate <= self.heavy:
            raise ValueError("mild <= moderate <= heavy")
        if self.gap_notice > self.gap_warn:
            raise ValueError("gap_notice is at most gap_warn")
        return self


class ColumnSkew(Contract):
    skewness: float | None = Field(
        description="Bias-corrected sample skewness; None when undefined, as for a "
        "constant column or fewer than three values."
    )
    level: Level | None = Field(
        description="How far the skewness is from zero; None when it is undefined."
    )
    gap_in_std: float | None = Field(
        description="|mean − median| in standard deviations; None without a spread."
    )


class SkewnessOutput(CheckOutputModel):
    columns: dict[str, ColumnSkew] = Field(
        description="Every numeric column with summary statistics, by name."
    )


def _level(size: float, params: SkewnessParams) -> Level:
    if size > params.heavy:
        return "heavy"
    if size > params.moderate:
        return "moderate"
    if size > params.mild:
        return "mild"
    return "symmetric"


@check(
    id="skewness",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.DISTRIBUTION],
    params=SkewnessParams,
    output=SkewnessOutput,
    requires=Requires(profile=["counts", "numeric_stats", "quantiles"]),
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["distribution", "skew"],
)
def skewness(ctx: CheckContext, params: SkewnessParams) -> CheckOutput:
    """Flag numeric columns that lean to one side or whose mean sits off the median."""
    columns: dict[str, ColumnSkew] = {}
    for column in ctx.profile.columns:
        if not isinstance(column, NumericProfile) or column.stats is None:
            continue
        stats = column.stats
        median = next((q.value for q in stats.quantiles if q.p == 0.5), None)
        gap = None
        if stats.std and stats.mean is not None and median is not None:
            gap = abs(stats.mean - median) / stats.std
        size = abs(stats.skewness) if stats.skewness is not None else None
        level = _level(size, params) if size is not None else None
        columns[column.name] = ColumnSkew(
            skewness=stats.skewness, level=level, gap_in_std=gap
        )
        metrics = {
            "mean": stats.mean,
            "median": median,
            "std": stats.std,
            "min": stats.min,
        }
        used = {
            "n_used": column.counts.non_null - column.non_finite,
            "n_total": column.counts.rows,
        }
        if level not in (None, "symmetric") and stats.skewness is not None:
            side = "right (positive)" if stats.skewness > 0 else "left (negative)"
            ctx.finding(
                code="skewed",
                subject=Subject.column(column.name),
                severity=Severity.INFO if level == "mild" else Severity.WARN,
                confidence=Confidence.HIGH,
                title=f"{level.capitalize()} {side.split()[0]} skew "
                f"({stats.skewness:+.2f})",
                message=(
                    f"{column.name!r} has {level} {side} skew (skewness "
                    f"{stats.skewness:.2f}): its values bunch toward the "
                    f"{'lower' if stats.skewness > 0 else 'higher'} end with a tail "
                    f"toward the {'higher' if stats.skewness > 0 else 'lower'} end, so "
                    "the median describes a typical value better than the mean. "
                    "Linear and distance-based models are affected; tree-based "
                    "models are largely robust. Check that the tail is genuine and "
                    "not a few anomalous values."
                ),
                evidence=Evidence(
                    metrics={"skewness": stats.skewness, **metrics},
                    method="sample_skewness",
                    **used,
                ),
            )
        spread = stats.std is not None and stats.std**2 > params.gap_min_variance
        if gap is not None and spread and gap >= params.gap_notice:
            above = stats.mean > median  # type: ignore[operator]
            ctx.finding(
                code="mean_median_gap",
                subject=Subject.column(column.name),
                severity=Severity.WARN if gap >= params.gap_warn else Severity.INFO,
                confidence=Confidence.HIGH,
                title=f"Mean-median gap ({gap:.2f}σ)",
                message=(
                    f"The mean of {column.name!r} ({stats.mean:.4g}) is {gap:.2f} "
                    f"standard deviations {'above' if above else 'below'} its median "
                    f"({median:.4g}), pulled {'up' if above else 'down'} by "
                    f"{'high' if above else 'low'} values. Models that assume "
                    "normality (linear regression, LDA, Gaussian naive Bayes) are "
                    "affected. Check the histogram to see whether a long tail or a "
                    "cluster of outliers causes it."
                ),
                evidence=Evidence(
                    metrics={"gap_in_std": gap, **metrics},
                    method="mean_median_gap",
                    **used,
                ),
            )
    return ctx.output(SkewnessOutput(columns=columns))
