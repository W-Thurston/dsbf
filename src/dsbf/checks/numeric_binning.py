# dsbf/checks/numeric_binning.py
"""How to cut a numeric column into bins, where binning would help a model."""

import math
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
    CheckParams,
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    Guidance,
    NumericProfile,
    Scalar,
    Severity,
    Subject,
)

Strategy = Literal["quantile", "equal_width"]

# The `method` of the `bin` actions this check emits.
BIN_METHODS: tuple[Strategy, ...] = ("quantile", "equal_width")

_LOW_P, _HIGH_P = 0.01, 0.99


class NumericBinningParams(CheckParams):
    skew_threshold: float = Field(
        1.0,
        gt=0,
        description="Absolute skewness from which equal-width bins would crowd most "
        "rows into a few of them, so bins of equal row count are suggested.",
    )
    min_core_span: float = Field(
        0.5,
        gt=0,
        le=1,
        description="Share of the range (max − min) that the middle 98% of the "
        "values must cover for equal-width bins to be suggested; below it a long "
        "tail leaves most bins nearly empty.",
    )
    min_distinct: int = Field(
        10,
        ge=2,
        description="Distinct values at or below which a column is already "
        "discrete and is left alone.",
    )
    max_bins: int = Field(10, ge=3, description="Most bins suggested.")
    min_rows_per_bin: int = Field(
        30, ge=1, description="Rows each bin should hold; fewer bins are suggested."
    )


class ColumnBinning(Contract):
    strategy: Strategy
    n_bins: int = Field(ge=2)
    skewness: float | None
    core_span: float | None = Field(
        description="Share of the range the middle 98% of the values covers; None "
        "when the profile has no 1% and 99% quantiles."
    )
    spike_value: float | None = Field(
        description="A value that holds at least 1/n_bins of the rows, which "
        "quantile edges collapse onto."
    )
    spike_share: float | None


class NumericBinningOutput(CheckOutputModel):
    columns: dict[str, ColumnBinning] = Field(
        description="Each numeric column with a suggestion, by name."
    )
    skipped_columns: dict[str, str] = Field(
        description="Numeric columns left out, with the reason."
    )


def _n_bins(n: int, params: NumericBinningParams) -> int:
    """Sturges' rule, kept between 3 and ``max_bins`` and to ``min_rows_per_bin``."""
    bins = min(params.max_bins, max(3, math.ceil(math.log2(n)) + 1))
    while bins > 2 and n / bins < params.min_rows_per_bin:
        bins -= 1
    return bins


def _core_span(profile: NumericProfile) -> float | None:
    stats = profile.stats
    if stats is None or stats.min is None or stats.max is None:
        return None
    quantiles = {q.p: q.value for q in stats.quantiles}
    if _LOW_P not in quantiles or _HIGH_P not in quantiles or stats.max <= stats.min:
        return None
    return (quantiles[_HIGH_P] - quantiles[_LOW_P]) / (stats.max - stats.min)


@check(
    id="numeric_binning",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.USABILITY],
    params=NumericBinningParams,
    output=NumericBinningOutput,
    requires=Requires(profile=["counts", "numeric_stats", "quantiles"]),
    cost=Cost.CHEAP,
    codes=["binning_strategy"],
    packs=["core"],
    tags=["binning", "ml"],
)
def numeric_binning(ctx: CheckContext, params: NumericBinningParams) -> CheckOutput:
    """Suggest quantile or equal-width bins, and how many, for each numeric column."""
    columns: dict[str, ColumnBinning] = {}
    skipped: dict[str, str] = {}
    for column in ctx.profile.columns:
        if not isinstance(column, NumericProfile) or column.stats is None:
            continue
        stats = column.stats
        n = column.counts.non_null - column.non_finite
        if not stats.std or column.counts.distinct <= 1:
            skipped[column.name] = "constant"
            continue
        if column.counts.distinct <= params.min_distinct:
            skipped[column.name] = (
                f"{column.counts.distinct} distinct values, already discrete"
            )
            continue
        if n < 2 * params.min_rows_per_bin:
            skipped[column.name] = (
                f"{n} values, fewer than two bins of {params.min_rows_per_bin}"
            )
            continue
        n_bins = _n_bins(n, params)
        span = _core_span(column)
        skew = stats.skewness
        tailed = skew is not None and abs(skew) >= params.skew_threshold
        crowded = span is not None and span < params.min_core_span
        strategy: Strategy = "quantile" if tailed or crowded else "equal_width"
        spike_share = column.zeros / n if n else 0.0
        spike = strategy == "quantile" and spike_share >= 1 / n_bins
        columns[column.name] = ColumnBinning(
            strategy=strategy,
            n_bins=n_bins,
            skewness=skew,
            core_span=span,
            spike_value=0.0 if spike else None,
            spike_share=spike_share if spike else None,
        )
        reasons = []
        if tailed:
            reasons.append(f"its skewness is {skew:.2f}")
        if crowded:
            reasons.append(
                f"its middle 98% covers only {span:.0%} of its range, so a tail "
                "stretches the range"
            )
        if not reasons:
            reasons.append(
                "its values are spread over its range"
                + (f" (the middle 98% covers {span:.0%} of it)" if span else "")
            )
        why = " and ".join(reasons)
        params_out: dict[str, Scalar] = {"n_bins": n_bins}
        detail = (
            "Cut at evenly spaced values between the minimum and maximum."
            if strategy == "equal_width"
            else "Cut at the quantiles, so each bin holds about the same number of "
            "rows; drop duplicate edges."
        )
        condition = f"{why}; bin edges fitted on the training split"
        if spike:
            params_out["spike_value"] = 0.0
            detail += (
                f" {spike_share:.0%} of the rows are exactly 0, which collapse "
                "quantile edges: bin the zeros on their own and cut the rest."
            )
        ctx.finding(
            code="binning_strategy",
            subject=Subject.column(column.name),
            severity=Severity.INFO,
            confidence=Confidence.MODERATE,
            title=f"Bin by {strategy.replace('_', '-')} into {n_bins} bins",
            message=(
                f"{column.name!r}: {why}, so {strategy.replace('_', '-')} binning "
                f"into {n_bins} bins keeps at least {n / n_bins:,.0f} rows in an "
                "average bin. Binning lets a linear model fit a step shape and "
                "makes it robust to outliers; it discards the order within a bin, "
                "and tree-based models gain nothing from it."
            ),
            evidence=Evidence(
                metrics={
                    "skewness": skew,
                    "core_span": span,
                    "n_bins": n_bins,
                    "distinct": column.counts.distinct,
                    "spike_share": spike_share if spike else None,
                    "strategy": strategy,
                },
                n_used=n,
                n_total=column.counts.rows,
                method="skewness_core_span",
            ),
            guidance=Guidance(
                phase="ml",
                actions=(
                    Action(
                        action="bin",
                        columns=(column.name,),
                        method=strategy,
                        condition=condition,
                        detail=detail,
                        params=params_out,
                    ),
                ),
            ),
        )
    return ctx.output(NumericBinningOutput(columns=columns, skipped_columns=skipped))
