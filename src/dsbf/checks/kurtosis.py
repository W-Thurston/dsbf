# dsbf/checks/kurtosis.py
"""Numeric columns whose tails are much heavier or lighter than a normal one's."""

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

Tails = Literal["thin", "normal", "mild", "heavy"]


class KurtosisParams(CheckParams):
    mild: float = Field(
        1.0,
        description="Excess kurtosis from which tails are called heavier than normal.",
    )
    strong: float = Field(
        3.0,
        description="Excess kurtosis from which tails are heavy and the finding is "
        "a warning.",
    )
    thin: float = Field(
        -1.0, description="Excess kurtosis at or below which tails are called thin."
    )

    @model_validator(mode="after")
    def _ordered(self) -> "KurtosisParams":
        if not self.thin < self.mild <= self.strong:
            raise ValueError("thin < mild <= strong")
        return self


class ColumnKurtosis(Contract):
    kurtosis: float | None = Field(
        description="Bias-corrected excess kurtosis (0 for a normal distribution); "
        "None when undefined, as for fewer than four values or a constant column."
    )
    tails: Tails | None = Field(description="None when the kurtosis is undefined.")


class KurtosisOutput(CheckOutputModel):
    columns: dict[str, ColumnKurtosis] = Field(
        description="Every numeric column with summary statistics, by name."
    )


def _tails(value: float, params: KurtosisParams) -> Tails:
    if value >= params.strong:
        return "heavy"
    if value >= params.mild:
        return "mild"
    if value <= params.thin:
        return "thin"
    return "normal"


@check(
    id="kurtosis",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.DISTRIBUTION],
    params=KurtosisParams,
    output=KurtosisOutput,
    requires=Requires(profile=["counts", "numeric_stats"]),
    cost=Cost.CHEAP,
    codes=["heavy_tails", "thin_tails"],
    packs=["core"],
    tags=["distribution", "tails", "outlier-risk"],
)
def kurtosis(ctx: CheckContext, params: KurtosisParams) -> CheckOutput:
    """Flag numeric columns with heavy or thin tails, from their excess kurtosis."""
    columns: dict[str, ColumnKurtosis] = {}
    for column in ctx.profile.columns:
        if not isinstance(column, NumericProfile) or column.stats is None:
            continue
        value = column.stats.kurtosis
        tails = _tails(value, params) if value is not None else None
        columns[column.name] = ColumnKurtosis(kurtosis=value, tails=tails)
        if value is None or tails in (None, "normal"):
            continue
        evidence = Evidence(
            metrics={"kurtosis": value, "skewness": column.stats.skewness},
            n_used=column.counts.non_null - column.non_finite,
            n_total=column.counts.rows,
            method="excess_kurtosis",
        )
        if tails == "thin":
            ctx.finding(
                code="thin_tails",
                subject=Subject.column(column.name),
                severity=Severity.INFO,
                confidence=Confidence.HIGH,
                title=f"Thin tails (kurtosis {value:.2f})",
                message=(
                    f"{column.name!r} has an excess kurtosis of {value:.2f}, below "
                    "the 0 of a normal distribution: its values spread more evenly "
                    "across the range, with fewer extremes. This is common for "
                    "bounded or discretised columns. Outlier methods calibrated for "
                    "a normal distribution will flag fewer values than expected."
                ),
                evidence=evidence,
            )
            continue
        ctx.finding(
            code="heavy_tails",
            subject=Subject.column(column.name),
            severity=Severity.WARN if tails == "heavy" else Severity.INFO,
            confidence=Confidence.HIGH,
            title=f"{'Heavy' if tails == 'heavy' else 'Somewhat heavy'} tails "
            f"(kurtosis {value:.2f})",
            message=(
                f"{column.name!r} has an excess kurtosis of {value:.2f}, above the "
                "0 of a normal distribution: extreme values are more common than "
                "its standard deviation suggests. Outlier detection and normality "
                "tests are strongly influenced by them. Distance-based models and "
                "regularised regression are sensitive to a fat-tailed feature; "
                "tree-based models are not."
            ),
            evidence=evidence,
        )
    return ctx.output(KurtosisOutput(columns=columns))
