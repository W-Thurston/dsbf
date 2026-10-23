# dsbf/checks/imputation.py
"""How to fill the missing values of a column, from how many are missing and why."""

from typing import Any, Literal

from pydantic import Field

from dsbf.engine import (
    CheckContext,
    CheckOutput,
    CheckOutputModel,
    Requires,
    check,
)
from dsbf.model import (
    Action,
    BooleanProfile,
    CategoricalProfile,
    CheckParams,
    Confidence,
    Contract,
    Cost,
    DatetimeProfile,
    Dimension,
    Evidence,
    Guidance,
    NumericProfile,
    Severity,
    Subject,
)

Tier = Literal["low", "moderate", "significant", "severe"]
Strategy = Literal[
    "mean",
    "median",
    "mode",
    "constant_missing",
    "forward_fill",
    "iterative",
    "indicator_only",
    "indicator_and_drop",
]
Mechanism = Literal["depends_on_observed", "no_dependence_found", "untestable"]

# The `method` of the `impute` actions this check emits; `add_indicator` and `drop`
# are actions of their own.
IMPUTE_METHODS: tuple[str, ...] = (
    "mean",
    "median",
    "mode",
    "constant_missing",
    "forward_fill",
    "iterative",
)

_IMPUTABLE = (NumericProfile, CategoricalProfile, BooleanProfile, DatetimeProfile)
_TIER_OF_SEVERITY = {
    Severity.ERROR: "severe",
    Severity.WARN: "significant",
    Severity.INFO: "moderate",
}


class ImputationParams(CheckParams):
    min_null_share: float = Field(
        0.01,
        ge=0,
        lt=1,
        description="Share of rows that must be null for a column to be advised.",
    )
    skew_threshold: float = Field(
        1.0,
        gt=0,
        description="Absolute skewness from which the median, not the mean, fills "
        "a numeric column.",
    )


class ColumnImputation(Contract):
    semantic_type: str
    null_share: float = Field(gt=0, le=1)
    tier: Tier = Field(
        description="From the severity `null_rate` gave the column: info is "
        "moderate, warn significant, error severe, and a share it did not report "
        "is low."
    )
    strategy: Strategy
    add_indicator: bool
    mechanism: Mechanism | None = Field(
        description="What `missingness_mechanism` found; None when it did not "
        "analyse the column or did not run."
    )


class ImputationOutput(CheckOutputModel):
    columns: dict[str, ColumnImputation] = Field(
        description="Each column with nulls that can be filled, by name."
    )


def _mechanism(ctx: CheckContext) -> dict[str, Any]:
    result = ctx.result("missingness_mechanism")
    if result is None or not result.output:
        return {}
    return result.output.get("columns", {})


_WHY_MECHANISM = {
    "depends_on_observed": "whether it is missing depends on other columns, so the "
    "other columns carry information about the missing values, and a model that "
    "uses them biases the filled values less than a constant",
    "no_dependence_found": "whether it is missing shows no dependence on the "
    "other columns, which is consistent with values missing at random (a test "
    "cannot show that they are)",
    "untestable": "whether it is missing could not be tested against the other columns",
}


def _advise(
    profile: Any,
    tier: Tier,
    mechanism: Mechanism | None,
    params: ImputationParams,
) -> tuple[Strategy, bool, str]:
    """The strategy, whether to add a missing indicator, and why in words."""
    depends = mechanism == "depends_on_observed"
    if tier == "severe":
        return (
            "indicator_and_drop",
            True,
            "most of its values are missing, so any fill would be mostly invented",
        )
    significant = tier == "significant"
    if isinstance(profile, NumericProfile):
        skew = profile.stats.skewness if profile.stats else None
        skewed = skew is not None and abs(skew) >= params.skew_threshold
        if depends and tier != "low":
            return (
                "iterative",
                True,
                "a model of the other columns fills it better than one constant",
            )
        if skewed:
            return (
                "median",
                significant or depends,
                f"it is skewed ({skew:.2f}), so the mean follows the tail",
            )
        return "mean", significant or depends, "its values are roughly symmetric"
    if isinstance(profile, (CategoricalProfile, BooleanProfile)):
        if significant or (depends and tier != "low"):
            return (
                "constant_missing",
                False,
                "an explicit missing level keeps the signal in the gaps",
            )
        return "mode", False, "the commonest value is a safe fill at this level"
    if isinstance(profile, DatetimeProfile) and profile.inferred_frequency:
        return (
            "forward_fill",
            significant or depends,
            f"it has a regular {profile.inferred_frequency} frequency, so the "
            "previous value is a fair fill",
        )
    return (
        "indicator_only",
        True,
        "a date has no safe fill when it is not on a regular frequency",
    )


_TITLE: dict[Strategy, str] = {
    "mean": "Fill with the mean",
    "median": "Fill with the median",
    "mode": "Fill with the mode",
    "constant_missing": "Fill with a missing level",
    "forward_fill": "Forward fill",
    "iterative": "Fill from the other columns",
    "indicator_only": "Add a missing indicator",
    "indicator_and_drop": "Add a missing indicator and drop",
}

_DETAIL = {
    "mean": "Fill with the mean of the training split.",
    "median": "Fill with the median of the training split.",
    "mode": "Fill with the most frequent value of the training split.",
    "constant_missing": "Fill with a level called `missing`.",
    "forward_fill": "Fill with the previous value in time order.",
    "iterative": "Fill from a regression on the other columns, fitted on the "
    "training split.",
}


@check(
    id="imputation",
    version="1.0.0",
    kind="synthesis",
    applies_to=None,
    dimensions=[Dimension.USABILITY],
    params=ImputationParams,
    output=ImputationOutput,
    requires=Requires(checks=["null_rate"]),
    after=["missingness_mechanism"],
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["missingness", "imputation", "ml"],
)
def imputation(ctx: CheckContext, params: ImputationParams) -> CheckOutput:
    """Suggest how to fill each column's nulls from the null rate and its mechanism."""
    null_rate = ctx.result("null_rate")
    tiers: dict[str, Tier] = {}
    if null_rate is not None:
        for finding in null_rate.findings:
            tiers[finding.subject.columns[0]] = _TIER_OF_SEVERITY.get(  # type: ignore[assignment]
                finding.severity, "moderate"
            )
    mechanisms = _mechanism(ctx)
    columns: dict[str, ColumnImputation] = {}
    for profile in ctx.profile.columns:
        counts = profile.counts
        share = counts.nulls / counts.rows if counts.rows else 0.0
        if counts.nulls == 0 or share < params.min_null_share:
            continue
        if not isinstance(
            profile,
            (NumericProfile, CategoricalProfile, BooleanProfile, DatetimeProfile),
        ):
            continue
        tier = tiers.get(profile.name, "low")
        entry = mechanisms.get(profile.name)
        mechanism: Mechanism | None = entry["assessment"] if entry else None
        strategy, indicator, reason = _advise(profile, tier, mechanism, params)
        columns[profile.name] = ColumnImputation(
            semantic_type=profile.semantic_type,
            null_share=share,
            tier=tier,
            strategy=strategy,
            add_indicator=indicator,
            mechanism=mechanism,
        )
        why = (
            f"{_WHY_MECHANISM[mechanism]}"
            if mechanism
            else "whether it is missing was not analysed against the other columns"
        )
        actions: list[Action] = []
        if strategy in IMPUTE_METHODS:
            actions.append(
                Action(
                    action="impute",
                    columns=(profile.name,),
                    method=strategy,
                    condition=reason,
                    detail=_DETAIL[strategy],
                )
            )
        if indicator:
            actions.append(
                Action(
                    action="add_indicator",
                    columns=(profile.name,),
                    detail="Add a 0/1 column that is 1 where the value is missing, "
                    "before filling it.",
                )
            )
        if strategy == "indicator_and_drop":
            actions.append(
                Action(
                    action="drop",
                    columns=(profile.name,),
                    condition=reason,
                    detail="Drop the column once the indicator exists, unless the "
                    "values are needed and can be collected again.",
                )
            )
        metrics: dict[str, Any] = {
            "null_share": share,
            "null_count": counts.nulls,
            "tier": tier,
            "strategy": strategy,
            "mechanism": mechanism,
        }
        if isinstance(profile, NumericProfile) and profile.stats:
            metrics["skewness"] = profile.stats.skewness
        ctx.finding(
            code="imputation_strategy",
            subject=Subject.column(profile.name),
            severity=Severity.WARN if tier == "severe" else Severity.INFO,
            confidence=Confidence.MODERATE,
            title=f"{_TITLE[strategy]} ({share:.1%} missing)",
            message=(
                f"{profile.name!r} is missing in {counts.nulls:,} of "
                f"{counts.rows:,} rows ({share:.1%}, a {tier} share): {reason}; "
                f"{why}."
            ),
            evidence=Evidence(
                metrics=metrics,
                n_used=counts.rows,
                n_total=counts.rows,
                method="null_share_and_mechanism",
            ),
            guidance=Guidance(phase="ml", actions=tuple(actions)),
        )
    return ctx.output(ImputationOutput(columns=columns))
