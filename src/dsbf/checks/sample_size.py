# dsbf/checks/sample_size.py
"""Whether the dataset has enough rows for its number of features."""

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
    CheckParams,
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    Severity,
    Subject,
)

Verdict = Literal["adequate", "limited", "insufficient"]

# Columns that identify a row, or locate it in time, are not predictors.
_NOT_FEATURES = ("identifier", "datetime", "unknown")


class SampleSizeParams(CheckParams):
    n_features: int | None = Field(
        None,
        ge=1,
        description="The number of predictors, when only some columns will be used. "
        "By default every column except identifiers, datetimes and unknown types.",
    )
    linear_rows_per_feature: int = Field(
        20,
        ge=1,
        description="Rows per predictor a linear or logistic regression needs "
        "(Harrell 2001; Peduzzi 1996).",
    )
    linear_min_rows: int = Field(50, ge=1, description="Rows it needs at least.")
    tree_rows_per_feature: int = Field(
        10, ge=1, description="Rows per predictor a tree-based model needs."
    )
    tree_min_rows: int = Field(100, ge=1, description="Rows it needs at least.")
    general_rows_per_feature: int = Field(
        5, ge=1, description="Rows per predictor for a train, validation, test split."
    )
    general_min_rows: int = Field(
        1000,
        ge=1,
        description="Rows a 60/20/20 split needs for a validation set that gives "
        "reliable metrics.",
    )


class RuleResult(Contract):
    label: str
    required_rows: int = Field(ge=1)
    adequate: bool
    shortfall: int = Field(ge=0)


class SampleSizeOutput(CheckOutputModel):
    rows: int = Field(ge=0)
    n_features: int = Field(ge=1)
    rows_per_feature: float = Field(ge=0)
    verdict: Verdict = Field(
        description="`adequate` when every rule passes, `insufficient` when neither "
        "the linear nor the tree rule does, `limited` otherwise."
    )
    rules: dict[str, RuleResult] = Field(
        description="`linear_model`, `tree_model` and `general_ml`."
    )


# Each rule's key, label and the severity of its finding.
_RULES = (
    ("linear_model", "Linear / Logistic Regression", Severity.WARN),
    ("tree_model", "Tree-Based Models", Severity.INFO),
    ("general_ml", "General ML / Train-Val-Test Split", Severity.INFO),
)
_CAUSES = {
    "linear_model": "A linear model fits one coefficient per feature, and with fewer "
    "than about {per} rows per feature the estimates are unstable.",
    "tree_model": "A tree-based model is less sensitive to the ratio, but with too "
    "few rows it overfits.",
    "general_ml": "A 60/20/20 split leaves a validation set too small for reliable "
    "metrics, so the variance of the evaluation outweighs the differences between "
    "models.",
}


@check(
    id="sample_size",
    version="1.0.0",
    kind="rule",
    # Every column, to count the features; the findings are about the dataset.
    applies_to=SemanticTypes(),
    dimensions=[Dimension.USABILITY],
    params=SampleSizeParams,
    output=SampleSizeOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.CHEAP,
    codes=[
        "too_few_rows_general_ml",
        "too_few_rows_linear_model",
        "too_few_rows_tree_model",
    ],
    packs=["core"],
    tags=["sample-size", "ml-readiness"],
)
def sample_size(ctx: CheckContext, params: SampleSizeParams) -> CheckOutput:
    """Compare the rows with what rules of thumb ask for the number of features."""
    rows = ctx.profile.n_rows
    features = params.n_features or sum(
        c.semantic_type not in _NOT_FEATURES for c in ctx.profile.columns
    )
    features = max(features, 1)
    thresholds = {
        "linear_model": (params.linear_rows_per_feature, params.linear_min_rows),
        "tree_model": (params.tree_rows_per_feature, params.tree_min_rows),
        "general_ml": (params.general_rows_per_feature, params.general_min_rows),
    }
    rules: dict[str, RuleResult] = {}
    for key, label, severity in _RULES:
        per_feature, floor = thresholds[key]
        required = max(features * per_feature, floor)
        rules[key] = RuleResult(
            label=label,
            required_rows=required,
            adequate=rows >= required,
            shortfall=max(required - rows, 0),
        )
        if rules[key].adequate:
            continue
        shortfall = rules[key].shortfall
        ctx.finding(
            code=f"too_few_rows_{key}",
            subject=Subject.dataset(),
            severity=severity,
            confidence=Confidence.MODERATE,
            title=f"Too few rows for {label} ({rows:,} of ~{required:,})",
            message=(
                f"The dataset has {rows:,} rows and {features:,} features, but "
                f"{label} wants about {required:,} rows, {shortfall:,} more. "
                f"{_CAUSES[key].format(per=per_feature)} These are rules of thumb, "
                "not guarantees. Collect more data, drop low-signal or redundant "
                "columns, use a simpler model, or regularise strongly, and "
                "cross-validate with many folds to use every row."
            ),
            evidence=Evidence(
                metrics={
                    "rows": rows,
                    "n_features": features,
                    "required_rows": required,
                    "shortfall": shortfall,
                    "rows_per_feature": rows / features,
                },
                n_used=rows,
                n_total=rows,
                method="rows_per_feature_rule",
            ),
        )
    if all(r.adequate for r in rules.values()):
        verdict: Verdict = "adequate"
    elif rules["linear_model"].adequate or rules["tree_model"].adequate:
        verdict = "limited"
    else:
        verdict = "insufficient"
    return ctx.output(
        SampleSizeOutput(
            rows=rows,
            n_features=features,
            rows_per_feature=rows / features,
            verdict=verdict,
            rules=rules,
        )
    )
