# dsbf/checks/class_imbalance.py
"""A target column in which one class is nearly all of the rows."""

from pydantic import Field

from dsbf.checks.dominant_value import most_frequent
from dsbf.engine import (
    CheckContext,
    CheckOutput,
    CheckOutputModel,
    Requires,
    SemanticTypes,
    check,
)
from dsbf.model import (
    BooleanProfile,
    CategoricalProfile,
    CheckParams,
    CheckStatus,
    Confidence,
    Cost,
    Dimension,
    Evidence,
    Severity,
    Subject,
)
from dsbf.profiling.values import MAX_TOP_VALUE_CHARS


class ClassImbalanceParams(CheckParams):
    target: str | None = Field(
        None,
        description="The column to predict. `dsbf.profile(target=...)` sets it; "
        "without one the check is not applicable.",
    )
    majority_share: float = Field(
        0.90,
        gt=0,
        le=1,
        description="Share of the non-null target from which the majority class "
        "makes the target imbalanced.",
    )


class ClassImbalanceOutput(CheckOutputModel):
    target: str
    n_classes: int = Field(ge=1)
    majority_class: str
    majority_count: int = Field(ge=1)
    majority_share: float = Field(gt=0, le=1)
    minority_class: str | None = Field(
        description="The rarest class; None when the profile lists too few values "
        "to know it."
    )
    minority_count: int | None = Field(ge=1)
    imbalance_ratio: float | None = Field(
        ge=1, description="Majority count over minority count, when both are known."
    )
    imbalanced: bool = Field(description="The share is at least `majority_share`.")


@check(
    id="class_imbalance",
    version="1.1.0",
    kind="rule",
    applies_to=SemanticTypes("categorical", "boolean"),
    dimensions=[Dimension.DISTRIBUTION],
    params=ClassImbalanceParams,
    output=ClassImbalanceOutput,
    requires=Requires(profile=["counts"], params=["target"]),
    cost=Cost.CHEAP,
    codes=["class_imbalance"],
    packs=["core"],
    tags=["target", "imbalance", "ml-readiness"],
)
def class_imbalance(ctx: CheckContext, params: ClassImbalanceParams) -> CheckOutput:
    """Flag a target whose most frequent class is at least `majority_share` of it."""
    name = params.target
    if name is None:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_target",
            "No target column is set, so there is no class distribution to check.",
            severity=Severity.INFO,
            hint="Pass target=... to dsbf.profile(), or set the `target` parameter.",
        )
    if name not in ctx.columns:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "target_not_a_class",
            f"The target {name!r} is not a categorical or boolean column of the "
            "data, so it has no classes to count. A numeric column the profile "
            "types numeric is a regression target.",
            severity=Severity.INFO,
        )
    column = ctx.profile.column(name)
    assert isinstance(column, (CategoricalProfile, BooleanProfile))
    counts = column.counts
    top = most_frequent(column)
    if top is None or counts.non_null == 0:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_class_listed",
            f"The profile lists no value of the target {name!r}, so its majority "
            "class is unknown.",
            severity=Severity.INFO,
        )
    majority, majority_count = top
    share = majority_count / counts.non_null
    classes = _classes(column)
    minority = classes[-1] if classes and len(classes) == counts.distinct else None
    ratio = majority_count / minority[1] if minority else None
    imbalanced = share >= params.majority_share
    if imbalanced:
        ctx.finding(
            code="class_imbalance",
            subject=Subject.column(name),
            severity=Severity.WARN,
            confidence=Confidence.HIGH,
            title=f"Class imbalance: {majority!r} is {share:.1%} of the target",
            message=(
                f"{majority!r} is {majority_count:,} of the {counts.non_null:,} "
                f"non-null values of the target {name!r} ({share:.1%}), among "
                f"{counts.distinct:,} classes. A classifier that always predicts it "
                f"scores {share:.1%} accuracy without learning anything. Check "
                "whether the imbalance reflects the population or an artefact of "
                "how the data was collected, filtered or labelled; then judge a "
                "model by precision, recall, F1 or AUC-PR, and weight or resample "
                "the training split only."
            ),
            evidence=Evidence(
                metrics={
                    "majority_share": share,
                    "majority_class": majority,
                    "majority_count": majority_count,
                    "n_classes": counts.distinct,
                    "threshold": params.majority_share,
                    "imbalance_ratio": ratio,
                },
                n_used=counts.non_null,
                n_total=counts.rows,
                method="majority_class_share",
            ),
        )
    return ctx.output(
        ClassImbalanceOutput(
            target=name,
            n_classes=max(counts.distinct, 1),
            majority_class=majority,
            majority_count=majority_count,
            majority_share=share,
            minority_class=minority[0] if minority else None,
            minority_count=minority[1] if minority else None,
            imbalance_ratio=ratio,
            imbalanced=imbalanced,
        )
    )


def _classes(column: CategoricalProfile | BooleanProfile) -> list[tuple[str, int]]:
    """The classes the profile lists, most frequent first, as text with counts."""
    if isinstance(column, BooleanProfile):
        listed = [("True", column.true_count), ("False", column.false_count)]
        return sorted((c for c in listed if c[1] > 0), key=lambda c: -c[1])
    classes = [(str(entry.value), entry.count) for entry in column.top_values]
    mode = column.mode_count
    if mode is not None and (not classes or classes[0][1] < mode):
        # The most frequent class is a string too long to list.
        classes.insert(0, (f"(a value of over {MAX_TOP_VALUE_CHARS} characters)", mode))
    return classes
