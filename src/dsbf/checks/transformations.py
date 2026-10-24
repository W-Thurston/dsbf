# dsbf/checks/transformations.py
"""Transforms that reduce a numeric column's skew, with the reduction measured."""

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from pydantic import Field
from scipy import stats

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
    Severity,
    Subject,
)

Method = Literal["log1p", "sqrt", "square", "box_cox", "yeo_johnson"]

# The simplest first: the check suggests the first that reaches the target skew.
# These are the `method`s of the `transform` actions this check emits.
TRANSFORM_METHODS: tuple[Method, ...] = (
    "log1p",
    "sqrt",
    "square",
    "box_cox",
    "yeo_johnson",
)


class TransformationsParams(CheckParams):
    skew_threshold: float = Field(
        1.0, gt=0, description="Absolute skewness from which a column is previewed."
    )
    target_skew: float = Field(
        0.5,
        ge=0,
        description="Absolute skewness after the transform at which the simplest "
        "transform that gets there is suggested.",
    )
    min_reduction: float = Field(
        0.25,
        gt=0,
        le=1,
        description="Share of the absolute skewness a transform must remove to be "
        "suggested when none reaches the target.",
    )
    min_n: int = Field(10, ge=4, description="Finite values a column needs.")
    max_rows: int = Field(
        20_000,
        ge=100,
        description="Most values transformed; a larger column is sampled with `seed`.",
    )
    seed: int = Field(42, description="Seed of the row sample.")


class Candidate(Contract):
    skewness_after: float | None = Field(
        description="Skewness of the transformed values; None when it was not tried."
    )
    reduction: float | None = Field(
        description="Share of the absolute skewness the transform removes."
    )
    lam: float | None = Field(
        description="The fitted power for Box–Cox and Yeo–Johnson."
    )
    skipped: str | None = Field(description="Why the transform was not tried or used.")


class ColumnTransforms(Contract):
    skewness_before: float
    n_used: int = Field(ge=1)
    sampled: bool
    recommended: Method | None
    candidates: dict[str, Candidate]


class TransformationsOutput(CheckOutputModel):
    columns: dict[str, ColumnTransforms] = Field(
        description="Each skewed numeric column that was tried, by name."
    )
    skipped_columns: dict[str, str] = Field(
        description="Skewed columns left out, with the reason."
    )


def _skew(values: np.ndarray) -> float | None:
    with np.errstate(all="ignore"):
        found = float(stats.skew(values, bias=False))
    return found if math.isfinite(found) else None


# The skewness a transform leaves is close to 0 by design, so the digits past the
# fourth are rounding noise that differs between platforms; a report keeps four.
REPORTED_DECIMALS = 4


def _reported(value: float) -> float:
    return round(value, REPORTED_DECIMALS)


def _fit_box_cox(x: np.ndarray) -> tuple[np.ndarray, float]:
    transformed, lam = stats.boxcox(x)
    return transformed, float(lam)


def _fit_yeo_johnson(x: np.ndarray) -> tuple[np.ndarray, float]:
    transformed, lam = stats.yeojohnson(x)
    return transformed, float(lam)


def _fixed(function: Callable[[np.ndarray], np.ndarray]):
    return lambda x: (function(x), None)


# Method → (precondition, why the precondition is needed, transform).
_TRANSFORMS: dict[
    Method,
    tuple[
        Callable[[np.ndarray, float], bool],
        str,
        Callable[[np.ndarray], tuple[np.ndarray, float | None]],
    ],
] = {
    "log1p": (
        lambda x, skew: x.min() >= 0 and skew > 0,
        "needs values of 0 or more and a right skew",
        _fixed(np.log1p),
    ),
    "sqrt": (
        lambda x, skew: x.min() >= 0 and skew > 0,
        "needs values of 0 or more and a right skew",
        _fixed(np.sqrt),
    ),
    "square": (
        lambda x, skew: x.min() >= 0 and skew < 0,
        "needs values of 0 or more (below 0, x² folds the distribution) and a "
        "left skew",
        _fixed(np.square),
    ),
    "box_cox": (
        lambda x, skew: x.min() > 0,
        "needs values above 0",
        _fit_box_cox,
    ),
    "yeo_johnson": (lambda x, skew: True, "", _fit_yeo_johnson),
}


def _try(
    method: Method, x: np.ndarray, before: float
) -> tuple[Candidate, np.ndarray | None]:
    precondition, why, apply = _TRANSFORMS[method]
    if not precondition(x, before):
        return Candidate(
            skewness_after=None, reduction=None, lam=None, skipped=why
        ), None
    try:
        with np.errstate(all="ignore"):
            transformed, lam = apply(x)
    except (ValueError, FloatingPointError, OverflowError, RuntimeError) as error:
        return Candidate(
            skewness_after=None,
            reduction=None,
            lam=None,
            skipped=f"the fit failed: {error}",
        ), None
    if not np.isfinite(transformed).all():
        return Candidate(
            skewness_after=None,
            reduction=None,
            lam=None,
            skipped="the result has values that are not finite",
        ), None
    after = _skew(transformed)
    if after is None:
        return Candidate(
            skewness_after=None,
            reduction=None,
            lam=lam,
            skipped="the result has no skewness (it is constant)",
        ), None
    return Candidate(
        skewness_after=_reported(after),
        reduction=_reported((abs(before) - abs(after)) / abs(before)),
        lam=lam,
        skipped=None,
    ), transformed


def _choose(
    tried: dict[Method, Candidate], params: TransformationsParams
) -> Method | None:
    usable = {m: c for m, c in tried.items() if c.skipped is None}
    for method, candidate in usable.items():
        if abs(candidate.skewness_after) <= params.target_skew:  # type: ignore[arg-type]
            return method
    if not usable:
        return None
    best = max(usable, key=lambda m: usable[m].reduction)  # type: ignore[arg-type,return-value]
    if usable[best].reduction >= params.min_reduction:  # type: ignore[operator]
        return best
    return None


@check(
    id="transformations",
    version="1.0.1",
    kind="analysis",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.DISTRIBUTION],
    params=TransformationsParams,
    output=TransformationsOutput,
    requires=Requires(profile=["counts", "numeric_stats"]),
    cost=Cost.MODERATE,
    packs=["core"],
    tags=["transform", "skew", "ml"],
)
def transformations(ctx: CheckContext, params: TransformationsParams) -> CheckOutput:
    """Try log, root, square, Box–Cox and Yeo–Johnson on each skewed column."""
    columns: dict[str, ColumnTransforms] = {}
    skipped: dict[str, str] = {}
    rng = np.random.default_rng(params.seed)
    for column in ctx.profile.columns:
        if not isinstance(column, NumericProfile) or column.stats is None:
            continue
        skew = column.stats.skewness
        if skew is None or abs(skew) < params.skew_threshold:
            continue
        ctx.checkpoint()
        x = np.asarray(ctx.values(column.name), dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) < params.min_n:
            skipped[column.name] = f"{len(x)} finite values, fewer than {params.min_n}"
            continue
        sampled = len(x) > params.max_rows
        if sampled:
            x = x[rng.choice(len(x), params.max_rows, replace=False)]
        before = _skew(x)
        if before is None or abs(before) < 1e-12:
            skipped[column.name] = "the values have no skewness (they are constant)"
            continue
        tried: dict[Method, Candidate] = {}
        chosen: Method | None = None
        for method in TRANSFORM_METHODS:
            ctx.checkpoint()
            if chosen is not None:
                tried[method] = Candidate(
                    skewness_after=None,
                    reduction=None,
                    lam=None,
                    skipped=f"not needed: {chosen} already reaches the target",
                )
                continue
            tried[method], _ = _try(method, x, before)
            usable = tried[method].skipped is None
            if usable and abs(tried[method].skewness_after) <= params.target_skew:  # type: ignore[arg-type]
                chosen = method
        recommended = _choose(tried, params)
        before = _reported(before)
        columns[column.name] = ColumnTransforms(
            skewness_before=before,
            n_used=len(x),
            sampled=sampled,
            recommended=recommended,
            candidates={m: c for m, c in tried.items()},
        )
        used = {"n_used": len(x), "n_total": column.counts.rows}
        confidence = Confidence.MODERATE if sampled else Confidence.HIGH
        if recommended is None:
            ctx.finding(
                code="no_effective_transform",
                subject=Subject.column(column.name),
                severity=Severity.INFO,
                confidence=confidence,
                title=f"No transform removes the skew ({before:+.2f})",
                message=(
                    f"{column.name!r} has skewness {before:.2f}, and no log, root, "
                    "power or Yeo–Johnson transform removes at least "
                    f"{params.min_reduction:.0%} of it. A few extreme values or a "
                    "second mode, not a smooth tail, usually cause that: look at "
                    "the outliers and the histogram before transforming."
                ),
                evidence=Evidence(
                    metrics={"skewness_before": before},
                    method="skew_reduction",
                    **used,
                ),
            )
            continue
        best = tried[recommended]
        lam = best.lam
        fitted = lam is not None
        ctx.finding(
            code="transform_suggested",
            subject=Subject.column(column.name),
            severity=Severity.INFO,
            confidence=confidence,
            title=f"Apply {recommended} (skew {before:+.2f} → "
            f"{best.skewness_after:+.2f})",
            message=(
                f"{column.name!r} has skewness {before:.2f}. {recommended} brings "
                f"it to {best.skewness_after:.2f}, removing {best.reduction:.0%} of "
                "it"
                + (f" (fitted power {lam:.3g})" if fitted else "")
                + ". Linear, regularised and distance-based models gain from a "
                "more symmetric column; tree-based models are unchanged by it."
            ),
            evidence=Evidence(
                metrics={
                    "skewness_before": before,
                    "skewness_after": best.skewness_after,
                    "reduction": best.reduction,
                    "transform": recommended,
                    "lambda": lam,
                },
                method="skew_reduction",
                **used,
            ),
            guidance=Guidance(
                phase="ml",
                actions=(
                    Action(
                        action="transform",
                        columns=(column.name,),
                        method=recommended,
                        condition=_TRANSFORMS[recommended][1] or None,
                        detail=(
                            "Fit the power on the training split and apply it to "
                            "the rest."
                            if fitted
                            else f"Apply {recommended} to the values."
                        ),
                        params={"lambda": lam} if fitted else {},
                    ),
                ),
            ),
        )
    return ctx.output(TransformationsOutput(columns=columns, skipped_columns=skipped))
