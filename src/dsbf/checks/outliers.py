# dsbf/checks/outliers.py
"""Numeric values far from the rest of their column, by three univariate rules."""

from typing import Literal

import numpy as np
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
    CheckStatus,
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    Severity,
    Subject,
)

Method = Literal["iqr", "zscore", "mad"]
METHODS: tuple[Method, ...] = ("iqr", "zscore", "mad")
LABELS = {"iqr": "IQR", "zscore": "z-score", "mad": "MAD"}

# Scales a deviation from the median, in MADs, to a standard-normal z-score.
_MAD_SCALE = 0.6745
_TOP_VALUES = 10


class OutliersParams(CheckParams):
    methods: tuple[Method, ...] = Field(
        METHODS,
        min_length=1,
        description="The rules to apply. Consensus needs at least two of them.",
    )
    iqr_multiplier: float = Field(
        1.5, gt=0, description="Tukey's fence distance, in interquartile ranges."
    )
    zscore_threshold: float = Field(
        3.0, gt=0, description="|z| above which a value is flagged."
    )
    mad_threshold: float = Field(
        3.5,
        gt=0,
        description="Modified z-score (Iglewicz and Hoaglin, 1993) above which a "
        "value is flagged.",
    )
    min_n: int = Field(
        10, ge=4, description="Finite values a column needs to be examined."
    )
    warn_share: float = Field(
        0.05,
        gt=0,
        le=1,
        description="Share of the values one rule flags from which the finding is a "
        "warning even without consensus.",
    )


class IqrResult(Contract):
    count: int = Field(ge=0)
    share: float = Field(ge=0, le=1, description="Of the finite values.")
    q1: float
    q3: float
    lower_fence: float
    upper_fence: float


class ZScoreResult(Contract):
    count: int = Field(ge=0)
    share: float = Field(ge=0, le=1, description="Of the finite values.")
    mean: float
    std: float
    threshold: float
    max_score: float = Field(ge=0, description="The largest |z|.")


class MadResult(Contract):
    count: int = Field(ge=0)
    share: float = Field(ge=0, le=1, description="Of the finite values.")
    median: float
    mad: float = Field(gt=0, description="Median absolute deviation from the median.")
    threshold: float
    max_score: float = Field(ge=0, description="The largest modified z-score.")


class ColumnOutliers(Contract):
    n_used: int = Field(ge=0, description="Finite values examined.")
    iqr: IqrResult | None
    zscore: ZScoreResult | None = Field(
        description="None when the column has no spread (std is 0)."
    )
    mad: MadResult | None = Field(
        description="None when more than half the values equal the median (MAD is 0)."
    )
    methods_flagging: tuple[Method, ...] = Field(
        description="The rules that flag at least one value."
    )
    consensus: bool = Field(description="At least two rules flag the column.")
    top_values: tuple[float, ...] = Field(
        description="Up to ten flagged values farthest from the median."
    )


class OutliersOutput(CheckOutputModel):
    columns: dict[str, ColumnOutliers] = Field(
        description="Every numeric column with enough values, by name."
    )
    skipped: dict[str, str] = Field(
        description="Columns not examined, with the reason."
    )


@check(
    id="outliers",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.DISTRIBUTION],
    params=OutliersParams,
    output=OutliersOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.MODERATE,
    packs=["core"],
    tags=["outliers", "robust", "mad"],
)
def outliers(ctx: CheckContext, params: OutliersParams) -> CheckOutput:
    """Flag numeric values beyond the IQR fences, a z-score or a modified z-score."""
    columns: dict[str, ColumnOutliers] = {}
    skipped: dict[str, str] = {}
    for name in ctx.columns:
        raw = ctx.values(name)
        values = raw[np.isfinite(raw)].astype(np.float64, copy=False)
        if values.size < params.min_n:
            skipped[name] = f"fewer than {params.min_n} finite values"
            continue
        with np.errstate(all="ignore"):
            result = _examine(values, params)
        if result is None:
            skipped[name] = "values too large to compute with"
            continue
        columns[name] = result
        if result.methods_flagging:
            _report(ctx, name, result, params)
    if not columns:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_column_examined",
            f"No numeric column has {params.min_n} finite values.",
            severity=Severity.INFO,
        )
    return ctx.output(OutliersOutput(columns=columns, skipped=skipped))


def _examine(values: np.ndarray, params: OutliersParams) -> ColumnOutliers | None:
    """The rules applied to one column; None when its numbers overflow."""
    n = int(values.size)
    flagged = np.zeros(n, dtype=bool)
    iqr = zscore = mad = None
    median = float(np.median(values))
    if "iqr" in params.methods:
        q1, q3 = (float(q) for q in np.percentile(values, [25, 75]))
        spread = q3 - q1
        lower = q1 - params.iqr_multiplier * spread
        upper = q3 + params.iqr_multiplier * spread
        if not np.isfinite([q1, q3, lower, upper]).all():
            return None
        mask = (values < lower) | (values > upper)
        flagged |= mask
        count = int(np.count_nonzero(mask))
        iqr = IqrResult(
            count=count,
            share=count / n,
            q1=q1,
            q3=q3,
            lower_fence=lower,
            upper_fence=upper,
        )
    if "zscore" in params.methods:
        std = float(np.std(values, ddof=1))
        if std > 0:
            mean = float(np.mean(values))
            scores = np.abs(values - mean) / std
            if not np.isfinite([mean, std, scores.max()]).all():
                return None
            mask = scores > params.zscore_threshold
            flagged |= mask
            count = int(np.count_nonzero(mask))
            zscore = ZScoreResult(
                count=count,
                share=count / n,
                mean=mean,
                std=std,
                threshold=params.zscore_threshold,
                max_score=float(scores.max()),
            )
    if "mad" in params.methods:
        deviation = np.abs(values - median)
        spread = float(np.median(deviation))
        if spread > 0:
            scores = _MAD_SCALE * deviation / spread
            if not np.isfinite([median, spread, scores.max()]).all():
                return None
            mask = scores > params.mad_threshold
            flagged |= mask
            count = int(np.count_nonzero(mask))
            mad = MadResult(
                count=count,
                share=count / n,
                median=median,
                mad=spread,
                threshold=params.mad_threshold,
                max_score=float(scores.max()),
            )
    flagging = tuple(
        method
        for method, result in zip(METHODS, (iqr, zscore, mad), strict=True)
        if result is not None and result.count > 0
    )
    return ColumnOutliers(
        n_used=n,
        iqr=iqr,
        zscore=zscore,
        mad=mad,
        methods_flagging=flagging,
        consensus=len(flagging) >= 2,
        top_values=_farthest(values, flagged, median),
    )


def _farthest(
    values: np.ndarray, flagged: np.ndarray, median: float
) -> tuple[float, ...]:
    """Up to ten of the flagged values, the farthest from the median first."""
    candidates = values[flagged]
    if not candidates.size:
        return ()
    distance = np.abs(candidates - median)
    take = min(_TOP_VALUES, candidates.size)
    order = np.argsort(-distance, kind="stable")[:take]
    return tuple(float(candidates[i]) for i in order)


def _report(
    ctx: CheckContext, name: str, result: ColumnOutliers, params: OutliersParams
) -> None:
    results = {"iqr": result.iqr, "zscore": result.zscore, "mad": result.mad}
    share = max(r.share for r in results.values() if r is not None)
    severe = result.consensus or share >= params.warn_share
    parts = [
        f"{LABELS[m]}: {results[m].count:,} ({results[m].share:.1%})"  # type: ignore[union-attr]
        for m in result.methods_flagging
    ]
    verdict = (
        f"flagged by {len(result.methods_flagging)} of {len(params.methods)} rules, "
        "so likely real"
        if result.consensus
        else "flagged by one rule only; verify before acting"
    )
    metrics: dict[str, object] = {
        "outlier_share": share,
        "methods_flagging": result.methods_flagging,
        "consensus": result.consensus,
        "top_values": result.top_values,
    }
    for method, found in results.items():
        if found is not None:
            metrics[f"{method}_count"] = found.count
    if result.iqr is not None:
        metrics["lower_fence"] = result.iqr.lower_fence
        metrics["upper_fence"] = result.iqr.upper_fence
    ctx.finding(
        code="outliers_detected",
        subject=Subject.column(name),
        severity=Severity.WARN if severe else Severity.INFO,
        confidence=Confidence.HIGH if result.consensus else Confidence.MODERATE,
        title=(
            f"Outliers: {len(result.methods_flagging)} rules agree"
            if result.consensus
            else "Outliers: one rule flags values"
        ),
        message=(
            f"{name!r} has values far from the rest: {'; '.join(parts)} — "
            f"{verdict}. Find out whether the extremes are real rare events, entry "
            "errors or a separate group before deciding what to do with them."
        ),
        evidence=Evidence(
            metrics=metrics,  # type: ignore[arg-type]
            n_used=result.n_used,
            n_total=ctx.profile.n_rows,
            method="+".join(result.methods_flagging),
        ),
    )
