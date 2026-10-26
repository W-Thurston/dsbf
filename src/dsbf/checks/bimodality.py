# dsbf/checks/bimodality.py
"""Numeric columns whose values form two separate peaks."""

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
    NumericProfile,
    Severity,
    Subject,
)

# Ashman's D from which the two components are cleanly apart: they overlap by under
# about 7% of their mass. Below 2 the pair is one skewed or heavy-tailed peak.
_CLEAN_SEPARATION = 3.0
# Points at which the density is read between and beyond the component means.
_GRID = 200


class BimodalityParams(CheckParams):
    bic_threshold: float = Field(
        10.0,
        ge=0,
        description="Improvement in BIC from a second component above which it is "
        "worth having. Raftery's scale calls 10 'very strong'.",
    )
    min_separation: float = Field(
        2.0,
        gt=0,
        description="Ashman's D between the components, from which they are two "
        "peaks and not one skewed or heavy-tailed one.",
    )
    max_valley_ratio: float = Field(
        0.75,
        gt=0,
        le=1,
        description="Most the histogram may dip between the peaks, as the density "
        "at the valley over that at the lower peak. A flat or one-humped column "
        "has a ratio near 1, and a mixture fits it as two Gaussians all the same.",
    )
    min_weight: float = Field(
        0.05,
        gt=0,
        lt=0.5,
        description="Share of the values the smaller component must hold.",
    )
    strong_bic_per_value: float = Field(
        0.05,
        gt=0,
        description="Improvement in BIC per value from which, with a clean "
        "separation, the finding is a warning.",
    )
    skewness_guard: float = Field(
        2.0,
        gt=0,
        description="|skewness| above which a column is not fitted: a long tail "
        "looks like a second component.",
    )
    min_n: int = Field(
        10,
        ge=4,
        description="Finite values a column needs. Below about 100 the BIC rarely "
        "separates two peaks, even 4 spreads apart.",
    )
    max_rows: int = Field(
        10_000,
        ge=1000,
        description="Most values fitted; a larger column is sampled with `seed`.",
    )
    seed: int = Field(42, description="Seed of the mixture fits and of the sample.")


class ColumnBimodality(Contract):
    n_used: int = Field(ge=0, description="Values fitted, after any sampling.")
    n_finite: int = Field(ge=0, description="Finite values in the column.")
    bic_1_component: float
    bic_2_components: float
    bic_delta: float = Field(
        description="BIC of one component minus BIC of two; the larger, the better "
        "two fit. Unlike either BIC, it doesn't depend on the likelihood's "
        "convention."
    )
    separation: float = Field(
        ge=0, description="Ashman's D: the gap between the means over their spread."
    )
    means: tuple[float, float] = Field(description="Component means, low first.")
    stds: tuple[float, float] = Field(description="Component std deviations.")
    weights: tuple[float, float] = Field(
        description="Component shares, in the order of the means."
    )
    valley_ratio: float = Field(
        ge=0,
        description="Density of the histogram at its lowest point between the "
        "component means over that at the lower of the two peaks; 1 when there is "
        "no dip.",
    )
    bimodal: bool


class BimodalityOutput(CheckOutputModel):
    columns: dict[str, ColumnBimodality] = Field(
        description="Every numeric column that was fitted, by name."
    )
    skipped: dict[str, str] = Field(description="Columns not fitted, with the reason.")


@check(
    id="bimodality",
    version="1.1.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.DISTRIBUTION],
    params=BimodalityParams,
    output=BimodalityOutput,
    requires=Requires(profile=["counts", "numeric_stats"]),
    cost=Cost.MODERATE,
    extras=["ml"],
    packs=["core"],
    tags=["distribution", "bimodal", "mixture"],
)
def bimodality(ctx: CheckContext, params: BimodalityParams) -> CheckOutput:
    """Flag numeric columns that two Gaussian peaks fit clearly better than one.

    Peaks 4 spreads apart are found in over 80% of columns of 100 values and in all
    of 200 or more; 3 spreads apart, only in about half of 1 000. Under about 100
    values the BIC cannot tell two peaks from one, so a column that short is rarely
    flagged, and a uniform column of that size is flagged by chance in about 7% of
    samples (under 3% from 200 values).
    """
    from sklearn.mixture import GaussianMixture  # the `ml` extra

    rng = np.random.default_rng(params.seed)
    columns: dict[str, ColumnBimodality] = {}
    skipped: dict[str, str] = {}
    for column in ctx.profile.columns:
        if not isinstance(column, NumericProfile):
            continue
        name = column.name
        raw = ctx.values(name)
        values = raw[np.isfinite(raw)].astype(np.float64, copy=False)
        if values.size < params.min_n:
            skipped[name] = f"fewer than {params.min_n} finite values"
            continue
        if np.ptp(values) == 0:
            skipped[name] = "constant"
            continue
        skewness = column.stats.skewness if column.stats is not None else None
        if skewness is not None and abs(skewness) > params.skewness_guard:
            skipped[name] = f"skewness {skewness:.2f} is beyond the guard"
            continue
        if values.size > params.max_rows:
            keep = np.sort(rng.choice(values.size, params.max_rows, replace=False))
            sample = values[keep]
        else:
            sample = values
        data = sample.reshape(-1, 1)
        try:
            with np.errstate(all="ignore"):
                one = GaussianMixture(1, random_state=params.seed).fit(data)
                two = GaussianMixture(2, random_state=params.seed).fit(data)
                result = _summarise(one, two, data, int(values.size), params)
        except ValueError as error:
            skipped[name] = "the mixture could not be fitted"
            ctx.note(
                "mixture_fit_failed",
                f"No mixture was fitted to {name!r}: {error}",
                severity=Severity.WARN,
                exception_type=type(error).__name__,
            )
            continue
        if result is None:
            skipped[name] = "values too large to compute with"
            continue
        columns[name] = result
        if result.bimodal:
            _report(ctx, name, result, params)
    if not columns:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_column_fitted",
            "No numeric column could be fitted: " + _why(skipped),
            severity=Severity.INFO,
        )
    return ctx.output(BimodalityOutput(columns=columns, skipped=skipped))


def _valley_ratio(values: np.ndarray, low: float, high: float) -> float:
    """How far the density dips between two peaks, against the lower of them.

    A mixture of two Gaussians fits a flat column as well as a bimodal one, so the
    data has to show the dip too. The density is a Gaussian kernel estimate with
    Silverman's bandwidth, which a histogram of a few hundred values can't match:
    its few bins put the dip between two bins or smooth it away. The valley is the
    lowest density between the component means, and each peak the highest density on
    its side of it.
    """
    spread = float(np.std(values))
    q1, q3 = np.percentile(values, [25, 75])
    scale = min(spread, float(q3 - q1) / 1.349) or spread
    bandwidth = 0.9 * scale * values.size**-0.2
    if bandwidth <= 0:
        return 1.0
    grid = np.linspace(low - 2 * bandwidth, high + 2 * bandwidth, _GRID)
    # The estimate in blocks, so the grid by values matrix stays small.
    density = np.empty(grid.size)
    for start in range(0, grid.size, 64):
        block = grid[start : start + 64, None]
        density[start : start + 64] = np.exp(
            -0.5 * ((block - values[None, :]) / bandwidth) ** 2
        ).sum(axis=1)
    between = (grid >= low) & (grid <= high)
    if not between.any():
        return 1.0
    valley_at = int(np.flatnonzero(between)[np.argmin(density[between])])
    peaks = min(density[: valley_at + 1].max(), density[valley_at:].max())
    if peaks <= 0:
        return 1.0
    return float(density[valley_at] / peaks)


def _why(skipped: dict[str, str]) -> str:
    return "; ".join(sorted(set(skipped.values()))) or "there are none"


def _summarise(
    one, two, data: np.ndarray, n_finite: int, params: BimodalityParams
) -> ColumnBimodality | None:
    """The comparison of the two fits; None when the numbers overflow."""
    bic_one, bic_two = float(one.bic(data)), float(two.bic(data))
    order = np.argsort(two.means_.ravel())
    means = two.means_.ravel()[order]
    stds = np.sqrt(two.covariances_.reshape(-1)[order])
    weights = two.weights_[order]
    spread = float(np.sqrt(stds[0] ** 2 + stds[1] ** 2))
    gap = abs(means[1] - means[0])
    separation = float(np.sqrt(2) * gap / spread) if spread else 0.0
    delta = bic_one - bic_two
    valley = _valley_ratio(data.ravel(), float(means[0]), float(means[1]))
    numbers = [bic_one, bic_two, delta, separation, valley, *means, *stds, *weights]
    if not np.isfinite(numbers).all():
        return None
    return ColumnBimodality(
        n_used=int(data.shape[0]),
        n_finite=n_finite,
        bic_1_component=bic_one,
        bic_2_components=bic_two,
        bic_delta=delta,
        separation=separation,
        means=(float(means[0]), float(means[1])),
        stds=(float(stds[0]), float(stds[1])),
        weights=(float(weights[0]), float(weights[1])),
        valley_ratio=valley,
        bimodal=bool(
            delta > params.bic_threshold
            and separation >= params.min_separation
            and weights.min() >= params.min_weight
            and valley <= params.max_valley_ratio
        ),
    )


def _report(
    ctx: CheckContext, name: str, result: ColumnBimodality, params: BimodalityParams
) -> None:
    per_value = result.bic_delta / result.n_used
    strong = (
        result.separation >= _CLEAN_SEPARATION
        and per_value >= params.strong_bic_per_value
    )
    low, high = result.means
    ctx.finding(
        code="bimodal",
        subject=Subject.column(name),
        severity=Severity.WARN if strong else Severity.INFO,
        confidence=Confidence.HIGH if strong else Confidence.MODERATE,
        title=f"Bimodal distribution (peaks near {low:.4g} and {high:.4g})",
        message=(
            f"{name!r} fits two Gaussian peaks, near {low:.4g} "
            f"({result.weights[0]:.0%} of values) and {high:.4g} "
            f"({result.weights[1]:.0%}), better than one: BIC improves by "
            f"{result.bic_delta:.1f} and the peaks are {result.separation:.1f} "
            "spreads apart. Two peaks often mean two populations, such as two "
            "instruments, seasons or groups; look for a column that splits them."
        ),
        evidence=Evidence(
            metrics={
                "bic_1_component": result.bic_1_component,
                "bic_2_components": result.bic_2_components,
                "bic_delta": result.bic_delta,
                "bic_delta_per_value": per_value,
                "separation": result.separation,
                "valley_ratio": result.valley_ratio,
                "mean_low": low,
                "mean_high": high,
                "weight_low": result.weights[0],
                "weight_high": result.weights[1],
                "n_finite": result.n_finite,
            },
            n_used=result.n_used,
            n_total=ctx.profile.n_rows,
            method="gmm_bic",
        ),
    )
