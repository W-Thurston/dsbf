# dsbf/checks/normality.py
"""Numeric columns whose values are distinguishable from a normal distribution."""

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

Pattern = Literal[
    "heavy_tails", "right_tail", "left_tail", "close_to_normal", "moderate_deviation"
]

# Why a column is not tested when its values overflow the arithmetic of the tests.
OVERFLOW = "values too large to compute with"

# A tail counts as deviating when it is this many times the average deviation.
_TAIL_RATIO = 1.5
# Under this, in standard deviations, no point of the Q-Q plot is far from the line.
_CLOSE = 0.1

_PATTERN_TEXT: dict[Pattern, str] = {
    "heavy_tails": "both tails deviate, as with heavy tails (high kurtosis)",
    "right_tail": "the upper tail deviates, as with right skew or high outliers",
    "left_tail": "the lower tail deviates, as with left skew or low outliers",
    "close_to_normal": "no quantile is far from a normal's",
    "moderate_deviation": "the quantiles deviate across the range",
}


class NormalityParams(CheckParams):
    alpha: float = Field(
        0.05,
        gt=0,
        lt=1,
        description="Chance that a normal column is rejected. Two tests are run, so "
        "each is held to half of it.",
    )
    min_deviation: float = Field(
        0.05,
        ge=0,
        description="Average distance of the column's quantiles from a normal's, in "
        "standard deviations, below which a rejection is not reported. At thousands "
        "of rows the tests reject deviations too small to matter: a t distribution "
        "with 30 degrees of freedom is 0.02 away, a gamma with skew 0.3 is 0.05.",
    )
    min_n: int = Field(
        8, ge=4, description="Finite values a column needs to be tested."
    )
    shapiro_max_n: int = Field(
        5000,
        ge=20,
        description="Largest sample for Shapiro-Wilk, which rejects trivial "
        "deviations above it; larger columns get a Kolmogorov-Smirnov test.",
    )
    qq_points: int = Field(
        100, ge=10, le=1000, description="Quantile pairs kept for a Q-Q plot."
    )
    qq_all_columns: bool = Field(
        False,
        description="Keep Q-Q quantiles for the columns that pass as well as the "
        "ones that fail.",
    )


class NormalityTest(Contract):
    test: Literal["shapiro_wilk", "ks_fitted_normal", "jarque_bera"]
    statistic: float
    p_value: float = Field(ge=0, le=1)
    rejects: bool = Field(description="p is below its share of `alpha`.")


class QQData(Contract):
    """Quantiles of the column against a normal fitted to it, ready to plot."""

    theoretical: tuple[float, ...] = Field(
        description="Normal quantiles, in the column's units."
    )
    empirical: tuple[float, ...] = Field(
        description="The column's quantiles at the same probabilities."
    )
    reference_x: tuple[float, float] = Field(
        description="Theoretical Q1 and Q3: the x of the reference line."
    )
    reference_y: tuple[float, float] = Field(
        description="Empirical Q1 and Q3: the y of the reference line."
    )
    fit_mean: float
    fit_std: float = Field(gt=0)
    mean_abs_deviation_in_std: float = Field(ge=0)
    max_deviation_in_std: float = Field(ge=0)
    lower_tail_deviation_in_std: float = Field(
        ge=0, description="Mean deviation over the lowest tenth of the points."
    )
    upper_tail_deviation_in_std: float = Field(
        ge=0, description="Mean deviation over the highest tenth of the points."
    )
    pattern: Pattern


class ColumnNormality(Contract):
    n_used: int = Field(ge=0, description="Finite values tested.")
    primary: NormalityTest = Field(
        description="Shapiro-Wilk up to `shapiro_max_n` values, Kolmogorov-Smirnov "
        "above it."
    )
    jarque_bera: NormalityTest
    rejected: bool = Field(description="Either test rejects normality.")
    mean_abs_deviation_in_std: float = Field(
        ge=0, description="Average Q-Q gap from a normal, in standard deviations."
    )
    non_normal: bool = Field(
        description="A test rejects normality and the gap is at least `min_deviation`."
    )
    qq: QQData | None = Field(
        description="None for a column that passes, unless `qq_all_columns` is set."
    )


class NormalityOutput(CheckOutputModel):
    columns: dict[str, ColumnNormality] = Field(
        description="Every numeric column that was tested, by name."
    )
    skipped: dict[str, str] = Field(description="Columns not tested, with the reason.")


@check(
    id="normality",
    version="1.1.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.DISTRIBUTION],
    params=NormalityParams,
    output=NormalityOutput,
    requires=Requires(profile=["counts", "numeric_stats"]),
    cost=Cost.MODERATE,
    codes=["non_normal"],
    packs=["core"],
    tags=["distribution", "normality", "qq"],
)
def normality(ctx: CheckContext, params: NormalityParams) -> CheckOutput:
    """Test numeric columns against a normal distribution and keep their Q-Q data."""
    columns: dict[str, ColumnNormality] = {}
    skipped: dict[str, str] = {}
    for name in ctx.columns:
        raw = ctx.values(name)
        values = raw[np.isfinite(raw)].astype(np.float64, copy=False)
        if values.size < params.min_n:
            skipped[name] = f"fewer than {params.min_n} finite values"
            continue
        if np.ptp(values) == 0:
            skipped[name] = "constant"
            continue
        result = _test(values, params)
        if result is None:
            skipped[name] = OVERFLOW
            continue
        columns[name] = result
        if result.non_normal:
            _report(ctx, name, result, params)
    if not columns:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_column_tested",
            f"No numeric column has {params.min_n} or more distinct finite values.",
            severity=Severity.INFO,
        )
    return ctx.output(NormalityOutput(columns=columns, skipped=skipped))


def _test(values: np.ndarray, params: NormalityParams) -> ColumnNormality | None:
    """The tests of one column; None when its values are too large to compute with."""
    n = int(values.size)
    with np.errstate(all="ignore"):
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
    if not (np.isfinite(mean) and np.isfinite(std) and std > 0):
        return None
    level = params.alpha / 2
    if n <= params.shapiro_max_n:
        statistic, p_value = stats.shapiro(values)
        primary = _result("shapiro_wilk", statistic, p_value, level)
    else:
        # The sample's own mean and std set the normal, so the p-value is
        # conservative (no Lilliefors correction). Standardising first keeps scipy
        # off the `args=` form of `kstest`, which 1.18 breaks.
        statistic, p_value = stats.kstest((values - mean) / std, "norm")[:2]
        primary = _result("ks_fitted_normal", statistic, p_value, level)
    statistic, p_value = stats.jarque_bera(values)
    jarque_bera = _result("jarque_bera", statistic, p_value, level)
    if primary is None or jarque_bera is None:
        return None
    rejected = primary.rejects or jarque_bera.rejects
    qq = _qq(values, mean, std, params.qq_points)
    if qq is None:
        return None
    non_normal = rejected and qq.mean_abs_deviation_in_std >= params.min_deviation
    return ColumnNormality(
        n_used=n,
        primary=primary,
        jarque_bera=jarque_bera,
        rejected=rejected,
        mean_abs_deviation_in_std=qq.mean_abs_deviation_in_std,
        non_normal=non_normal,
        qq=qq if non_normal or params.qq_all_columns else None,
    )


def _result(
    test: Literal["shapiro_wilk", "ks_fitted_normal", "jarque_bera"],
    statistic: float,
    p_value: float,
    alpha: float,
) -> NormalityTest | None:
    """The test's result, or None when scipy returned no number (overflow)."""
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        return None
    return NormalityTest(
        test=test,
        statistic=float(statistic),
        p_value=float(p_value),
        rejects=bool(p_value < alpha),
    )


def _qq(values: np.ndarray, mean: float, std: float, points: int) -> QQData | None:
    """Quantile pairs at evenly spaced probabilities, away from 0 and 1.

    None when the values are so large that the quantiles overflow.
    """
    n = values.size
    k = min(points, n)
    probabilities = np.linspace(1 / (n + 1), n / (n + 1), k)
    with np.errstate(all="ignore"):
        theoretical = stats.norm.ppf(probabilities, loc=mean, scale=std)
        empirical = np.quantile(values, probabilities)
        deviation = np.abs(empirical - theoretical) / std
    if not (np.isfinite(theoretical).all() and np.isfinite(deviation).all()):
        return None
    tail = max(1, k // 10)
    lower, upper = float(deviation[:tail].mean()), float(deviation[-tail:].mean())
    average, worst = float(deviation.mean()), float(deviation.max())
    edges = [0.25, 0.75]
    with np.errstate(all="ignore"):
        reference_y = np.quantile(values, edges)
    if not np.isfinite(reference_y).all():
        return None
    return QQData(
        theoretical=tuple(round(float(v), 6) for v in theoretical),
        empirical=tuple(round(float(v), 6) for v in empirical),
        reference_x=_pair(stats.norm.ppf(edges, loc=mean, scale=std)),
        reference_y=_pair(reference_y),
        fit_mean=mean,
        fit_std=std,
        mean_abs_deviation_in_std=average,
        max_deviation_in_std=worst,
        lower_tail_deviation_in_std=lower,
        upper_tail_deviation_in_std=upper,
        pattern=_pattern(average, worst, lower, upper),
    )


def _pair(values: np.ndarray) -> tuple[float, float]:
    return (round(float(values[0]), 6), round(float(values[1]), 6))


def _pattern(average: float, worst: float, lower: float, upper: float) -> Pattern:
    far = average * _TAIL_RATIO
    if lower > far and upper > far:
        return "heavy_tails"
    if upper > far:
        return "right_tail"
    if lower > far:
        return "left_tail"
    if worst < _CLOSE:
        return "close_to_normal"
    return "moderate_deviation"


def _report(
    ctx: CheckContext, name: str, result: ColumnNormality, params: NormalityParams
) -> None:
    rejecting = [t for t in (result.primary, result.jarque_bera) if t.rejects]
    tests = " and ".join(
        f"{t.test.replace('_', ' ').title()} (p={t.p_value:.4g})" for t in rejecting
    )
    qq = result.qq
    assert qq is not None  # a column that fails always keeps its Q-Q data
    metrics: dict[str, object] = {
        "alpha": params.alpha,
        "primary_test": result.primary.test,
        "primary_statistic": result.primary.statistic,
        "primary_p_value": result.primary.p_value,
        "jarque_bera_statistic": result.jarque_bera.statistic,
        "jarque_bera_p_value": result.jarque_bera.p_value,
        "mean_abs_deviation_in_std": qq.mean_abs_deviation_in_std,
        "qq_pattern": qq.pattern,
    }
    column = ctx.profile.column(name)
    if isinstance(column, NumericProfile) and column.stats is not None:
        # What the transform tips condition on: the sign of the values and the skew.
        metrics["skewness"] = column.stats.skewness
        metrics["min"] = column.stats.min
    ctx.finding(
        code="non_normal",
        subject=Subject.column(name),
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        title=f"Non-normal distribution ({rejecting[0].test.replace('_', ' ')})",
        message=(
            f"{name!r} (n={result.n_used:,}) rejects normality at α={params.alpha:g}: "
            f"{tests}. On its Q-Q plot {_PATTERN_TEXT[qq.pattern]}, with an average "
            f"gap of {qq.mean_abs_deviation_in_std:.2f} standard deviations from a "
            f"normal's quantiles, at least the {params.min_deviation:g} below which "
            "a rejection is too small to report. Models that assume normal inputs or "
            "residuals are affected; tree-based models are not."
        ),
        evidence=Evidence(
            metrics=metrics,  # type: ignore[arg-type]
            n_used=result.n_used,
            n_total=ctx.profile.n_rows,
            method=f"{result.primary.test}+jarque_bera",
        ),
    )
