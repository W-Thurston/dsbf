# dsbf/checks/multivariate_outliers.py
"""Rows that are unusual across the numeric columns together."""

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
    Cost,
    Dimension,
    Evidence,
    NumericProfile,
    Severity,
    Subject,
)

_MIN_FEATURES = 2
_MIN_ROWS = 10
_EXAMPLES = 20
# The fit that measures distances leaves out the rows beyond this share of a normal
# cloud, then corrects the spread it measured for the tail it cut off.
_TRIM = 0.975
_FIT_ROUNDS = 10
# Eigenvalues below this share of the largest are a column that repeats others.
_RANK_TOLERANCE = 1e-10


class MultivariateOutliersParams(CheckParams):
    contamination: float = Field(
        0.05,
        gt=0,
        le=0.5,
        description="Share of the rows the forest ranks as most anomalous. It flags "
        "exactly this share of whatever it is given, so the count is only a ranking; "
        "whether any row stands out is `alpha` and `min_excess`.",
    )
    alpha: float = Field(
        0.01,
        gt=0,
        lt=1,
        description="Chance that data whose columns are jointly normal after ranking "
        "has a row called far, over all its rows. A row is far when its robust "
        "Mahalanobis distance, taken on each column's normal scores (so a skewed or "
        "heavy-tailed column does not make rows far by itself), is beyond the "
        "chi-squared point that leaves `alpha / rows` above it.",
    )
    min_rows_per_column: int = Field(
        15,
        ge=2,
        description="Rows per column below which no row is called far: the spread "
        "of so few rows is too loose a ruler.",
    )
    min_excess: float = Field(
        0.05,
        ge=0,
        description="How far the forest's top anomaly score must be above the top "
        "score it gives a normal sample of the same shape for a row to stand out. "
        "Scores vary by 0.01 to 0.02 between samples of that shape (by up to 0.08 "
        "now and then at 1 000 rows, so about 5% of clean tables of that size get "
        "a low-confidence finding), and a median standing in for missing values "
        "lifts them by 0.02.",
    )
    n_estimators: int = Field(100, ge=10, le=1000, description="Trees in the forest.")
    min_n: int = Field(
        10, ge=4, description="Finite values a column needs to be a feature."
    )
    max_missing_share: float = Field(
        0.2,
        ge=0,
        le=1,
        description="Share of missing values above which a column is not a feature. "
        "A missing value takes the column's median, so a column that is mostly "
        "missing makes the rows that have a value look far from the rest.",
    )
    max_rows: int = Field(
        100_000,
        ge=1000,
        description="Most rows fitted; a larger frame is sampled with `seed`.",
    )
    seed: int = Field(42, description="Seed of the forest and of the sample.")


class MultivariateOutliersOutput(CheckOutputModel):
    feature_columns: tuple[str, ...] = Field(description="Columns the forest saw.")
    rows_used: int = Field(ge=0, description="Rows fitted, after any sampling.")
    rows_total: int = Field(ge=0)
    contamination: float
    n_flagged: int = Field(
        ge=0, description="Rows the forest ranks as most anomalous (`contamination`)."
    )
    far_rows: int = Field(
        ge=0,
        description="Rows whose robust Mahalanobis distance, on the rank-normalised "
        "columns, is beyond the limit, so far from the rest of the cloud together; 0 "
        "too when there are too few rows per column to say.",
    )
    far_limit: float | None = Field(
        description="That limit, a squared distance; None when it was not applied."
    )
    example_rows: tuple[int, ...] = Field(
        description="Positions in the data of the rows that stand out most: the far "
        "rows, then the forest's ranking."
    )
    max_anomaly_score: float = Field(
        description="Highest anomaly score, from 0 to 1; near 0.5 for all rows when "
        "none stands out, and approaching 1 for a clear anomaly."
    )
    reference_max_score: float = Field(
        description="The highest score the same forest gives a normal sample of the "
        "same rows, columns and correlations: what chance alone reaches."
    )
    excess: float = Field(description="`max_anomaly_score` less that reference.")


@check(
    id="multivariate_outliers",
    version="1.2.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.DISTRIBUTION],
    params=MultivariateOutliersParams,
    output=MultivariateOutliersOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.MODERATE,
    extras=["ml"],
    packs=["core"],
    tags=["outliers", "isolation-forest", "multivariate"],
)
def multivariate_outliers(
    ctx: CheckContext, params: MultivariateOutliersParams
) -> CheckOutput:
    """Rank the rows by how easily an isolation forest separates them from the rest."""
    from sklearn.ensemble import IsolationForest  # the `ml` extra

    rows = ctx.profile.n_rows
    numeric = [
        column
        for column in ctx.profile.columns
        if isinstance(column, NumericProfile)
        and column.counts.non_null - column.non_finite >= params.min_n
    ]
    mostly_missing = [
        column.name
        for column in numeric
        if 1 - (column.counts.non_null - column.non_finite) / max(rows, 1)
        > params.max_missing_share
    ]
    if mostly_missing:
        ctx.note(
            "features_mostly_missing",
            f"{', '.join(map(repr, mostly_missing[:10]))}"
            f"{' and more' if len(mostly_missing) > 10 else ''} "
            f"{'is' if len(mostly_missing) == 1 else 'are'} missing in over "
            f"{params.max_missing_share:.0%} of the rows, so "
            f"{'it is' if len(mostly_missing) == 1 else 'they are'} left out: a "
            "median standing in for most of a column makes the rows that have a "
            "value look far from the rest.",
        )
    features = tuple(c.name for c in numeric if c.name not in mostly_missing)
    if len(features) < _MIN_FEATURES or rows < _MIN_ROWS:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "too_little_to_compare",
            f"Needs {_MIN_FEATURES} numeric columns with {params.min_n} finite "
            f"values and {_MIN_ROWS} rows; there are {len(features)} columns and "
            f"{rows} rows.",
            severity=Severity.INFO,
        )
    if rows > params.max_rows:
        rng = np.random.default_rng(params.seed)
        positions = np.sort(rng.choice(rows, params.max_rows, replace=False))
    else:
        positions = np.arange(rows)
    names: list[str] = []
    scaled: list[np.ndarray] = []
    for name in features:
        column = ctx.values(name, dropna=False)[positions].astype(np.float64)
        column[~np.isfinite(column)] = np.nan
        # A missing or infinite value takes the column's median (0 if none is left).
        seen = ~np.isnan(column)
        fill = float(np.median(column[seen])) if seen.any() else 0.0
        column = np.where(seen, column, fill)
        low, high = float(column.min()), float(column.max())
        with np.errstate(all="ignore"):
            width = high - low
        if not np.isfinite(width):
            ctx.note(
                "feature_dropped",
                f"{name!r} was left out: its range is too large to compute with.",
            )
            continue
        if width == 0:
            continue
        # Rescaling to 0..1 keeps the values inside the single precision the forest
        # works in, and the ranks taken below do not depend on it.
        names.append(name)
        scaled.append((column - low) / width)
    if len(names) < _MIN_FEATURES:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "too_little_to_compare",
            f"Needs {_MIN_FEATURES} numeric columns that vary; {len(names)} do.",
            severity=Severity.INFO,
        )
    features = tuple(names)
    # Ranks, not values: the reference below is a normal sample, so a skewed column
    # would score above it by its shape alone.
    matrix = _normal_scores(np.column_stack(scaled))
    forest = IsolationForest(
        n_estimators=params.n_estimators,
        contamination=params.contamination,
        random_state=params.seed,
        n_jobs=1,
    )
    flagged = forest.fit_predict(matrix) == -1
    anomaly = -forest.score_samples(matrix)
    n_flagged = int(np.count_nonzero(flagged))
    # What chance reaches: the same forest on a normal sample of the same shape and
    # correlation, since columns that move together isolate rows more easily.
    reference = _normal_like(matrix, np.random.default_rng(params.seed))
    reference_forest = IsolationForest(
        n_estimators=params.n_estimators,
        contamination=params.contamination,
        random_state=params.seed,
        n_jobs=1,
    ).fit(reference)
    reference_max = float((-reference_forest.score_samples(reference)).max())
    excess = float(anomaly.max()) - reference_max

    far: np.ndarray = np.empty(0, dtype=int)
    far_limit = None
    if matrix.shape[0] >= params.min_rows_per_column * matrix.shape[1]:
        distance, rank = _robust_distances(matrix)
        far_limit = float(stats.chi2.isf(params.alpha / matrix.shape[0], rank))
        far = np.flatnonzero(distance > far_limit)
        far = far[np.lexsort((far, -distance[far]))]

    # Rows tie when the forest isolates them at the same depth; scores that differ
    # only in the last digits on another machine count as tied, and the earlier row
    # comes first, so the order is the same everywhere.
    ranked = np.lexsort((positions, -np.round(anomaly, 9)))
    far_set = set(far.tolist())
    order = [int(i) for i in far] + [int(i) for i in ranked if i not in far_set]
    examples = tuple(int(positions[i]) for i in order[:_EXAMPLES])
    output = MultivariateOutliersOutput(
        feature_columns=features,
        rows_used=int(positions.size),
        rows_total=rows,
        contamination=params.contamination,
        n_flagged=n_flagged,
        far_rows=int(far.size),
        far_limit=far_limit,
        example_rows=examples,
        max_anomaly_score=float(anomaly.max()),
        reference_max_score=reference_max,
        excess=excess,
    )
    if far.size or excess >= params.min_excess:
        _report(ctx, output, params, features, int(positions.size), rows)
    return ctx.output(output)


def _normal_like(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A normal sample shaped like ``matrix`` with the correlation of its columns."""
    values, vectors = np.linalg.eigh(np.atleast_2d(np.corrcoef(matrix, rowvar=False)))
    root = vectors * np.sqrt(np.clip(values, 0, None))
    return rng.standard_normal(matrix.shape) @ root.T


def _normal_scores(matrix: np.ndarray) -> np.ndarray:
    """Each column replaced by the normal quantile of its rank (ties share a rank).

    A skewed or heavy-tailed column puts rows far from the centre on its own, at a rate
    the chi-squared limit knows nothing about; ranks leave only how the columns
    move together, which is what the distance is for. A single column's extreme value is
    the ``outliers`` check's finding.
    """
    ranks = stats.rankdata(matrix, axis=0)
    return np.asarray(stats.norm.ppf((ranks - 0.5) / matrix.shape[0]))


def _robust_distances(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """Squared Mahalanobis distance of every row from a fit that ignores far rows.

    The mean and covariance are fitted again and again to the rows inside the
    ``_TRIM`` share of a normal cloud, with the covariance scaled up for the tail that
    was cut. A few far rows therefore do not stretch the ruler that measures them, as
    they do the ordinary mean and covariance. The distances are taken in the space the
    covariance spans, so a column that repeats others costs no degree of freedom;
    that rank is returned with them.
    """
    n, d = matrix.shape
    cut = float(stats.chi2.isf(1 - _TRIM, d))
    scale = float(stats.chi2.cdf(cut, d + 2)) / _TRIM
    keep = np.ones(n, dtype=bool)
    for round_ in range(_FIT_ROUNDS):
        sub = matrix[keep]
        centre = sub.mean(axis=0)
        covariance = np.atleast_2d(np.cov(sub, rowvar=False))
        if round_:
            covariance = covariance / scale
        values, vectors = np.linalg.eigh(covariance)
        spanned = values > values.max() * _RANK_TOLERANCE
        projected = (matrix - centre) @ vectors[:, spanned]
        distance = np.einsum("ij,j,ij->i", projected, 1 / values[spanned], projected)
        inside = distance <= cut
        if inside.sum() <= d + 1 or np.array_equal(inside, keep):
            break
        keep = inside
    return distance, int(spanned.sum())


def _report(
    ctx: CheckContext,
    output: MultivariateOutliersOutput,
    params: MultivariateOutliersParams,
    features: tuple[str, ...],
    rows_used: int,
    rows: int,
) -> None:
    far = output.far_rows
    evidence = (
        f"{far:,} of {rows_used:,} rows are beyond the limit of a robust Mahalanobis "
        f"distance on the rank-normalised columns (squared {output.far_limit:.1f}), "
        f"which columns that are jointly normal after ranking reach in "
        f"{params.alpha:.0%} of samples"
        if far
        else f"the forest's highest anomaly score, {output.max_anomaly_score:.2f}, is "
        f"{output.excess:.2f} above the {output.reference_max_score:.2f} it gives a "
        "normal sample of the same shape and correlation"
    )
    ctx.finding(
        code="multivariate_anomalies",
        subject=Subject.row_set(*features),
        severity=Severity.INFO,
        confidence=Confidence.MODERATE if far else Confidence.LOW,
        title=(
            f"Multivariate anomalies: {far:,} rows"
            if far
            else "Multivariate anomalies: a row stands out"
        ),
        message=(
            f"Across {len(features)} numeric columns together, {evidence}. A row can "
            "be ordinary in every column and unusual in their combination; look at "
            "the rows in `example_rows`. The forest ranks "
            f"{output.n_flagged:,} rows as most anomalous in any data, so that count "
            "alone says nothing."
        ),
        evidence=Evidence(
            metrics={
                "far_rows": far,
                "far_limit": output.far_limit,
                "n_flagged": output.n_flagged,
                "contamination": params.contamination,
                "n_features": len(features),
                "max_anomaly_score": output.max_anomaly_score,
                "reference_max_score": output.reference_max_score,
                "excess": output.excess,
                "example_rows": output.example_rows,
            },
            n_used=rows_used,
            n_total=rows,
            method="robust_mahalanobis+isolation_forest" if far else "isolation_forest",
        ),
    )
