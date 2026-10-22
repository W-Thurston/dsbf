# dsbf/checks/multivariate_outliers.py
"""Rows that are unusual across the numeric columns together."""

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


class MultivariateOutliersParams(CheckParams):
    contamination: float = Field(
        0.05,
        gt=0,
        le=0.5,
        description="Share of the rows to flag. The forest flags exactly this share "
        "of whatever it is given, so a count is only a ranking of the rows.",
    )
    n_estimators: int = Field(100, ge=10, le=1000, description="Trees in the forest.")
    min_n: int = Field(
        10, ge=4, description="Finite values a column needs to be a feature."
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
    n_flagged: int = Field(ge=0)
    example_rows: tuple[int, ...] = Field(
        description="Positions in the data of the most anomalous rows, most first."
    )
    max_anomaly_score: float = Field(
        description="Highest anomaly score, from 0 to 1; near 0.5 for all rows when "
        "none stands out, and approaching 1 for a clear anomaly."
    )


@check(
    id="multivariate_outliers",
    version="1.0.0",
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

    features = tuple(
        column.name
        for column in ctx.profile.columns
        if isinstance(column, NumericProfile)
        and column.counts.non_null - column.non_finite >= params.min_n
    )
    rows = ctx.profile.n_rows
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
        # The forest splits each column at random between its smallest and largest
        # value, so rescaling a column to 0..1 changes nothing it does, and keeps the
        # values inside the single precision it works in.
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
    matrix = np.column_stack(scaled)
    forest = IsolationForest(
        n_estimators=params.n_estimators,
        contamination=params.contamination,
        random_state=params.seed,
        n_jobs=1,
    )
    flagged = forest.fit_predict(matrix) == -1
    anomaly = -forest.score_samples(matrix)
    n_flagged = int(np.count_nonzero(flagged))
    # Rows tie when the forest isolates them at the same depth; scores that differ
    # only in the last digits on another machine count as tied, and the earlier row
    # comes first, so the order is the same everywhere.
    ranked = np.lexsort((positions, -np.round(anomaly, 9)))
    examples = tuple(int(p) for p in positions[ranked[:_EXAMPLES]])
    output = MultivariateOutliersOutput(
        feature_columns=features,
        rows_used=int(positions.size),
        rows_total=rows,
        contamination=params.contamination,
        n_flagged=n_flagged,
        example_rows=examples,
        max_anomaly_score=float(anomaly.max()),
    )
    if n_flagged:
        ctx.finding(
            code="multivariate_anomalies",
            subject=Subject.row_set(*features),
            severity=Severity.INFO,
            confidence=Confidence.LOW,
            title=f"Multivariate anomalies: {n_flagged:,} rows",
            message=(
                f"An isolation forest over {len(features)} numeric columns ranks "
                f"{n_flagged:,} of {positions.size:,} rows as the most easily "
                "separated from the rest. It flags the share set by `contamination` "
                f"({params.contamination:.0%}) in any data, so the count says "
                f"nothing; the highest anomaly score ({output.max_anomaly_score:.2f}"
                ") says whether any row stands out. Look at the rows in "
                "`example_rows`: a row can be ordinary in every column and unusual "
                "in their combination."
            ),
            evidence=Evidence(
                metrics={
                    "n_flagged": n_flagged,
                    "contamination": params.contamination,
                    "n_features": len(features),
                    "max_anomaly_score": output.max_anomaly_score,
                    "example_rows": examples,
                },
                n_used=int(positions.size),
                n_total=rows,
                method="isolation_forest",
            ),
        )
    return ctx.output(output)
