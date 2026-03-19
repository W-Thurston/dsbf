# dsbf/eda/tasks/compute_pairwise_associations.py

from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scipy.stats import chi2_contingency, pointbiserialr

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import (
    TaskResult,
    add_reliability_warning,
    log_reliability_warnings,
    make_failure_result,
)
from dsbf.utils.backend import is_polars

# ── Metric helpers ─────────────────────────────────────────────────────────────


def _pearson_r(a: pd.Series, b: pd.Series) -> float | None:
    """
    Compute Pearson correlation between two continuous Series.

    Args:
        a: First numeric Series.
        b: Second numeric Series.

    Returns:
        Pearson r in [-1, 1], or None if fewer than 3 complete pairs exist.

    """
    paired: DataFrame = pd.concat([a, b], axis=1).dropna()
    if len(paired) < 3:
        return None
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))


def _cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """
    Compute Cramér's V association between two categorical Series.

    Args:
        a: First categorical Series.
        b: Second categorical Series.

    Returns:
        Cramér's V in [0, 1], or None if fewer than 3 complete pairs exist
        or the contingency table is degenerate.

    """
    paired: DataFrame = pd.concat([a, b], axis=1).dropna()
    if len(paired) < 3:
        return None
    try:
        contingency: DataFrame = pd.crosstab(paired.iloc[:, 0], paired.iloc[:, 1])
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.to_numpy().sum()
        r, k = contingency.shape
        denom: int = min(k - 1, r - 1)
        if denom <= 0 or n == 0:
            return None
        return float(np.sqrt(chi2 / n / denom))
    except Exception:  # noqa: BLE001
        return None


def _point_biserial(continuous: pd.Series, binary: pd.Series) -> float | None:
    """
    Compute point-biserial correlation between a continuous and binary Series.

    Args:
        continuous: Numeric Series.
        binary: Series with exactly two distinct non-null values (0/1, True/False,
            or any two-level categorical).

    Returns:
        Point-biserial r in [-1, 1], or None if the binary column does not have
        exactly 2 levels or fewer than 3 complete pairs exist.

    """
    paired: DataFrame = pd.concat([continuous, binary], axis=1).dropna()
    if len(paired) < 3:
        return None
    try:
        b = pd.Categorical(paired.iloc[:, 1]).codes
        if len(np.unique(b)) != 2:
            return None
        r, _ = pointbiserialr(b, paired.iloc[:, 0].values)
        return float(r)
    except Exception:  # noqa: BLE001
        return None


def _eta_squared(continuous: pd.Series, categorical: pd.Series) -> float | None:
    """
    Compute eta squared (η²) for a continuous variable grouped by a categorical.

    Eta squared measures the proportion of variance in the continuous variable
    explained by group membership. Uses one-way ANOVA decomposition:
    SS_between / SS_total.

    Args:
        continuous: Numeric Series to measure variance in.
        categorical: Grouping Series (multi-level categorical).

    Returns:
        η² in [0, 1], or None if fewer than 2 groups with ≥ 2 observations exist
        or total variance is zero.

    """
    paired: DataFrame = pd.concat([continuous, categorical], axis=1).dropna()
    if len(paired) < 3:
        return None
    try:
        groups: list = [
            grp.iloc[:, 0].to_numpy()
            for _, grp in paired.groupby(paired.iloc[:, 1])
            if len(grp) >= 2
        ]
        if len(groups) < 2:
            return None
        all_vals: ndarray = np.concatenate(groups)
        grand_mean = all_vals.mean()
        ss_total: ndarray[tuple[Any, ...]] = np.sum((all_vals - grand_mean) ** 2)
        if ss_total == 0:
            return None
        ss_between: int = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        return float(np.clip(ss_between / ss_total, 0.0, 1.0))
    except Exception:  # noqa: BLE001
        return None


# ── Strength labelling ─────────────────────────────────────────────────────────


def _strength(value: float, metric_type: str) -> str:  # noqa: C901, PLR0911
    """
    Map a metric value to a human-readable strength label.

    Thresholds follow standard effect-size conventions:

    - ``pearson_r`` / ``point_biserial_r``: |r| ≥ 0.7 strong, ≥ 0.4 moderate,
      ≥ 0.2 weak, else negligible.
    - ``cramers_v``: V ≥ 0.5 strong, ≥ 0.3 moderate, ≥ 0.1 weak, else negligible.
    - ``eta_squared``: η² ≥ 0.14 strong (Cohen large), ≥ 0.06 moderate,
      ≥ 0.01 weak, else negligible.

    Args:
        value: Raw metric value (sign is ignored for directional metrics).
        metric_type: One of ``pearson_r``, ``point_biserial_r``, ``cramers_v``,
            or ``eta_squared``.

    Returns:
        One of ``"strong"``, ``"moderate"``, ``"weak"``, ``"negligible"``,
        or ``"unknown"`` if the metric type is unrecognised.

    """
    v: float = abs(value)
    if metric_type in ("pearson_r", "point_biserial_r"):
        if v >= 0.7:
            return "strong"
        if v >= 0.4:
            return "moderate"
        if v >= 0.2:
            return "weak"
        return "negligible"
    if metric_type == "cramers_v":
        if v >= 0.5:
            return "strong"
        if v >= 0.3:
            return "moderate"
        if v >= 0.1:
            return "weak"
        return "negligible"
    if metric_type == "eta_squared":
        if v >= 0.14:
            return "strong"
        if v >= 0.06:
            return "moderate"
        if v >= 0.01:
            return "weak"
        return "negligible"
    return "unknown"


# ── Column type dispatch ───────────────────────────────────────────────────────

_SKIP_INTENTS: set[str] = {"id", "datetime", "text", "unknown"}


def _is_binary(series: pd.Series) -> bool:
    """Return True if the series has exactly two distinct non-null values."""
    return series.dropna().nunique() == 2


def _metric_for_pair(  # noqa: PLR0911
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    intent_a: str,
    intent_b: str,
) -> tuple[float, str] | None:
    """
    Choose and compute the appropriate association metric for a column pair.

    Dispatch table:

    - ``continuous x continuous``  → Pearson r
    - ``continuous x categorical`` (binary) → point-biserial r
    - ``continuous x categorical`` (multi)  → eta squared (η²)
    - ``categorical x categorical`` → Cramér's V

    Args:
        df: Source DataFrame (pandas).
        col_a: First column name.
        col_b: Second column name.
        intent_a: Semantic intent of col_a (e.g. ``"continuous"``).
        intent_b: Semantic intent of col_b.

    Returns:
        ``(value, metric_type)`` tuple, or None if the pair should be skipped
        (incompatible intent combination or insufficient data).

    """
    a, b = df[col_a], df[col_b]

    if intent_a == "continuous" and intent_b == "continuous":
        val: float | None = _pearson_r(a, b)
        return (val, "pearson_r") if val is not None else None

    if intent_a == "continuous" and intent_b == "categorical":
        if _is_binary(b):
            val = _point_biserial(a, b)
            return (val, "point_biserial_r") if val is not None else None
        val = _eta_squared(a, b)
        return (val, "eta_squared") if val is not None else None

    if intent_a == "categorical" and intent_b == "continuous":
        if _is_binary(a):
            val = _point_biserial(b, a)
            return (val, "point_biserial_r") if val is not None else None
        val = _eta_squared(b, a)
        return (val, "eta_squared") if val is not None else None

    if intent_a == "categorical" and intent_b == "categorical":
        val = _cramers_v(a, b)
        return (val, "cramers_v") if val is not None else None

    return None  # datetime, id, text, unknown — skip


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    display_name="Compute Pairwise Associations",
    description=(
        "Computes the most appropriate association metric for every column pair: "
        "Pearson r (continuousxcontinuous), point-biserial r (continuousxbinary), "
        "eta squared (continuousxcategorical), or Cramér's V (categoricalxcategorical)."
        " Columns typed as id, datetime, or text are skipped."
    ),
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    domain="core",
    runtime_estimate="moderate",
    phase="eda",
    tags=["correlation", "association", "relationships"],
    expected_semantic_types=["any"],
)
class ComputePairwiseAssociations(BaseTask):
    """
    Computes the most appropriate pairwise association metric for all column pairs.

    Selects the metric based on the semantic intent of each column:

    - ``continuous x continuous`` → Pearson r
    - ``continuous x binary categorical`` → point-biserial r
    - ``continuous x multi-level categorical`` → eta squared (η²)
    - ``categorical x categorical`` → Cramér's V
    - Any column typed as ``id``, ``datetime``, ``text``, or ``unknown`` is skipped.

    When semantic types are not yet populated in context (e.g. standalone test runs),
    a fallback inference from pandas dtypes is used so the task produces useful
    output rather than an empty result.

    Output data structure (keyed by ``"COL_A|COL_B"`` in lexicographic order)::

        {
            "YARDS_WINNER|YARDS_LOSER": {
                "metric":       0.412,
                "metric_type":  "pearson_r",
                "strength":     "moderate",
                "col_a_intent": "continuous",
                "col_b_intent": "continuous",
            },
            ...
        }

    Output is consumed by the Relationships tab association matrix and the
    column-level association detail panel.
    """

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """
        Execute pairwise association computation and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data
            if is_polars(df):
                # scipy and pandas association helpers require numpy/pandas arrays.
                self._log(
                    "    Converting Polars to pandas for association computation.",
                    "debug",
                )
                df = df.to_pandas()

            min_sample_size = int(self.get_task_param("min_sample_size") or 30)

            # Retrieve semantic types populated by infer_types.
            semantic_types: dict[str, str] = {}
            if self.context:
                semantic_types = self.context.get_metadata("semantic_types") or {}

            # Fallback: derive basic intent from pandas dtype when semantic types
            # are absent (e.g. standalone test runs without infer_types upstream).
            if not semantic_types:
                for col in df.columns:
                    dtype = df[col].dtype
                    n_unique = df[col].nunique()
                    n_rows: int = len(df)
                    if pd.api.types.is_numeric_dtype(dtype):
                        if n_unique / n_rows > 0.95 and pd.api.types.is_integer_dtype(
                            dtype,
                        ):
                            semantic_types[col] = "id"
                        else:
                            semantic_types[col] = "continuous"
                    elif pd.api.types.is_datetime64_any_dtype(dtype):
                        semantic_types[col] = "datetime"
                    elif n_unique / n_rows > 0.95 or n_unique > 500:
                        semantic_types[col] = "id"
                    else:
                        semantic_types[col] = "categorical"
                self._log(
                    "    No semantic types in context — inferred from dtype as "
                    "fallback.",
                    "debug",
                )

            eligible: list[str] = [
                col
                for col in df.columns
                if semantic_types.get(col, "unknown") not in _SKIP_INTENTS
            ]
            skipped: list[str] = [
                col
                for col in df.columns
                if semantic_types.get(col, "unknown") in _SKIP_INTENTS
            ]

            self._log(
                f"    Computing associations for {len(eligible)} eligible columns "
                f"({len(skipped)} skipped: id/datetime/text/unknown).",
                "debug",
            )

            associations: dict[str, dict[str, Any]] = {}
            metric_type_counts: dict[str, int] = {}

            for i, col_a in enumerate(eligible):
                for col_b in eligible[i + 1 :]:
                    intent_a: str = semantic_types.get(col_a, "unknown")
                    intent_b: str = semantic_types.get(col_b, "unknown")

                    # Skip pairs with insufficient complete rows to be meaningful.
                    valid_n = df[[col_a, col_b]].dropna().shape[0]
                    if valid_n < min_sample_size:
                        self._log(
                            f"    Skipping {col_a}|{col_b}: "
                            f"only {valid_n} complete rows (min={min_sample_size}).",
                            "debug",
                        )
                        continue

                    pair_result: tuple[float, str] | None = _metric_for_pair(
                        df, col_a, col_b, intent_a, intent_b
                    )
                    if pair_result is None:
                        continue

                    value, metric_type = pair_result

                    # Lexicographic key ensures A|B == B|A lookups are consistent.
                    key: str = f"{col_a}|{col_b}"
                    associations[key] = {
                        "metric": round(value, 6),
                        "metric_type": metric_type,
                        "strength": _strength(value, metric_type),
                        "col_a_intent": intent_a,
                        "col_b_intent": intent_b,
                    }
                    metric_type_counts[metric_type] = (
                        metric_type_counts.get(metric_type, 0) + 1
                    )

            strength_counts: dict[str, int] = {
                s: sum(1 for v in associations.values() if v["strength"] == s)
                for s in ("strong", "moderate", "weak", "negligible")
            }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed {len(associations)} pairwise associations "
                        f"across {len(eligible)} columns."
                    ),
                    "pair_count": len(associations),
                    "metric_type_counts": metric_type_counts,
                    "strength_counts": strength_counts,
                },
                data=associations,
                metadata={
                    "min_sample_size": min_sample_size,
                    "eligible_columns": eligible,
                    "skipped_columns": skipped,
                    "suggested_viz_type": "heatmap",
                    "recommended_section": "Relationships",
                    "display_priority": "high",
                    "column_types": self.get_column_type_info(list(df.columns)),
                },
            )

            flags: dict = self.ensure_reliability_flags()

            if flags.get("low_row_count"):
                add_reliability_warning(
                    self.output,
                    level="strong_warning",
                    code="low_row_count",
                    description=(
                        "Association metrics may be unreliable with fewer than "
                        "30 observations. Treat all values with caution."
                    ),
                    recommendation=(
                        "Collect more data or use bootstrapped confidence intervals."
                    ),
                )

            if flags.get("zero_variance_cols"):
                add_reliability_warning(
                    self.output,
                    level="strong_warning",
                    code="zero_variance",
                    description=(
                        "Columns with near-zero variance produce undefined or "
                        "misleading association metrics: "
                        f"{flags['zero_variance_cols']}."
                    ),
                    recommendation=(
                        "Drop constant columns before interpreting associations."
                    ),
                )

            log_reliability_warnings(self, self.output)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
