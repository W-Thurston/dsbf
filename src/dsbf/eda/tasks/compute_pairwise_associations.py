# dsbf/eda/tasks/compute_pairwise_associations.py
#
# This task is the single source of truth for all pairwise column relationships.
# It supersedes compute_correlations.py, which has been removed.
#
# New in this version:
#   - Spearman rank correlation for continuous pairs (method="spearman" or "both")
#   - correlation_matrix nested dict in data for heatmap consumption
#   - Reliability warnings ported from the former compute_correlations task

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scipy.stats import chi2_contingency, kendalltau, pointbiserialr, spearmanr

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


def _spearman_r(a: pd.Series, b: pd.Series) -> float | None:
    """
    Compute Spearman rank correlation between two continuous Series.

    Spearman is a monotonic (rank-based) correlation that is robust to outliers
    and non-normal distributions. Unlike Pearson, it detects any monotonic
    relationship, not just linear ones.

    Args:
        a: First numeric Series.
        b: Second numeric Series.

    Returns:
        Spearman r in [-1, 1], or None if fewer than 3 complete pairs exist.

    """
    paired: DataFrame = pd.concat([a, b], axis=1).dropna()
    if len(paired) < 3:
        return None
    r, _ = spearmanr(paired.iloc[:, 0].values, paired.iloc[:, 1].values)
    return float(r)


def _kendalls_tau(a: pd.Series, b: pd.Series) -> float | None:
    """
    Compute Kendall's tau-b rank correlation between two continuous Series.

    Preferred over Spearman for small samples (n < 30): the p-value is more
    accurate, and tau has a direct probabilistic interpretation (proportion of
    concordant minus discordant pairs). Also robust to ties via the -b variant.

    Args:
        a: First numeric Series.
        b: Second numeric Series.

    Returns:
        Kendall's tau-b in [-1, 1], or None if fewer than 3 complete pairs.

    """
    paired: DataFrame = pd.concat([a, b], axis=1).dropna()
    if len(paired) < 3:
        return None
    tau, _ = kendalltau(paired.iloc[:, 0].values, paired.iloc[:, 1].values)
    return float(tau)


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
    except Exception:
        return None


def _point_biserial(continuous: pd.Series, binary: pd.Series) -> float | None:
    """
    Compute point-biserial correlation between a continuous and binary Series.

    Args:
        continuous: Numeric Series.
        binary: Series with exactly two distinct non-null values.

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
    except Exception:
        return None


def _eta_squared(continuous: pd.Series, categorical: pd.Series) -> float | None:
    """
    Compute eta squared (η²) for a continuous variable grouped by a categorical.

    Uses one-way ANOVA decomposition: SS_between / SS_total.

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
    except Exception:
        return None


# ── Strength labelling ─────────────────────────────────────────────────────────


def _strength(value: float, metric_type: str) -> str:
    """
    Map a metric value to a human-readable strength label.

    Thresholds follow standard effect-size conventions:

    - ``pearson_r`` / ``spearman_r`` / ``point_biserial_r``:
      |r| ≥ 0.7 strong, ≥ 0.4 moderate, ≥ 0.2 weak, else negligible.
    - ``cramers_v``:
      V ≥ 0.5 strong, ≥ 0.3 moderate, ≥ 0.1 weak, else negligible.
    - ``eta_squared``:
      η² ≥ 0.14 strong (Cohen large), ≥ 0.06 moderate, ≥ 0.01 weak,
      else negligible.

    Args:
        value: Raw metric value (sign is ignored for directional metrics).
        metric_type: One of ``pearson_r``, ``spearman_r``, ``point_biserial_r``,
            ``cramers_v``, or ``eta_squared``.

    Returns:
        One of ``"strong"``, ``"moderate"``, ``"weak"``, ``"negligible"``,
        or ``"unknown"`` if the metric type is unrecognised.

    """
    v: float = abs(value)
    if metric_type in ("pearson_r", "spearman_r", "point_biserial_r", "kendalls_tau"):
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


def _metric_for_pair(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    intent_a: str,
    intent_b: str,
    method: str = "auto",
    n_valid_pairs: int = 0,
) -> tuple[float, str] | None:
    """
    Choose and compute the appropriate association metric for a column pair.

    Dispatch table:

    - ``continuous x continuous``             → Pearson, Spearman, Kendall's, or auto
    - ``continuous x categorical`` (binary)   → point-biserial r
    - ``continuous x categorical`` (multi)    → eta squared (η²)
    - ``categorical x categorical``           → Cramér's V

    For continuous pairs, ``method`` controls which metric is used:

    - ``"auto"`` (default): Kendall's tau when ``n_valid_pairs < 30`` (small
      sample, tau's p-value is more accurate); Pearson otherwise.
    - ``"pearson"``: always Pearson r.
    - ``"spearman"``: always Spearman r.
    - ``"kendall"``: always Kendall's tau.
    - ``"both"``: Pearson as primary (Spearman stored separately by caller).

    Args:
        df: Source DataFrame (pandas).
        col_a: First column name.
        col_b: Second column name.
        intent_a: Semantic intent of col_a.
        intent_b: Semantic intent of col_b.
        method: Correlation method for continuous pairs. Default: ``"auto"``.
        n_valid_pairs: Number of complete row pairs (used for auto routing).

    Returns:
        ``(value, metric_type)`` tuple, or None if the pair should be skipped.

    """
    a, b = df[col_a], df[col_b]

    if intent_a == "continuous" and intent_b == "continuous":
        # Auto routing: tau for small samples, pearson for large
        if method == "kendall" or (method == "auto" and n_valid_pairs < 30):
            val: float | None = _kendalls_tau(a, b)
            return (val, "kendalls_tau") if val is not None else None
        if method == "spearman":
            val = _spearman_r(a, b)
            return (val, "spearman_r") if val is not None else None
        # "pearson", "both", or "auto" with n >= 30 → Pearson
        val = _pearson_r(a, b)
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

    return None  # datetime, id, text, unknown - skip


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    display_name="Compute Pairwise Associations",
    description=(
        "Computes the most appropriate association metric for every column pair: "
        "Pearson r, Spearman r, or Kendall's tau (continuousxcontinuous), "
        "point-biserial r (continuousxbinary), eta squared (continuousxcategorical), "
        "or Cramér's V (categoricalxcategorical). Default method='auto' routes to "
        "Kendall's tau for small samples (n<30) and Pearson for large samples. "
        "Columns typed as id, datetime, or text are skipped. Also produces a "
        "correlation_matrix dict for heatmap rendering."
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
    Compute the most appropriate pairwise association metric for all column pairs.

    Selects the metric based on the semantic intent of each column:

    - ``continuous x continuous``           → auto-routed (see below), Pearson r,
      Spearman r, Kendall's tau, or both
    - ``continuous x binary categorical``   → point-biserial r
    - ``continuous x multi-level categorical`` → eta squared (η²)
    - ``categorical x categorical``         → Cramér's V
    - Any column typed as ``id``, ``datetime``, ``text``, or ``unknown`` is skipped.

    **method="auto" (default):**
    Routes continuous pairs to Kendall's tau when ``n_valid_pairs < 30`` and to
    Pearson r otherwise. Kendall's tau is preferred for small samples because its
    p-value is more accurate than Spearman's and it has a direct probabilistic
    interpretation (proportion of concordant minus discordant pairs). This
    supersedes the standalone ``kendalls_tau`` task.

    When ``method="both"``, each continuous-continuous pair entry includes both
    ``pearson_r`` and ``spearman_r`` values, and ``metric_type`` is set to
    ``"pearson_r"`` (primary). The Spearman value is stored under ``spearman_r``
    in the entry dict for direct access.

    Additionally produces a ``correlation_matrix`` nested dict in ``data``
    (continuous numeric pairs only, for all continuous-pair metric types) for
    consumption by ``generate_dataset_summary_plots``.

    Reliability warnings (low_n, zero_variance, extreme_outliers, high_skew)
    are attached when data conditions may distort results.

    Configurable parameters (via config["tasks"]["compute_pairwise_associations"]):
        method (str): Correlation method for continuous pairs.
            ``"auto"`` (default), ``"pearson"``, ``"spearman"``, ``"kendall"``,
            or ``"both"``.
        min_sample_size (int): Minimum complete row pairs required to compute
            an association. Default: 30
        cat_cardinality_limit (int): Maximum unique values in a categorical
            column before Cramér's V is skipped for that column. Default: 50

    Output data structure (keyed by ``"COL_A|COL_B"``)::

        {
            "YARDS_WINNER|YARDS_LOSER": {
                "metric":       0.412,
                "metric_type":  "pearson_r",   # or "kendalls_tau", "spearman_r"
                "strength":     "moderate",
                "col_a_intent": "continuous",
                "col_b_intent": "continuous",
                # present only when method="both":
                "spearman_r":   0.389,
            },
            ...
            "__correlation_matrix__": {
                "col1": {"col1": 1.0, "col2": 0.41, ...},
                ...
            }
        }
    """

    def run(self) -> None:
        """
        Execute pairwise association computation and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data
            if is_polars(df):
                self._log(
                    "    Converting Polars to pandas for association computation.",
                    "debug",
                )
                df = df.to_pandas()

            method: str = str(self.get_task_param("method") or "auto")
            min_sample_size = int(self.get_task_param("min_sample_size") or 30)
            cat_cardinality_limit = int(
                self.get_task_param("cat_cardinality_limit") or 50
            )

            # Retrieve semantic types populated by infer_types.
            semantic_types: dict[str, str] = {}
            if self.context:
                semantic_types = self.context.get_metadata("semantic_types") or {}

            # Fallback: derive basic intent from pandas dtype when semantic types
            # are absent (e.g. standalone test runs without infer_types upstream).
            if not semantic_types:
                n_rows: int = len(df)
                for col in df.columns:
                    dtype = df[col].dtype
                    n_unique = df[col].nunique()
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
                    (
                        "    No semantic types in context - "
                        "inferred from dtype as fallback."
                    ),
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

            # High-cardinality guard: skip Cramér's V pairs for columns with too
            # many unique values to avoid OOM on large contingency tables.
            cat_unique: dict[str, int] = {
                col: int(df[col].nunique())
                for col in eligible
                if semantic_types.get(col) == "categorical"
            }

            associations: dict[str, dict[str, Any]] = {}
            metric_type_counts: dict[str, int] = {}
            spearman_skipped: list[str] = []

            for i, col_a in enumerate(eligible):
                for col_b in eligible[i + 1 :]:
                    intent_a: str = semantic_types.get(col_a, "unknown")
                    intent_b: str = semantic_types.get(col_b, "unknown")

                    # Skip high-cardinality categorical pairs.
                    if (
                        intent_a == "categorical"
                        and intent_b == "categorical"
                        and (
                            cat_unique.get(col_a, 0) > cat_cardinality_limit
                            or cat_unique.get(col_b, 0) > cat_cardinality_limit
                        )
                    ):
                        spearman_skipped.append(f"{col_a}|{col_b}")
                        continue

                    # Skip pairs with insufficient complete rows.
                    valid_n = df[[col_a, col_b]].dropna().shape[0]
                    if valid_n < min_sample_size:
                        self._log(
                            f"    Skipping {col_a}|{col_b}: "
                            f"only {valid_n} complete rows (min={min_sample_size}).",
                            "debug",
                        )
                        continue

                    pair_result: tuple[float, str] | None = _metric_for_pair(
                        df,
                        col_a,
                        col_b,
                        intent_a,
                        intent_b,
                        method,
                        n_valid_pairs=valid_n,
                    )
                    if pair_result is None:
                        continue

                    value, metric_type = pair_result
                    key: str = f"{col_a}|{col_b}"
                    entry: dict[str, Any] = {
                        "metric": round(value, 6),
                        "metric_type": metric_type,
                        "strength": _strength(value, metric_type),
                        "col_a_intent": intent_a,
                        "col_b_intent": intent_b,
                    }

                    # When method="both", also compute and store Spearman for
                    # continuous pairs alongside the primary Pearson value.
                    if (
                        method == "both"
                        and intent_a == "continuous"
                        and intent_b == "continuous"
                    ):
                        sp: float | None = _spearman_r(df[col_a], df[col_b])
                        if sp is not None:
                            entry["spearman_r"] = round(sp, 6)
                            entry["spearman_strength"] = _strength(sp, "spearman_r")

                    associations[key] = entry
                    metric_type_counts[metric_type] = (
                        metric_type_counts.get(metric_type, 0) + 1
                    )

            # ── Correlation matrix for heatmap rendering ──────────────────────
            # Build a nested dict from numeric-pair Pearson values so
            # generate_dataset_summary_plots can read it from context results
            # rather than recomputing the matrix from the raw DataFrame.
            numeric_cols: list[str] = [
                col for col in eligible if semantic_types.get(col) == "continuous"
            ]
            correlation_matrix: dict[str, dict[str, float]] = {}
            if len(numeric_cols) >= 2:
                for col in numeric_cols:
                    correlation_matrix[col] = {col: 1.0}
                for key, entry in associations.items():
                    if entry["metric_type"] in (
                        "pearson_r",
                        "spearman_r",
                        "kendalls_tau",
                    ):
                        col_a, col_b = key.split("|", 1)
                        val = entry["metric"]
                        correlation_matrix.setdefault(col_a, {})[col_b] = val
                        correlation_matrix.setdefault(col_b, {})[col_a] = val

            strength_counts: dict[str, int] = {
                s: sum(1 for v in associations.values() if v["strength"] == s)
                for s in ("strong", "moderate", "weak", "negligible")
            }

            data: dict[str, Any] = {**associations}
            if correlation_matrix:
                # Stored under a sentinel key so the API can extract it separately
                # without it being treated as a column-pair association entry.
                data["__correlation_matrix__"] = correlation_matrix

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
                data=data,
                metadata={
                    "method": method,
                    "min_sample_size": min_sample_size,
                    "cat_cardinality_limit": cat_cardinality_limit,
                    "eligible_columns": eligible,
                    "skipped_columns": skipped,
                    "numeric_columns": numeric_cols,
                    "suggested_viz_type": "heatmap",
                    "recommended_section": "Relationships",
                    "display_priority": "high",
                    "column_types": self.get_column_type_info(list(df.columns)),
                },
            )

            # ── Reliability warnings ──────────────────────────────────────────
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

            if flags.get("extreme_outliers"):
                code: Literal["extreme_outliers_low_n", "extreme_outliers"] = (
                    "extreme_outliers_low_n"
                    if flags.get("low_row_count")
                    else "extreme_outliers"
                )
                add_reliability_warning(
                    self.output,
                    level="heuristic_caution",
                    code=code,
                    description=(
                        "Some features contain extreme z-scores (|z| > 3), but "
                        "sample size is small (N < 30). Outlier estimates may be "
                        "unreliable."
                        if flags.get("low_row_count")
                        else "Some features contain extreme z-scores (|z| > 3), "
                        "which may distort Pearson correlation."
                    ),
                    recommendation=(
                        "Interpret outlier influence with caution or validate "
                        "using robust statistics."
                        if flags.get("low_row_count")
                        else "Winsorize outliers or use Spearman correlation "
                        "(set method='spearman') or Kendall's tau "
                        "(set method='kendall' or method='auto' on small samples)."
                    ),
                )

            if flags.get("high_skew"):
                code_s: Literal["high_skew_low_n", "high_skew"] = (
                    "high_skew_low_n" if flags.get("low_row_count") else "high_skew"
                )
                add_reliability_warning(
                    self.output,
                    level="heuristic_caution",
                    code=code_s,
                    description=(
                        "High skew was detected, but sample size is small (N < 30). "
                        "Skew estimates may be unstable."
                        if flags.get("low_row_count")
                        else "One or more features are highly skewed, which may "
                        "distort Pearson correlation strength."
                    ),
                    recommendation=(
                        "Interpret skewness cautiously or validate with bootstrapping."
                        if flags.get("low_row_count")
                        else "Use Spearman correlation (set method='spearman') or "
                        "log-transform skewed variables."
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
