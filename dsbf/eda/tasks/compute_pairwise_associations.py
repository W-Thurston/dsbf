# dsbf/eda/tasks/compute_pairwise_associations.py

from typing import Any

import numpy as np
import pandas as pd
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
    """Pearson r for two continuous series."""
    paired = pd.concat([a, b], axis=1).dropna()
    if len(paired) < 3:
        return None
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))


def _cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """Cramér's V for two categorical series."""
    paired = pd.concat([a, b], axis=1).dropna()
    if len(paired) < 3:
        return None
    try:
        contingency = pd.crosstab(paired.iloc[:, 0], paired.iloc[:, 1])
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.values.sum()
        r, k = contingency.shape
        denom = min(k - 1, r - 1)
        if denom <= 0 or n == 0:
            return None
        return float(np.sqrt(chi2 / n / denom))
    except Exception:
        return None


def _point_biserial(continuous: pd.Series, binary: pd.Series) -> float | None:
    """Point-biserial r: continuous × binary (0/1 or True/False)."""
    paired = pd.concat([continuous, binary], axis=1).dropna()
    if len(paired) < 3:
        return None
    try:
        # Encode binary column as 0/1 integers
        b = pd.Categorical(paired.iloc[:, 1]).codes
        if len(np.unique(b)) != 2:
            return None
        r, _ = pointbiserialr(b, paired.iloc[:, 0].values)
        return float(r)
    except Exception:
        return None


def _eta_squared(continuous: pd.Series, categorical: pd.Series) -> float | None:
    """
    Eta squared (η²) - proportion of variance in the continuous variable
    explained by group membership in the categorical variable.
    Uses one-way ANOVA decomposition: SS_between / SS_total.
    """
    paired = pd.concat([continuous, categorical], axis=1).dropna()
    if len(paired) < 3:
        return None
    try:
        groups = [
            grp.iloc[:, 0].values
            for _, grp in paired.groupby(paired.iloc[:, 1])
            if len(grp) >= 2
        ]
        if len(groups) < 2:
            return None
        all_vals = np.concatenate(groups)
        grand_mean = all_vals.mean()
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        if ss_total == 0:
            return None
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        return float(np.clip(ss_between / ss_total, 0.0, 1.0))
    except Exception:
        return None


# ── Strength labelling ─────────────────────────────────────────────────────────


def _strength(value: float, metric_type: str) -> str:
    """
    Map a metric value to a human-readable strength label.

    Thresholds:
      pearson_r / point_biserial_r : |r| ≥ 0.7 strong, ≥ 0.4 moderate,
                                          ≥ 0.2 weak, else negligible
      cramers_v                    : V  ≥ 0.5 strong, ≥ 0.3 moderate,
                                          ≥ 0.1 weak, else negligible
      eta_squared                  : η² ≥ 0.14 strong (Cohen large),
                                          ≥ 0.06 moderate, ≥ 0.01 weak,
                                          else negligible
    """
    v = abs(value)
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

_SKIP_INTENTS = {"id", "datetime", "text", "unknown"}


def _is_binary(series: pd.Series) -> bool:
    """True if the series has exactly two distinct non-null values."""
    return series.dropna().nunique() == 2


def _metric_for_pair(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    intent_a: str,
    intent_b: str,
) -> tuple[float, str] | None:
    """
    Choose and compute the appropriate association metric for a column pair.

    Returns (value, metric_type) or None if the pair should be skipped.

    Dispatch table:
      continuous  × continuous  → Pearson r
      continuous  × categorical (binary) → point-biserial r
      continuous  × categorical (multi)  → eta squared (η²)
      categorical × categorical → Cramér's V
    """
    a, b = df[col_a], df[col_b]

    if intent_a == "continuous" and intent_b == "continuous":
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
        "Pearson r (continuous×continuous), point-biserial r (continuous×binary), "
        "eta squared (continuous×categorical), or Cramér's V (categorical×categorical)."
        " Columns typed as id, datetime, or text are skipped."
    ),
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    domain="core",
    runtime_estimate="moderate",
    tags=["correlation", "association", "relationships"],
    expected_semantic_types=["any"],
)
class ComputePairwiseAssociations(BaseTask):
    """
    Produces a flat dict of all eligible column pairs with their association
    metric, metric type, strength label, and column intent types.

    Output data structure (keyed by "COL_A|COL_B"):
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

    Pairs are stored with the lexicographically earlier column name first so
    lookups from either direction are consistent.
    """

    def run(self) -> None:
        try:
            df = self.input_data
            if is_polars(df):
                self._log(
                    "    Converting Polars to Pandas for association computation.",
                    "debug",
                )
                df = df.to_pandas()

            min_sample_size = int(self.get_task_param("min_sample_size") or 30)

            # Retrieve semantic types from context
            semantic_types: dict[str, str] = {}
            if self.context:
                semantic_types = self.context.get_metadata("semantic_types") or {}

            # Fallback: if semantic types were not populated by infer_types (e.g.
            # standalone test run), derive a basic intent from pandas dtype so the
            # task produces useful output rather than skipping everything.
            if not semantic_types:
                for col in df.columns:
                    dtype = df[col].dtype
                    n_unique = df[col].nunique()
                    n_rows = len(df)
                    if pd.api.types.is_numeric_dtype(dtype):
                        # Treat high-uniqueness integers as ids; the rest continuous
                        if n_unique / n_rows > 0.95 and pd.api.types.is_integer_dtype(
                            dtype
                        ):
                            semantic_types[col] = "id"
                        else:
                            semantic_types[col] = "continuous"
                    elif pd.api.types.is_datetime64_any_dtype(dtype):
                        semantic_types[col] = "datetime"
                    else:
                        # String/object: id if near-unique, categorical otherwise
                        if n_unique / n_rows > 0.95 or n_unique > 500:
                            semantic_types[col] = "id"
                        else:
                            semantic_types[col] = "categorical"
                self._log(
                    "    No semantic types in context -"
                    " inferred from dtype as fallback.",
                    "debug",
                )

            # Eligible columns (skip id / datetime / text / unknown)
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
                    intent_a = semantic_types.get(col_a, "unknown")
                    intent_b = semantic_types.get(col_b, "unknown")

                    # Skip if either column has too few non-null rows
                    valid_n = df[[col_a, col_b]].dropna().shape[0]
                    if valid_n < min_sample_size:
                        self._log(
                            f"    Skipping {col_a}|{col_b}: "
                            f"only {valid_n} complete rows (min={min_sample_size}).",
                            "debug",
                        )
                        continue

                    result = _metric_for_pair(df, col_a, col_b, intent_a, intent_b)
                    if result is None:
                        continue

                    value, metric_type = result

                    # Canonical key: lexicographic order so A|B == B|A
                    key = f"{col_a}|{col_b}"

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

            strength_counts = {
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
                plots={},
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

            # Reliability warnings (reuse precomputed flags)
            flags = self.ensure_reliability_flags()

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
