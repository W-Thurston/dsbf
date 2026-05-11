# dsbf/eda/tasks/imputation_strategy_suggestions.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result

# ── Imputation strategy decision logic ────────────────────────────────────────
#
# Strategy is selected based on three signals, in priority order:
#
#   1. Null percentage tier - high missingness changes what is even possible
#   2. Semantic intent - categorical vs continuous vs datetime require different methods
#   3. Distribution shape - skewed continuous columns prefer median over mean
#
# Tiers:
#   severe:      pct >= 0.5  - imputation likely introduces more bias than dropping
#   significant: pct >= 0.2  - mean/mode unreliable; prefer model-based or indicator
#   moderate:    pct >= 0.05 - standard imputation applicable; method depends on dist
#   low:         pct <  0.05 - any method works; mean/mode/ffill all reasonable


def _select_strategy(
    col: str,
    pct: float,
    intent: str,
    is_skewed: bool,
) -> dict:
    """
    Select an imputation strategy for a single column.

    Args:
        col: Column name.
        pct: Proportion of null values (0.0-1.0).
        intent: Semantic intent from infer_types
            (``"continuous"``, ``"categorical"``, ``"datetime"``, etc.).
        is_skewed: True if the column has been flagged as highly skewed by
            detect_skewness or has skewness above threshold in reliability flags.

    Returns:
        Dict with ``strategy``, ``method``, ``tier``, ``add_indicator``,
        and ``rationale`` keys.

    """
    # --- Tier classification ---
    if pct >= 0.5:
        tier = "severe"
    elif pct >= 0.2:
        tier = "significant"
    elif pct >= 0.05:
        tier = "moderate"
    else:
        tier = "low"

    # --- Strategy selection ---
    if tier == "severe":
        return {
            "strategy": "drop_or_indicator",
            "method": "drop column or add binary is_missing indicator",
            "tier": tier,
            "add_indicator": True,
            "rationale": (
                f"{pct:.1%} missing. Imputing more than half the values "
                "introduces substantial bias regardless of method. Drop the column "
                "unless the fact of missingness is itself informative, in which "
                "case retain a binary is_missing indicator and drop the original."
            ),
        }

    if intent == "categorical":
        if tier in ("significant", "moderate"):
            return {
                "strategy": "mode_with_indicator",
                "method": "mode imputation + is_missing indicator",
                "tier": tier,
                "add_indicator": tier == "significant",
                "rationale": (
                    f"{pct:.1%} missing in a categorical column. Mode imputation "
                    "fills with the most frequent category. At this missingness "
                    "level also add a binary is_missing indicator to preserve "
                    "the signal from the gap pattern."
                ),
            }
        # low
        return {
            "strategy": "mode",
            "method": "mode imputation",
            "tier": tier,
            "add_indicator": False,
            "rationale": (
                f"{pct:.1%} missing in a categorical column. "
                "Mode imputation (most frequent category) is appropriate "
                "at this low missingness level."
            ),
        }

    if intent == "datetime":
        return {
            "strategy": "forward_fill",
            "method": "forward fill (ffill) or interpolation",
            "tier": tier,
            "add_indicator": tier in ("significant", "severe"),
            "rationale": (
                f"{pct:.1%} missing in a datetime column. Forward fill propagates "
                "the last known timestamp, which is appropriate for sequential "
                "data. Consider linear interpolation if timestamps represent "
                "evenly spaced intervals."
            ),
        }

    # --- Continuous columns ---
    if tier == "significant":
        if is_skewed:
            return {
                "strategy": "median_with_indicator",
                "method": "median imputation + is_missing indicator",
                "tier": tier,
                "add_indicator": True,
                "rationale": (
                    f"{pct:.1%} missing in a skewed continuous column. "
                    "Mean imputation is pulled by the long tail - median is "
                    "more robust. At this missingness level also add a binary "
                    "is_missing indicator."
                ),
            }
        return {
            "strategy": "mean_or_knn_with_indicator",
            "method": "mean or KNN imputation + is_missing indicator",
            "tier": tier,
            "add_indicator": True,
            "rationale": (
                f"{pct:.1%} missing in a continuous column. Mean imputation "
                "is reasonable for symmetric distributions but introduces "
                "bias at this level. KNN imputation uses similar rows and "
                "is preferable when correlated features exist. Add a binary "
                "is_missing indicator alongside the imputed values."
            ),
        }

    if tier == "moderate":
        if is_skewed:
            return {
                "strategy": "median",
                "method": "median imputation",
                "tier": tier,
                "add_indicator": False,
                "rationale": (
                    f"{pct:.1%} missing in a skewed continuous column. "
                    "Median is preferred over mean for skewed distributions - "
                    "it is not pulled by extreme values."
                ),
            }
        return {
            "strategy": "mean",
            "method": "mean imputation",
            "tier": tier,
            "add_indicator": False,
            "rationale": (
                f"{pct:.1%} missing in a continuous column. "
                "Mean imputation is appropriate at this missingness level "
                "for symmetric distributions."
            ),
        }

    # low tier, continuous
    return {
        "strategy": "mean_or_median",
        "method": "mean (symmetric) or median (skewed)",
        "tier": tier,
        "add_indicator": False,
        "rationale": (
            f"{pct:.1%} missing - any standard imputation method is appropriate. "
            "Use mean for symmetric distributions, median for skewed ones."
        ),
    }


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="imputation_strategy_suggestions",
    display_name="Imputation Strategy Suggestions",
    description=(
        "Recommends a per-column imputation strategy based on missingness "
        "percentage, semantic type, and distribution shape. Reads from "
        "summarize_nulls and infer_types."
    ),
    depends_on=["infer_types", "summarize_nulls"],
    profiling_depth="standard",
    stage="cleaned",
    phase="ml_readiness",
    domain="core",
    runtime_estimate="fast",
    tags=["missingness", "imputation", "ml_readiness"],
    expected_semantic_types=["any"],
)
class ImputationStrategySuggestions(BaseTask):
    """
    Recommend a per-column imputation strategy for columns with missing values.

    Reads null percentages from the ``summarize_nulls`` task result in context.
    Falls back to computing null percentages directly from the DataFrame if the
    task has not run.

    Strategy selection is driven by three signals in priority order:

    1. **Null percentage tier** - severe (≥50%), significant (≥20%),
       moderate (≥5%), low (<5%).
    2. **Semantic intent** - categorical columns use mode; datetime columns
       use forward fill; continuous columns use mean/median/KNN.
    3. **Distribution shape** - skewed continuous columns prefer median over
       mean. Skew is detected by checking the ``detect_skewness`` task result
       in context, or by falling back to a direct skewness calculation.

    Only columns meeting the ``min_null_pct`` threshold (default 0.01, i.e. 1%)
    are evaluated - completely clean columns produce no suggestion.

    For each column with a suggestion an ML guidance blurb is emitted with
    the recommended method and rationale. Severe missingness columns receive
    a ``warn`` level blurb; others receive ``info``.

    Configurable parameters (via config["tasks"]["imputation_strategy_suggestions"]):
        min_null_pct (float): Minimum null proportion to evaluate a column.
            Default: 0.01 (1%)
        skew_threshold (float): Absolute skewness above which a continuous
            column is considered skewed. Default: 1.0
    """

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """
        Execute imputation strategy suggestion and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run()

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No eligible columns found — imputation suggestions skipped.",
                    excluded,
                )
                return

            min_null_raw: Any | None = self.get_task_param("min_null_pct")
            min_null_pct: float = (
                float(min_null_raw) if min_null_raw is not None else 0.01
            )

            skew_thresh_raw: Any | None = self.get_task_param("skew_threshold")
            skew_threshold: float = (
                float(skew_thresh_raw) if skew_thresh_raw is not None else 1.0
            )

            # --- Read null percentages from summarize_nulls if available ---
            null_percentages: dict[str, float] = {}
            if self.context:
                nulls_result: TaskResult | None = self.context.results.get(
                    "summarize_nulls",
                )
                if nulls_result and nulls_result.status == "success":
                    null_percentages = nulls_result.data.get("null_percentages") or {}

            if not null_percentages:
                self._log(
                    "    summarize_nulls not in context - computing null "
                    "percentages directly.",
                    "debug",
                )
                n_rows: int = len(df)
                if n_rows > 0:
                    null_percentages = {
                        col: df[col].isna().sum() / n_rows for col in df.columns
                    }

            # --- Read semantic types ---
            semantic_types: dict[str, str] = {}
            if self.context:
                semantic_types = self.context.get_metadata("semantic_types") or {}

            # --- Detect skewed columns ---
            # Prefer detect_skewness task result; fall back to direct computation.
            skewed_cols: set[str] = set()
            if self.context:
                skew_result: TaskResult | None = self.context.results.get(
                    "detect_skewness",
                )
                if skew_result and skew_result.status == "success":
                    skew_data: dict[str, Any] = skew_result.data or {}
                    for col, skew_info in skew_data.items():
                        skew_val = (
                            skew_info.get("skewness")
                            if isinstance(skew_info, dict)
                            else skew_info
                        )
                        if (
                            skew_val is not None
                            and abs(float(skew_val)) >= skew_threshold
                        ):
                            skewed_cols.add(col)

            if not skewed_cols:
                # Fallback: compute skewness directly for numeric columns
                numeric_cols = df.select_dtypes(include="number").columns
                for col in numeric_cols:
                    series = df[col].dropna()
                    if len(series) >= 3:  # noqa: PLR2004
                        try:
                            sk = float(series.skew())
                            if abs(sk) >= skew_threshold:
                                skewed_cols.add(col)
                        except Exception:  # noqa: BLE001, S110
                            pass

            # --- Build suggestions ---
            suggestions: dict[str, dict] = {}

            for col, pct in null_percentages.items():
                if pct < min_null_pct:
                    continue
                if col not in df.columns:
                    continue

                intent: str = semantic_types.get(col, "continuous")
                is_skewed: bool = col in skewed_cols

                suggestion: dict = _select_strategy(col, pct, intent, is_skewed)
                suggestion["null_pct"] = round(pct, 4)
                suggestion["column"] = col
                suggestions[col] = suggestion

                self._log(
                    f"    '{col}' ({pct:.1%} null, {intent}): {suggestion['strategy']}",
                    "debug",
                )

            n_severe: int = sum(
                1 for s in suggestions.values() if s["tier"] == "severe"
            )
            self._log(
                f"    {len(suggestions)} column(s) with imputation suggestions "
                f"({n_severe} severe).",
                "debug",
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Imputation strategies suggested for "
                        f"{len(suggestions)} column(s) "
                        f"({n_severe} with severe missingness)."
                    ),
                    "suggestion_count": len(suggestions),
                    "severe_count": n_severe,
                },
                data={"suggestions": suggestions},
                metadata={
                    "min_null_pct": min_null_pct,
                    "skew_threshold": skew_threshold,
                    "suggested_viz_type": "table",
                    "recommended_section": "Missingness",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, suggestion in suggestions.items():
                self._attach_guidance(col, suggestion)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, suggestion: dict) -> None:
        """
        Generate an ML guidance blurb for a column's imputation suggestion.

        Args:
            col: Column name.
            suggestion: Strategy dict from ``_select_strategy`` with null_pct added.

        """
        method = suggestion["method"]
        rationale = suggestion["rationale"]
        tier = suggestion["tier"]
        add_indicator = suggestion["add_indicator"]
        pct = suggestion["null_pct"]

        actions: list[dict[str, str]] = [
            {
                "action": "impute",
                "method": method,
                "column": col,
                "detail": rationale,
            },
        ]

        if add_indicator:
            actions.append(
                {
                    "action": "add_indicator",
                    "method": f"df['{col}_missing'] = df['{col}'].isnull().astype(int)",
                    "column": col,
                    "detail": "Binary indicator preserves the missingness signal",
                },
            )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level="warn" if tier == "severe" else "info",
            title=f"Imputation: {method} ({pct:.1%} missing)",
            body=rationale.strip(),
            actions=actions,
            metric={
                "null_pct": pct,
                "strategy": suggestion["strategy"],
                "tier": tier,
                "add_indicator": add_indicator,
            },
        )
