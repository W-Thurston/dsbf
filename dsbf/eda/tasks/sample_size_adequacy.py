# dsbf/eda/tasks/sample_size_adequacy.py

from typing import Any, Literal

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

# ── Rule-of-thumb thresholds ───────────────────────────────────────────────────
#
# These are well-established heuristics, not statistical guarantees.
# All are framed as minimum rows-per-feature ratios or absolute minimums.
#
# References:
#   - Linear models: 10-20 observations per predictor (Harrell 2001, Peduzzi 1996)
#   - Logistic regression: ~20 events per predictor (EPV rule)
#   - Tree-based models: less sensitive; 100 rows/feature is conservative
#   - Neural networks: 10x parameters is a common starting heuristic
#   - General ML: 1000-row floor for meaningful train/val/test splits

_RULES: list[dict] = [
    {
        "rule": "linear_model",
        "label": "Linear / Logistic Regression",
        "min_rows_per_feature": 20,
        "absolute_min": 50,
        "rationale": (
            "Linear models estimate one coefficient per feature. "
            "Fewer than 20 rows per predictor produces unstable coefficient "
            "estimates with wide confidence intervals (Harrell 2001)."
        ),
    },
    {
        "rule": "tree_model",
        "label": "Tree-Based Models (RF, XGBoost, LightGBM)",
        "min_rows_per_feature": 10,
        "absolute_min": 100,
        "rationale": (
            "Tree models are less sensitive to the rows-per-feature ratio "
            "but still require enough samples to find meaningful splits. "
            "Below ~100 rows, overfitting risk is severe."
        ),
    },
    {
        "rule": "general_ml",
        "label": "General ML / Train-Val-Test Split",
        "min_rows_per_feature": 5,
        "absolute_min": 1000,
        "rationale": (
            "A 60/20/20 train/val/test split requires ~1000 rows to produce "
            "a validation set large enough for reliable metric estimation. "
            "Below this, evaluation variance dominates reported performance."
        ),
    },
]


def _evaluate_rule(n_rows: int, n_features: int, rule: dict) -> dict:
    """
    Evaluate a single adequacy rule against the dataset dimensions.

    Args:
        n_rows: Number of rows in the dataset.
        n_features: Number of analytical feature columns (excludes id/datetime).
        rule: Rule dict from _RULES.

    Returns:
        Dict with ``rule``, ``label``, ``required_rows``, ``actual_rows``,
        ``adequate``, ``shortfall``, and ``rationale`` keys.

    """
    required_by_ratio = n_features * rule["min_rows_per_feature"]
    required = max(required_by_ratio, rule["absolute_min"])
    adequate = n_rows >= required
    shortfall: int = max(0, required - n_rows) if not adequate else 0

    return {
        "rule": rule["rule"],
        "label": rule["label"],
        "required_rows": required,
        "actual_rows": n_rows,
        "adequate": adequate,
        "shortfall": shortfall,
        "rationale": rule["rationale"],
    }


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="sample_size_adequacy",
    display_name="Sample Size Adequacy",
    description=(
        "Evaluates whether the dataset has enough rows for the number of features "
        "and common modeling approaches. Applies rule-of-thumb thresholds for "
        "linear models, tree-based models, and general ML train/val/test splits."
    ),
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    phase="ml_readiness",
    domain="core",
    runtime_estimate="fast",
    tags=["sample_size", "ml_readiness", "overview"],
    expected_semantic_types=["any"],
)
class SampleSizeAdequacy(BaseTask):
    """
    Evaluate whether the dataset has sufficient rows for modeling.

    Applies three independent rule-of-thumb checks:

    - **Linear / Logistic Regression**: ≥ 20 rows per feature, minimum 50 rows.
      Based on the events-per-variable (EPV) principle - fewer than 20 obs/feature
      produces unstable coefficient estimates (Harrell 2001).
    - **Tree-Based Models**: ≥ 10 rows per feature, minimum 100 rows.
      Tree models are less sensitive to the ratio but below ~100 rows
      overfitting is severe regardless of algorithm.
    - **General ML / Train-Val-Test split**: ≥ 5 rows per feature, minimum
      1000 rows. A 60/20/20 split needs ~1000 rows for a validation set
      large enough to yield reliable metric estimates.

    Feature count excludes columns typed as ``id``, ``datetime``, or ``unknown``
    since these are not analytical predictors.

    For each failing rule an ML guidance blurb is emitted explaining the
    shortfall and concrete remediation options. Passing rules produce no
    guidance - the overall signal is surfaced through the summary verdict.

    Configurable parameters (via config["tasks"]["sample_size_adequacy"]):
        n_features_override (int): Override the feature count used in ratio
            calculations. Useful when only a subset of columns will be used
            as predictors. Default: derived from inferred semantic types.
    """

    def run(self) -> None:  # noqa: C901, PLR0912
        """
        Execute sample size adequacy evaluation and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_cols)} column(s)", "debug")

            n_rows: int = len(df)

            # Feature count: exclude id/datetime/unknown columns - they are not
            # predictors. Use an override if configured.
            override_raw: Any | None = self.get_task_param("n_features_override")
            if override_raw is not None:
                n_features = int(override_raw)
            else:
                semantic_types: dict[str, str] = {}
                if self.context:
                    semantic_types = self.context.get_metadata("semantic_types") or {}
                skip_intents: set[str] = {"id", "datetime", "unknown"}
                if semantic_types:
                    n_features: int = sum(
                        1
                        for col in df.columns
                        if semantic_types.get(col, "unknown") not in skip_intents
                    )
                else:
                    # Fallback when infer_types hasn't run: count all columns
                    n_features = len(df.columns)

            n_features = max(n_features, 1)  # guard against zero-feature edge case

            self._log(
                f"    n_rows={n_rows}, n_features={n_features} "
                f"(rows-per-feature ratio: {n_rows / n_features:.1f})",
                "debug",
            )

            rule_results: dict[str, dict] = {}
            for rule in _RULES:
                evaluation: dict = _evaluate_rule(n_rows, n_features, rule)
                rule_results[rule["rule"]] = evaluation
                status: str = (
                    "✓ adequate"
                    if evaluation["adequate"]
                    else (f"✗ shortfall of {evaluation['shortfall']:,} rows")
                )
                self._log(f"    {rule['label']}: {status}", "debug")

            n_passing: int = sum(1 for v in rule_results.values() if v["adequate"])
            n_failing: int = len(rule_results) - n_passing

            # Overall verdict: all pass → ready; some fail → limited;
            # linear model fails → severely limited
            if n_failing == 0:
                verdict = "adequate"
            elif rule_results["linear_model"]["adequate"]:
                verdict = "limited"  # tree/general fail, but linear is ok
            elif rule_results["tree_model"]["adequate"]:
                verdict = "limited"  # only tree passes
            else:
                verdict = "insufficient"  # nothing passes

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Dataset has {n_rows:,} rows and {n_features} feature(s). "
                        f"Verdict: {verdict} "
                        f"({n_passing}/{len(rule_results)} rules pass)."
                    ),
                    "n_rows": n_rows,
                    "n_features": n_features,
                    "rows_per_feature": round(n_rows / n_features, 2),
                    "verdict": verdict,
                    "rules_passing": n_passing,
                    "rules_failing": n_failing,
                },
                data=rule_results,
                metadata={
                    "n_features_source": (
                        "override" if override_raw is not None else "semantic_types"
                    ),
                    "suggested_viz_type": "table",
                    "recommended_section": "Overview",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for evaluation in rule_results.values():
                if not evaluation["adequate"]:
                    self._attach_guidance(evaluation)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, evaluation: dict) -> None:
        """
        Generate an ML guidance blurb for a failing adequacy rule.

        Args:
            evaluation: Rule evaluation dict from ``_evaluate_rule``.

        """
        label = evaluation["label"]
        required = evaluation["required_rows"]
        actual = evaluation["actual_rows"]
        shortfall = evaluation["shortfall"]
        rationale = evaluation["rationale"]
        rule = evaluation["rule"]

        body: str = (
            f"The dataset has {actual:,} rows but {label} requires "
            f"approximately {required:,} rows given the current feature count "
            f"(shortfall: {shortfall:,} rows). {rationale} "
            f"Options: collect more data, reduce the feature count by removing "
            f"low-signal or redundant columns, use a simpler model family, or "
            f"apply regularisation aggressively to compensate for limited data. "
            f"Cross-validation with many folds (e.g. leave-one-out or 10-fold) "
            f"is recommended to maximise use of available data."
        )

        # Severity: linear model failure is more serious than tree/general
        level: Literal["info", "warn"] = "warn" if rule == "linear_model" else "info"

        self.add_guidance(
            result=self.output,
            column="__dataset__",
            phase="ml",
            level=level,
            title=f"Insufficient Data for {label} ({actual:,} of ~{required:,} rows)",
            body=body.strip(),
            actions=[
                {
                    "action": "collect_more_data",
                    "detail": f"Acquire ~{shortfall:,} additional rows",
                },
                {
                    "action": "reduce_features",
                    "detail": (
                        "Remove low-signal or redundant columns to lower the "
                        "required row threshold"
                    ),
                },
                {
                    "action": "use_cross_validation",
                    "detail": "K-fold or leave-one-out CV to maximise data use",
                },
            ],
            metric={
                "rule": rule,
                "required_rows": required,
                "actual_rows": actual,
                "shortfall": shortfall,
            },
        )
