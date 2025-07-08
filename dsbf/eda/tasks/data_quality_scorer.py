# dsbf/eda/tasks/data_quality_scorer.py

from typing import Any, Dict, List

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    name="data_quality_scorer",
    display_name="Data Quality Scorer",
    description=(
        "Aggregates profiling results to generate an overall"
        " data health score and breakdown."
    ),
    tags=["scoring", "summary", "meta"],
    stage="report",
    domain="core",
    profiling_depth="basic",
    runtime_estimate="fast",
    expected_semantic_types=["any"],
)
class DataQualityScorer(BaseTask):
    """
    The DataQualityScorer aggregates the results of prior profiling tasks to generate
    an overall 0-100 data health score. It also provides category-level breakdowns,
    explanations for deductions, top issues contributing to score degradation, and
    recommended actions.

    Categories evaluated:
    - Completeness (missing values)
    - Consistency (type mismatches, format violations)
    - Distribution integrity (skew, outliers, zero variance)
    - Redundancy (high collinearity)
    - Drift (if reference data available)

    Category weights are read from task-specific config, defaulting to equal weights.
    """

    def run(self) -> None:
        """
        Execute the task. Scans the context results and reliability flags to compute
        category scores and overall health score. Generates structured output via
        TaskResult.
        """
        if self.context is None:
            raise RuntimeError("AnalysisContext is not set in this task.")

        results: Dict[str, TaskResult] = self.context.results
        flags: Dict[str, Any] = self.context.reliability_flags or {}

        # Use semantic typing to select relevant columns
        matched_cols, excluded = self.get_columns_by_intent()
        self._log(f"Processing {len(matched_cols)} column(s)", "debug")

        category_scores: Dict[str, int] = {}
        explanations: List[str] = []
        top_issues: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        # --- COMPLETENESS ---
        missing_tasks = [
            r
            for r in results.values()
            if r.name in {"null_summary", "missingness_heatmap"}
            and r.status == "success"
        ]
        missing_cols = set()
        high_missing_cols = set()

        for task in missing_tasks:
            data = task.data or {}
            for col, stats in data.get("missingness", {}).items():
                pct = stats.get("percent_missing", 0)
                if pct > 0:
                    missing_cols.add(col)
                if pct > 0.3:
                    high_missing_cols.add(col)

        completeness_deduction = 0
        if missing_cols:
            completeness_deduction += min(len(missing_cols) * 2, 50)
            explanations.append(f"{len(missing_cols)} column(s) have missing values.")
        if high_missing_cols:
            completeness_deduction += 10
            explanations.append(
                f"{len(high_missing_cols)} column(s) have >30% missing."
            )

        completeness_score = max(100 - completeness_deduction, 0)
        category_scores["completeness"] = completeness_score

        # --- CONSISTENCY ---
        consistency_score = 100
        consistency_deduction = 0

        for task_name in ["detect_mixed_type_columns", "regex_format_violations"]:
            task = results.get(task_name)
            if task and task.status == "success" and task.data:
                issues = task.data.get("columns_with_issues", [])
                count = len(issues)
                if count:
                    consistency_deduction += count * 5
                    explanations.append(f"{count} consistency issue(s) in {task_name}.")
                    top_issues.append({"task": task_name, "count": count})
                    recommendations.extend(task.recommendations or [])

        consistency_score = max(100 - consistency_deduction, 0)
        category_scores["consistency"] = consistency_score

        # --- DISTRIBUTION ---
        skew_vals: Dict[str, float] = flags.get("skew_vals", {})
        zero_var_cols: List[str] = flags.get("zero_variance_cols", [])
        skewed = [col for col, val in skew_vals.items() if abs(val) > 2]
        outliers: bool = flags.get("extreme_outliers", False)

        dist_deduction = 0
        if skewed:
            dist_deduction += min(len(skewed) * 5, 30)
            explanations.append(f"{len(skewed)} column(s) have high skew.")
        if zero_var_cols:
            dist_deduction += 10
            explanations.append("Columns with near-zero variance detected.")
        if outliers:
            dist_deduction += 10
            explanations.append("Extreme outliers detected in data.")

        distribution_score = max(100 - dist_deduction, 0)
        category_scores["distribution"] = distribution_score

        # --- REDUNDANCY ---
        redundancy_score = 100
        red_task = results.get("detect_collinear_features")
        if red_task and red_task.status == "success" and red_task.data:
            redundant_pairs = red_task.data.get("redundant_pairs", [])
            n_redundant = len(redundant_pairs)
            redundancy_deduction = min(n_redundant * 2, 30)
            redundancy_score = max(100 - redundancy_deduction, 0)
            if n_redundant:
                explanations.append(
                    f"{n_redundant} pairs of highly correlated features."
                )
                top_issues.append(
                    {"task": "detect_collinear_features", "count": n_redundant}
                )
                recommendations.extend(red_task.recommendations or [])

        category_scores["redundancy"] = redundancy_score

        # --- DRIFT ---
        drift_score = 100
        drift_task = results.get("detect_feature_drift")
        if drift_task and drift_task.status == "success" and drift_task.data:
            drifted = drift_task.data.get("drifted_features", [])
            n_drifted = len(drifted)
            drift_deduction = min(n_drifted * 5, 30)
            drift_score = max(100 - drift_deduction, 0)
            if n_drifted:
                explanations.append(
                    f"{n_drifted} feature(s) show signs of distributional drift."
                )
                top_issues.append({"task": "detect_feature_drift", "count": n_drifted})
                recommendations.extend(drift_task.recommendations or [])

        category_scores["drift"] = drift_score

        # --- OVERALL SCORE (weighted average) ---
        default_weights: Dict[str, float] = {
            "completeness": 1,
            "consistency": 1,
            "distribution": 1,
            "redundancy": 1,
            "drift": 1,
        }
        weights: Dict[str, float] = (
            self.get_task_param("weights", default_weights) or default_weights
        )

        total_weight: float = sum(weights.get(cat, 0.0) for cat in category_scores)
        weighted_sum: float = sum(
            category_scores[cat] * weights.get(cat, 0.0) for cat in category_scores
        )
        overall_score: int = (
            int(round(weighted_sum / total_weight)) if total_weight > 0 else 0
        )

        # --- PLOTTING ---
        plots: dict[str, dict[str, Any]] = {}

        try:
            all_scores = dict(category_scores)
            all_scores["overall"] = overall_score

            series = pd.Series(all_scores).sort_index()

            save_path = self.get_output_path("data_quality_score_breakdown.png")
            static = PlotFactory.plot_barplot_static(
                series, save_path=save_path, title="Data Quality Score Breakdown"
            )
            interactive = PlotFactory.plot_barplot_interactive(
                series,
                title="Data Quality Score Breakdown",
                annotations=[f"{k.capitalize()}: {v}" for k, v in all_scores.items()],
            )

            plots["data_quality_scores"] = {
                "static": static["path"],
                "interactive": interactive,
            }

        except Exception as e:
            self._log(
                f"[PlotFactory] Skipped quality score barplot: {e}", level="debug"
            )

        # --- FINAL RESULT OBJECT ---
        self.output = TaskResult(
            name=self.name,
            summary={
                "overall_score": overall_score,
                "category_breakdown": category_scores,
                "category_weights": weights,
                "explanation": explanations,
                "top_issues": top_issues,
            },
            recommendations=recommendations,
            metadata={
                "scoring_method": "configurable weighted average",
                "weights": weights,
                "suggested_viz_type": "bar",
                "recommended_section": "Summary",
                "display_priority": "high",
                "excluded_columns": excluded,
                "column_types": self.get_column_type_info(
                    matched_cols + list(excluded.keys())
                ),
            },
            plots=plots,
        )
