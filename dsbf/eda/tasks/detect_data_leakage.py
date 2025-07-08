# dsbf/eda/tasks/detect_data_leakage.py

from typing import Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    display_name="Detect Data Leakage",
    description="Heuristically detects columns that may leak target information.",
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="modeling",
    domain="core",
    runtime_estimate="moderate",
    tags=["leakage", "target"],
    expected_semantic_types=["categorical", "continuous"],
)
class DetectDataLeakage(BaseTask):
    """
    Detects potential data leakage by identifying highly correlated numeric features.

    Flags any column pairs with absolute correlation >= threshold.
    """

    def run(self) -> None:
        """
        Run the data leakage detection task.

        Produces a TaskResult containing:
        - leakage_pairs: dict of "col1|col2" → float correlation
        """

        try:

            # ctx = self.context
            df = self.input_data

            correlation_threshold = float(
                self.get_task_param("correlation_threshold") or 0.99
            )

            if is_polars(df):
                self._log(
                    "Falling back to Pandas: correlation matrix requires numeric types",
                    "debug",
                )
                df = df.to_pandas()

            # Use semantic typing to select relevant columns
            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"Processing {len(matched_cols)} ['categorical', 'continuous'] "
                "column(s)",
                "debug",
            )
            numeric_df = df.select_dtypes(include="number")
            corr_matrix = numeric_df.corr().abs()
            leakage_pairs: Dict[str, float] = {}

            # Scan upper triangle for highly correlated pairs
            for i, col1 in enumerate(corr_matrix.columns):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val >= correlation_threshold:
                        key = f"{col1}|{col2}"
                        leakage_pairs[key] = float(corr_val)

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Found {len(leakage_pairs)} highly correlated feature pairs."
                    )
                },
                data={"leakage_pairs": leakage_pairs},
                metadata={
                    "correlation_threshold": correlation_threshold,
                    "suggested_viz_type": "None",
                    "recommended_section": "Target",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys())
                    ),
                },
            )

            # Apply ML scoring to self.output
            if self.get_engine_param("enable_impact_scoring", True) and leakage_pairs:
                first_pair = next(iter(leakage_pairs))
                col1, col2 = first_pair.split("|")
                corr = leakage_pairs[first_pair]
                tip = get_recommendation_tip(
                    self.name, {"correlation_with_target": corr}
                )
                self.set_ml_signals(
                    result=self.output,
                    score=1.0,
                    tags=["drop", "check_leakage"],
                    recommendation=tip
                    or (
                        f"Columns '{col1}' and '{col2}' are "
                        "highly correlated (corr = {corr:.2f}). "
                        "This may indicate leakage — drop one before modeling."
                    ),
                )
                self.output.summary["column"] = col1

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
