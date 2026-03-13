# dsbf/eda/tasks/detect_data_leakage.py

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
                self.get_task_param("correlation_threshold") or 0.99,
            )

            if is_polars(df):
                self._log(
                    "    Falling back to Pandas: correlation matrix requires "
                    "numeric types",
                    "debug",
                )
                df = df.to_pandas()

            # Use semantic typing to select relevant columns
            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} ['categorical', 'continuous'] "
                "column(s)",
                "debug",
            )
            numeric_df = df.select_dtypes(include="number")
            corr_matrix = numeric_df.corr().abs()
            leakage_pairs: dict[str, float] = {}

            # Scan upper triangle for highly correlated pairs
            for i, col1 in enumerate(corr_matrix.columns):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val >= correlation_threshold:
                        key: str = f"{col1}|{col2}"
                        leakage_pairs[key] = float(corr_val)

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Found {len(leakage_pairs)} highly correlated feature pairs."
                    ),
                },
                data={"leakage_pairs": leakage_pairs},
                metadata={
                    "correlation_threshold": correlation_threshold,
                    "suggested_viz_type": "None",
                    "recommended_section": "Target",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            # Generate per-column guidance for every column involved in a leakage pair.
            # Each pair gets a blurb on both columns so the analyst sees the warning
            # regardless of which column they are inspecting.
            for pair_key, corr_val in leakage_pairs.items():
                col1, col2 = pair_key.split("|")
                self._attach_guidance(col1, col2, corr_val, correlation_threshold)
                self._attach_guidance(col2, col1, corr_val, correlation_threshold)

            # Apply ML scoring to self.output
            if self.get_engine_param("enable_impact_scoring", True) and leakage_pairs:
                first_pair: str = next(iter(leakage_pairs))
                col1, col2 = first_pair.split("|")
                corr: float = leakage_pairs[first_pair]
                tip: str | None = get_recommendation_tip(
                    self.name,
                    {"correlation_with_target": corr},
                )
                self.set_ml_signals(
                    result=self.output,
                    score=1.0,
                    tags=["drop", "check_leakage"],
                    recommendation=tip
                    or (
                        f"Columns '{col1}' and '{col2}' are "
                        "highly correlated (corr = {corr:.2f}). "
                        "This may indicate leakage - drop one before modeling."
                    ),
                )
                self.output.summary["column"] = col1

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(
        self,
        col: str,
        other_col: str,
        corr: float,
        threshold: float,
    ) -> None:
        """
        Generate EDA + ML guidance for a column involved in a near-perfect
        correlation pair.

        Both columns in the pair get a blurb - this way the analyst sees the
        warning regardless of which column they are currently inspecting.
        """
        corr_str: str = f"{corr:.4f}"

        eda_body: str = (
            f"{col} has an absolute Pearson correlation of {corr_str} with {other_col} "
            f"- near-perfect linear association. This is almost certainly not a "
            f"coincidence. The most common causes are: one column was derived from "
            f"the other (e.g. a ratio, running total, or lagged copy), both columns "
            f"measure the same underlying thing at different scales or units, or a "
            f"join or merge operation duplicated information. "
            f"Verify the data lineage of both columns before trusting any analysis "
            f"that uses them together."
        )

        ml_body: str = (
            f"{col} correlates with {other_col} at r = {corr_str}. "
            f"Including both in a model is almost always harmful: in linear models "
            f"the coefficients become numerically undefined (perfect multicollinearity)"
            f"; in tree models one column will shadow the other completely, wasting "
            f"a split at every node. More critically, if {other_col} contains "
            f"information that is only available after the prediction target is "
            f"observed (e.g. it's a post-event measurement), including it causes "
            f"target leakage - the model will appear to perform well in training "
            f"and fail completely in deployment. "
            f"Drop one of the pair. If unsure which is the derived column, "
            f"trace the data pipeline back to the source."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="error",
            title=f"Near-Perfect Correlation with {other_col} (r = {corr_str})",
            body=eda_body.strip(),
            actions=[],
            metric={
                "correlation": round(corr, 6),
                "correlated_with": other_col,
                "threshold": threshold,
            },
        )
        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level="error",
            title=f"Possible Data Leakage - Perfect Correlation with {other_col}",
            body=ml_body.strip(),
            actions=[
                {
                    "action": "drop",
                    "column": col,
                    "detail": f"Drop one of {col} / {other_col} "
                    "- keeping both is harmful for all model families",
                },
                {
                    "action": "investigate",
                    "column": col,
                    "detail": "Check whether either column is derived from the other "
                    "or encodes post-event information",
                },
            ],
            metric={
                "correlation": round(corr, 6),
                "correlated_with": other_col,
                "threshold": threshold,
            },
        )
