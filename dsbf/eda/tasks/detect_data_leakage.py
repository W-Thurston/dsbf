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
    phase="ml_readiness",
    tags=["leakage", "target"],
    expected_semantic_types=["categorical", "continuous"],
)
class DetectDataLeakage(BaseTask):
    """
    Detects potential data leakage by identifying near-perfectly correlated features.

    Flags any numeric column pair whose absolute Pearson correlation meets or
    exceeds a configurable threshold (default: 0.99). Near-perfect correlation
    almost always indicates that one column is derived from the other, or that
    both encode the same underlying measurement - either case causes target
    leakage if one of the columns encodes post-event information.

    Both columns in each flagged pair receive EDA and ML guidance blurbs so
    the analyst sees the warning regardless of which column they are inspecting.

    Only numeric columns are evaluated. Polars DataFrames are converted to
    pandas since the pandas correlation matrix is used for the pairwise scan.

    Configurable parameters (via config["tasks"]["detect_data_leakage"]):
        correlation_threshold (float): Minimum absolute Pearson correlation to
            flag a pair. Default: 0.99
    """

    def run(self) -> None:
        """
        Execute leakage detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            correlation_threshold = float(
                self.get_task_param("correlation_threshold") or 0.99,
            )

            if is_polars(df):
                # pandas corr() is used for the pairwise scan.
                self._log(
                    "    Converting to pandas: correlation matrix requires pandas.",
                    "debug",
                )
                df = df.to_pandas()

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} "
                "['categorical', 'continuous'] column(s)",
                "debug",
            )

            numeric_df = df.select_dtypes(include="number")
            corr_matrix = numeric_df.corr().abs()
            leakage_pairs: dict[str, float] = {}

            # Scan upper triangle only - each pair is stored once.
            for i, col1 in enumerate(corr_matrix.columns):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val >= correlation_threshold:
                        leakage_pairs[f"{col1}|{col2}"] = float(corr_val)

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
                    "suggested_viz_type": "none",
                    "recommended_section": "Target",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            # Emit guidance for every column in every flagged pair.
            # Both columns receive a blurb so the warning surfaces regardless
            # of which column the analyst is currently inspecting.
            for pair_key, corr_val in leakage_pairs.items():
                col1, col2 = pair_key.split("|")
                self._attach_guidance(col1, col2, corr_val, correlation_threshold)
                self._attach_guidance(col2, col1, corr_val, correlation_threshold)

            # ML impact scoring
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
                        f"Columns '{col1}' and '{col2}' are highly correlated "
                        f"(corr = {corr:.2f}). This may indicate leakage - "
                        "drop one before modeling."
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
        Generate EDA and ML guidance for a column in a near-perfect correlation pair.

        Args:
            col: The column receiving the guidance blurb.
            other_col: The column it is correlated with.
            corr: Absolute Pearson correlation value.
            threshold: The configured leakage detection threshold.

        """
        corr_str: str = f"{corr:.4f}"

        eda_body: str = (
            f"'{col}' has an absolute Pearson correlation of {corr_str} with "
            f"'{other_col}' - near-perfect linear association. This is almost "
            f"certainly not a coincidence. The most common causes are: one column "
            f"was derived from the other (e.g. a ratio, running total, or lagged "
            f"copy), both columns measure the same underlying thing at different "
            f"scales or units, or a join/merge operation duplicated information. "
            f"Verify the data lineage of both columns before trusting any analysis "
            f"that uses them together."
        )

        ml_body: str = (
            f"'{col}' correlates with '{other_col}' at r = {corr_str}. Including "
            f"both in a model is almost always harmful: in linear models the "
            f"coefficients become numerically undefined (perfect multicollinearity); "
            f"in tree models one column will shadow the other completely, wasting a "
            f"split at every node. More critically, if '{other_col}' contains "
            f"information only available after the prediction target is observed "
            f"(e.g. a post-event measurement), including it causes target leakage - "
            f"the model will appear to perform well in training and fail completely "
            f"in deployment. Drop one of the pair. If unsure which is derived, "
            f"trace the data pipeline back to the source."
        )

        metric: dict[str, float | str] = {
            "correlation": round(corr, 6),
            "correlated_with": other_col,
            "threshold": threshold,
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="error",
            title=f"Near-Perfect Correlation with '{other_col}' (r = {corr_str})",
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )
        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level="error",
            title=f"Possible Data Leakage - Perfect Correlation with '{other_col}'",
            body=ml_body.strip(),
            actions=[
                {
                    "action": "drop",
                    "column": col,
                    "detail": (
                        f"Drop one of '{col}' / '{other_col}' - keeping both is "
                        "harmful for all model families"
                    ),
                },
                {
                    "action": "investigate",
                    "column": col,
                    "detail": (
                        "Check whether either column is derived from the other or "
                        "encodes post-event information"
                    ),
                },
            ],
            metric=metric,
        )
