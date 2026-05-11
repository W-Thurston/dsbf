# dsbf/eda/tasks/detect_constant_columns.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    display_name="Detect Constant Columns",
    description="Flags columns with a single unique value.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    phase="eda",
    tags=["redundancy", "null-equivalent"],
    expected_semantic_types=["any"],
)
class DetectConstantColumns(BaseTask):
    """
    Identifies columns with exactly one unique value across all rows.

    A constant column carries no information - every row is identical for that
    feature. This is most often caused by a data loading artefact, an upstream
    filter that collapsed variation, or a column populated in error.

    Supports both Polars and Pandas DataFrames.

    Output is consumed by the Quality tab (Validity dimension) and the
    ML Readiness tab (Unusable Features dimension).
    """

    def run(self) -> None:
        """
        Execute constant column detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run_native()

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No columns found — constant column detection skipped.",
                    excluded,
                )
                return

            if is_polars(df):
                constant_columns: list[str] = [
                    col for col in df.columns if df[col].n_unique() == 1
                ]
            else:
                constant_columns = [col for col in df.columns if df[col].nunique() == 1]

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": f"Found {len(constant_columns)} constant column(s).",
                },
                data={"constant_columns": constant_columns},
                metadata={
                    "engine": "polars" if is_polars(df) else "pandas",
                    "suggested_viz_type": "none",
                    "recommended_section": "Redundancy",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col in constant_columns:
                self._attach_guidance(col)

            # ML impact scoring
            if (
                self.get_engine_param("enable_impact_scoring", True)
                and constant_columns
            ):
                top_col: str = constant_columns[0]
                tip: str | None = get_recommendation_tip(self.name, {"n_unique": 1})
                self.set_ml_signals(
                    result=self.output,
                    score=1.0,
                    tags=["drop"],
                    recommendation=tip
                    or (
                        f"Column '{top_col}' has a constant value. "
                        "Drop it to avoid redundant features."
                    ),
                )
                self.output.summary["column"] = top_col

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str) -> None:
        """
        Generate EDA and ML guidance blurbs for a confirmed constant column.

        Args:
            col: Column name confirmed to have exactly one unique value.

        """
        eda_body: str = (
            f"'{col}' has only one unique value across all rows - it is a constant "
            "column. It carries no information and cannot distinguish between "
            "observations. Verify this is not a data loading artefact, a column "
            "populated in error, or a filter applied upstream that collapsed "
            "the dataset to a single value."
        )

        ml_body: str = (
            f"'{col}' is a constant column with zero variance. Most ML frameworks "
            "will silently drop or error on constant features during fitting. "
            f"Drop '{col}' before modeling."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="error",
            title="Constant Column - Zero Information",
            body=eda_body.strip(),
            actions=[],
            metric={"n_unique": 1},
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level="error",
            title="Constant Column - Drop Before Modeling",
            body=ml_body.strip(),
            actions=[
                {
                    "action": "drop",
                    "column": col,
                    "detail": "Zero variance - provides no signal to any model",
                },
            ],
            metric={"n_unique": 1},
        )
