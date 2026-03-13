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
    tags=["redundancy", "null-equivalent"],
    expected_semantic_types=["any"],
)
class DetectConstantColumns(BaseTask):
    """
    Identifies columns with only one unique value in the dataset.

    Works for both Polars and Pandas DataFrames.
    """

    def run(self) -> None:
        """
        Executes the constant column detection logic.

        Produces a TaskResult with a list of constant column names.
        """
        try:
            # ctx = self.context
            df = self.input_data
            constant_columns: list[str]

            # Use semantic typing to select relevant columns
            matched_cols: list[str] = None
            excluded: tuple[list[str], dict[str, str]] = None
            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} ['categorical', 'text'] column(s)",
                "debug",
            )

            if is_polars(df):
                # Use Polars' n_unique per column
                constant_columns = [
                    col for col in df.columns if df[col].n_unique() == 1
                ]
            else:
                # Pandas variant
                constant_columns = [col for col in df.columns if df[col].nunique() == 1]

            # Build TaskResult
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": f"Found {len(constant_columns)} constant column(s).",
                },
                data={"constant_columns": constant_columns},
                metadata={
                    "engine": "polars" if is_polars(df) else "pandas",
                    "suggested_viz_type": "None",
                    "recommended_section": "Redundancy",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            # Generate per-column guidance
            for col in constant_columns:
                self._attach_guidance(col)

            # Apply ML scoring to self.output
            if (
                self.get_engine_param("enable_impact_scoring", True)
                and constant_columns
            ):
                col: str = constant_columns[0]
                result: TaskResult = self.output
                if result:
                    tip: str | None = get_recommendation_tip(self.name, {"n_unique": 1})
                    self.set_ml_signals(
                        result=result,
                        score=1.0,
                        tags=["drop"],
                        recommendation=tip
                        or (
                            f"Column '{col}' has a constant value."
                            " Drop it to avoid redundant features."
                        ),
                    )
                    result.summary["column"] = col

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
        """Generate EDA + ML guidance for a confirmed constant column."""
        eda_body: str = (
            f"{col} has only one unique value across all rows — it is a constant "
            f"column. It carries no information and cannot distinguish between "
            f"observations. Verify this is not a data loading artifact, a column "
            f"populated in error, or a filter applied upstream that collapsed "
            f"the dataset to a single value."
        )

        ml_body: str = (
            f"{col} is a constant column with zero variance. Most ML frameworks "
            f"will silently drop or error on constant features during fitting. "
            f"Drop {col} before modeling."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="error",
            title="Constant Column",
            body=eda_body.strip(),
            actions=[],
            metric={"n_unique": 1},
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level="error",
            title="Constant Column — Drop Before Modeling",
            body=ml_body.strip(),
            actions=[
                {
                    "action": "drop",
                    "column": col,
                    "detail": "Zero variance — provides no signal to any model",
                },
            ],
            metric={"n_unique": 1},
        )
