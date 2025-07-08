# dsbf/eda/tasks/detect_duplicate_columns.py

from typing import List, Tuple

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect Duplicate Columns",
    description="Finds columns that contain identical values.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    tags=["redundancy", "duplicates"],
    expected_semantic_types=["any"],
)
class DetectDuplicateColumns(BaseTask):
    """
    Detects columns that are exact duplicates of one another.
    Compares all unique column pairs using .equals().
    """

    def run(self) -> None:
        """
        Run duplicate column detection logic.
        Produces a TaskResult with a list of (col1, col2) tuples for identical columns.
        """
        try:

            # ctx = self.context
            df = self.input_data
            if is_polars(df):
                self._log(
                    (
                        "Falling back to Pandas: duplicate column"
                        " detection requires `.equals()`"
                    ),
                    "debug",
                )
                df = df.to_pandas()

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"Processing {len(matched_col)} column(s)", "debug")

            duplicate_pairs: List[Tuple[str, str]] = []
            columns = df.columns.tolist()
            seen = set()

            for i, col1 in enumerate(columns):
                for j in range(i + 1, len(columns)):
                    col2 = columns[j]
                    if (col1, col2) not in seen:
                        try:
                            if df[col1].equals(df[col2]):
                                duplicate_pairs.append((col1, col2))
                                seen.add((col1, col2))
                        except Exception as e:
                            self._log(
                                f"[DetectDuplicateColumns] Comparison failed for "
                                f"{col1} and {col2}: {e}",
                                "debug",
                            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Found {len(duplicate_pairs)} duplicate column pair(s)."
                    )
                },
                data={"duplicate_column_pairs": duplicate_pairs},
                metadata={
                    "suggested_viz_type": "None",
                    "recommended_section": "Redundancy",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
            )

            if self.get_engine_param("enable_impact_scoring", True) and duplicate_pairs:
                col1, col2 = duplicate_pairs[0]
                result = self.output
                if result:
                    from dsbf.utils.reco_engine import get_recommendation_tip

                    tip = get_recommendation_tip(self.name, {"correlation_with": 1.0})
                    self.set_ml_signals(
                        result=result,
                        score=0.85,
                        tags=["drop"],
                        recommendation=tip
                        or (
                            f"Column '{col2}' is a duplicate of '{col1}'. "
                            "Drop one to reduce redundancy and avoid overfitting."
                        ),
                    )
                    result.summary["column"] = col2

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
