# dsbf/eda/tasks/detect_duplicate_columns.py
from typing import TYPE_CHECKING

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.reco_engine import get_recommendation_tip

if TYPE_CHECKING:
    from pandas import DataFrame


@register_task(
    display_name="Detect Duplicate Columns",
    description="Finds columns that contain identical values.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    phase="eda",
    tags=["redundancy", "duplicates"],
    expected_semantic_types=["any"],
)
class DetectDuplicateColumns(BaseTask):
    """
    Detects columns that are exact duplicates of one another.

    Compares all unique column pairs using pandas ``Series.equals()``, which
    handles null values correctly (two nulls in the same position count as equal).

    Polars DataFrames are converted to pandas before comparison since Polars
    does not have a direct ``equals()`` equivalent across Series.

    Output is consumed by the Quality tab (Redundancy dimension). Guidance blurbs
    are emitted for every column that participates in a duplicate pair.
    """

    def run(self) -> None:  # noqa: C901
        """
        Execute duplicate column detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df: DataFrame = self.get_dataframe_pandas()

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} column(s)",
                "debug",
            )

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No columns found — duplicate column detection skipped.",
                    excluded,
                )
                return

            duplicate_pairs: list[tuple[str, str]] = []
            columns: list[str] = df.columns.tolist()
            seen: set[tuple[str, str]] = set()

            for i, col1 in enumerate(columns):
                for j in range(i + 1, len(columns)):
                    col2: str = columns[j]
                    if (col1, col2) not in seen:
                        try:
                            if df[col1].equals(df[col2]):
                                duplicate_pairs.append((col1, col2))
                                seen.add((col1, col2))
                        except Exception as e:  # noqa: BLE001
                            self._log(
                                f"    [{self.name}] Comparison failed for "
                                f"'{col1}' and '{col2}': {e}",
                                "debug",
                            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Found {len(duplicate_pairs)} duplicate column pair(s)."
                    ),
                },
                data={"duplicate_column_pairs": duplicate_pairs},
                metadata={
                    "suggested_viz_type": "none",
                    "recommended_section": "Redundancy",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col1, col2 in duplicate_pairs:
                self._attach_guidance(col1, col2)

            # ML impact scoring
            if self.get_engine_param("enable_impact_scoring", True) and duplicate_pairs:
                top_col1, top_col2 = duplicate_pairs[0]
                tip: str | None = get_recommendation_tip(
                    self.name,
                    {"correlation_with": 1.0},
                )
                self.set_ml_signals(
                    result=self.output,
                    score=0.85,
                    tags=["drop"],
                    recommendation=tip
                    or (
                        f"Column '{top_col2}' is a duplicate of '{top_col1}'. "
                        "Drop one to reduce redundancy and avoid overfitting."
                    ),
                )
                self.output.summary["column"] = top_col2

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, other_col: str) -> None:
        """
        Generate EDA and ML guidance for a confirmed duplicate column pair.

        Args:
            col: One column in the duplicate pair.
            other_col: The column it is identical to.

        """
        eda_body: str = (
            f"'{col}' is an exact duplicate of '{other_col}' - every value, "
            f"including nulls, is identical across all rows. This most commonly "
            f"occurs when a column is copied under a different name during a join "
            f"or transformation step, or when two features were derived from the "
            f"same source with no further transformation. Verify the data pipeline "
            f"to confirm which of the two is authoritative."
        )

        ml_body: str = (
            f"'{col}' is an exact duplicate of '{other_col}'. Including both in a "
            f"model adds zero information but increases dimensionality, slows "
            f"training, and inflates feature importance across the identical pair. "
            f"In linear models, the pair causes perfect multicollinearity. "
            f"Drop one of the two before modeling."
        )

        metric: dict[str, str] = {"duplicate_of": other_col}

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="warn",
            title=f"Exact Duplicate of '{other_col}'",
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )
        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level="warn",
            title="Duplicate Column - Drop Before Modeling",
            body=ml_body.strip(),
            actions=[
                {
                    "action": "drop",
                    "column": col,
                    "detail": (
                        f"Keep '{other_col}' or '{col}', drop the other - "
                        "identical columns provide no additional signal"
                    ),
                },
            ],
            metric=metric,
        )
