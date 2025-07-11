# dsbf/eda/tasks/detect_id_columns.py

from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect ID Columns",
    description="Flags likely ID-like columns with high uniqueness and low reuse.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    tags=["metadata", "id", "index"],
    expected_semantic_types=["id", "categorical", "text"],
)
class DetectIdColumns(BaseTask):
    """
    Detects columns likely to be unique identifiers (e.g., user IDs, UUIDs).
    A column is flagged if its number of unique values exceeds 95% of total rows.
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_col)} ['id', 'categorical', 'text']"
                " column(s)",
                "debug",
            )

            threshold_ratio = float(self.get_task_param("threshold_ratio") or 0.95)

            n_rows = df.shape[0]
            threshold = threshold_ratio * n_rows
            results: Dict[str, str] = {}

            if is_polars(df):
                for col in df.columns:
                    try:
                        n_unique = df[col].n_unique()
                        if n_unique >= threshold:
                            results[col] = f"{n_unique} unique values (likely ID)"
                            self._log(
                                (
                                    f"    {col} flagged as likely ID with {n_unique}"
                                    " unique values"
                                ),
                                "debug",
                            )
                    except Exception:
                        continue
            else:
                for col in df.columns:
                    try:
                        n_unique = df[col].nunique()
                        if n_unique >= threshold:
                            results[col] = f"{n_unique} unique values (likely ID)"
                            self._log(
                                (
                                    f"    {col} flagged as likely ID with {n_unique}"
                                    " unique values"
                                ),
                                "debug",
                            )
                    except Exception:
                        continue

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": (f"Detected {len(results)} likely ID column(s).")},
                data=results,
                metadata={
                    "rows": n_rows,
                    "threshold_ratio": threshold_ratio,
                    "suggested_viz_type": "None",
                    "recommended_section": "Schema",
                    "display_priority": "low",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
            )

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} — {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
