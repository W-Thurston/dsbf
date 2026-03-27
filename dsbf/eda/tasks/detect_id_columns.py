# dsbf/eda/tasks/detect_id_columns.py

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
    phase="eda",
    tags=["metadata", "id", "index"],
    expected_semantic_types=["id", "categorical", "text"],
)
class DetectIdColumns(BaseTask):
    """
    Detects columns that are likely unique row identifiers.

    A column is flagged when its unique value count exceeds a configurable
    proportion of total rows (default: 95%). This catches user IDs, UUIDs,
    order numbers, and other opaque keys that carry no analytical signal and
    should be excluded from model features.

    Near-unique numeric columns are excluded - high cardinality in a continuous
    column is expected and is not evidence of an identifier. The data_quality_scorer
    applies this filter when consuming this task's output.

    Supports both Polars and Pandas DataFrames.

    Configurable parameters (via config["tasks"]["detect_id_columns"]):
        threshold_ratio (float): Proportion of rows that must be unique for a
            column to be flagged. Default: 0.95
    """

    def run(self) -> None:  # noqa: C901
        """
        Execute ID column detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} "
                "['id', 'categorical', 'text'] column(s)",
                "debug",
            )

            threshold_ratio = float(self.get_task_param("threshold_ratio") or 0.95)
            n_rows = df.shape[0]
            threshold = threshold_ratio * n_rows

            results: dict[str, str] = {}

            if is_polars(df):
                for col in matched_cols:
                    try:
                        n_unique = df[col].n_unique()
                        if n_unique >= threshold:
                            results[col] = f"{n_unique} unique values (likely ID)"
                            self._log(
                                f"    '{col}' flagged as likely ID "
                                f"({n_unique} unique values)",
                                "debug",
                            )
                    except Exception:  # noqa: BLE001, PERF203, S112
                        continue
            else:
                for col in matched_cols:
                    try:
                        n_unique = df[col].nunique()
                        if n_unique >= threshold:
                            results[col] = f"{n_unique} unique values (likely ID)"
                            self._log(
                                f"    '{col}' flagged as likely ID "
                                f"({n_unique} unique values)",
                                "debug",
                            )
                    except Exception:  # noqa: BLE001, PERF203, S112
                        continue

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Detected {len(results)} likely ID column(s)."},
                data=results,
                metadata={
                    "rows": n_rows,
                    "threshold_ratio": threshold_ratio,
                    "suggested_viz_type": "none",
                    "recommended_section": "Schema",
                    "display_priority": "low",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col in results:
                self._attach_guidance(col)

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
        Generate EDA guidance for a confirmed likely-ID column.

        Args:
            col: Column name flagged as a likely identifier.

        """
        eda_body: str = (
            f"'{col}' has near-unique values across all rows, indicating it is "
            f"likely a row identifier (e.g. a user ID, order number, or UUID) "
            f"rather than an analytical feature. ID columns carry no signal "
            f"about the phenomenon being studied - they simply label each row. "
            f"Confirm this is intentional and exclude it from feature selection "
            f"before modeling."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="info",
            title="Likely Identifier Column - High Uniqueness",
            body=eda_body.strip(),
            actions=[],
            metric={"flagged_as": "id"},
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level="warn",
            title="ID Column - Exclude from Model Features",
            body=(
                f"'{col}' is a likely identifier. Including it as a model feature "
                f"causes memorisation: the model learns to associate each row's "
                f"identity with its target value, achieving perfect training "
                f"accuracy but zero generalisation. Drop this column before "
                f"training."
            ),
            actions=[
                {
                    "action": "drop",
                    "column": col,
                    "detail": "Identifier columns cause target leakage - "
                    "exclude from all feature matrices",
                },
            ],
            metric={"flagged_as": "id"},
        )
