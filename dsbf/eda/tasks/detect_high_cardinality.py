# dsbf/eda/tasks/detect_high_cardinality.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    display_name="Detect High Cardinality",
    description="Detects categorical columns with too many unique values.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    phase="eda",
    tags=["categorical", "cardinality"],
    expected_semantic_types=["categorical"],
)
class DetectHighCardinality(BaseTask):
    """
    Detects categorical columns whose unique value count exceeds a threshold.

    A high-cardinality categorical column has so many distinct values that
    standard one-hot encoding becomes impractical - it inflates dimensionality,
    creates sparse features, and degrades model performance. Common examples
    include city names, product SKUs, and user IDs stored as categoricals.

    Supports both Polars and Pandas DataFrames.

    Output is consumed by the Quality tab (Usability dimension) and ML Readiness
    tab (Encoding Required dimension).

    Configurable parameters (via config["tasks"]["detect_high_cardinality"]):
        cardinality_threshold (float): Unique value count above which a column
            is flagged. Default: 50
    """

    def run(self) -> None:  # noqa: C901
        """
        Execute high-cardinality detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run_native("'categorical'")

            if not matched_cols:
                self.output = self.make_empty_result(
                    (
                        "No categorical columns found — high cardinality detection"
                        " skipped."
                    ),
                    excluded,
                )
                return

            cardinality_threshold = float(
                self.get_task_param("cardinality_threshold") or 50,
            )

            results: dict[str, int] = {}

            if is_polars(df):
                for col in matched_cols:
                    try:
                        n_unique = df[col].n_unique()
                        if n_unique > cardinality_threshold:
                            results[col] = n_unique
                            self._log(
                                f"    '{col}' has {n_unique} unique values",
                                "debug",
                            )
                    except Exception:  # noqa: BLE001, PERF203, S112
                        continue
            else:
                for col in matched_cols:
                    try:
                        n_unique = df[col].nunique()
                        if n_unique > cardinality_threshold:
                            results[col] = n_unique
                            self._log(
                                f"    '{col}' has {n_unique} unique values",
                                "debug",
                            )
                    except Exception:  # noqa: BLE001, PERF203, S112
                        continue

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (f"Detected {len(results)} high-cardinality column(s)."),
                },
                data=results,
                metadata={
                    "cardinality_threshold": cardinality_threshold,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Cardinality",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, n_unique in results.items():
                self._attach_guidance(col, n_unique, cardinality_threshold)

            # ML impact scoring
            if self.get_engine_param("enable_impact_scoring", True) and results:
                top_col: str = next(iter(results))
                top_n: int = results[top_col]
                tip: str | None = get_recommendation_tip(self.name, {"n_unique": top_n})
                self.set_ml_signals(
                    result=self.output,
                    score=0.7,
                    tags=["transform", "monitor"],
                    recommendation=tip
                    or (
                        f"Column '{top_col}' has high cardinality "
                        f"({top_n} unique values). Consider frequency encoding, "
                        "bucketing, or dimensionality reduction."
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

    def _attach_guidance(self, col: str, n_unique: int, threshold: float) -> None:
        """
        Generate EDA and ML guidance for a high-cardinality column.

        Args:
            col: Column name.
            n_unique: Number of unique values detected.
            threshold: Configured cardinality threshold.

        """
        eda_body: str = (
            f"'{col}' has {n_unique} unique values, exceeding the high-cardinality "
            f"threshold of {int(threshold)}. High-cardinality categoricals are "
            f"difficult to summarise in a frequency table - the long tail of rare "
            f"values may contain meaningful patterns or may be noise. Check the "
            f"value count distribution and decide whether to keep all levels, "
            f"group rare values into an 'Other' bucket, or treat the column as "
            f"an identifier."
        )

        ml_body: str = (
            f"'{col}' has {n_unique} unique values. One-hot encoding will create "
            f"{n_unique} sparse binary features, inflating dimensionality and "
            f"degrading tree-based model performance through split fragmentation. "
            f"Preferred alternatives: frequency encoding (replace each category "
            f"with its occurrence count), target encoding (replace with mean target "
            f"value - apply only on training fold to prevent leakage), or hashing "
            f"trick for very high cardinality. For linear models, target encoding "
            f"or embeddings are typically most effective."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="warn",
            title=f"High Cardinality ({n_unique} unique values)",
            body=eda_body.strip(),
            actions=[],
            metric={"n_unique": n_unique, "threshold": threshold},
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level="warn",
            title=f"High Cardinality - Avoid One-Hot Encoding ({n_unique} levels)",
            body=ml_body.strip(),
            actions=[
                {
                    "action": "encode",
                    "method": "frequency_encoding",
                    "column": col,
                    "detail": "Replace each category with its row count - "
                    "simple and leakage-free",
                },
                {
                    "action": "encode",
                    "method": "target_encoding",
                    "column": col,
                    "detail": "Replace with mean target value - apply on "
                    "training fold only to prevent leakage",
                },
                {
                    "action": "group",
                    "method": "bucket_rare_values",
                    "column": col,
                    "detail": "Collapse low-frequency values into an 'Other' "
                    "bucket before encoding",
                },
            ],
            metric={"n_unique": n_unique, "threshold": threshold},
        )
