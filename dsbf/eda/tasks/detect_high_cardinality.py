# dsbf/eda/tasks/detect_high_cardinality.py

from typing import Any, Literal

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

# from dsbf.utils.plot_factory import PlotFactory
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    display_name="Detect High Cardinality",
    description="Detects columns with too many unique values.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    tags=["categorical", "cardinality"],
    expected_semantic_types=["categorical"],
)
class DetectHighCardinality(BaseTask):
    """Detects columns with a number of unique values greater than a threshold."""

    def run(self) -> None:
        """
        Execute the high-cardinality detection task and store the results in

            `self.output`.
        """
        try:
            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_col)} 'categorical' column(s)",
                "debug",
            )

            cardinality_threshold = float(
                self.get_task_param("cardinality_threshold") or 50,
            )

            n_rows = df.shape[0]
            results: dict[str, int] = {}

            if is_polars(df):
                for col in df.columns:
                    try:
                        n_unique = df[col].n_unique()
                        if n_unique > cardinality_threshold:
                            results[col] = n_unique
                            self._log(
                                f"    {col} has {n_unique} unique values", "debug"
                            )
                    except Exception:
                        continue
            else:
                for col in df.columns:
                    try:
                        n_unique = df[col].nunique()
                        if n_unique > cardinality_threshold:
                            results[col] = n_unique
                            self._log(
                                f"    {col} has {n_unique} unique values", "debug"
                            )
                    except Exception:
                        continue

            # Build TaskResult
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (f"Detected {len(results)} high-cardinality column(s)."),
                },
                data=results,
                plots={},
                metadata={
                    "cardinality_threshold": cardinality_threshold,
                    "n_rows": n_rows,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Cardinality",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys()),
                    ),
                },
            )

            # Generate per-column guidance for all flagged columns
            for col, n_unique in results.items():
                self._attach_guidance(col, n_unique, n_rows)

            # Apply ML scoring to self.output
            if self.get_engine_param("enable_impact_scoring", True) and results:
                col: str = next(iter(results))  # First offending column
                n_unique: int = results[col]
                result: TaskResult = self.output
                if result:
                    tip: str | None = get_recommendation_tip(
                        self.name, {"n_unique": n_unique}
                    )
                    self.set_ml_signals(
                        result=result,
                        score=0.7,
                        tags=["transform", "monitor"],
                        recommendation=tip
                        or (
                            f"Column '{col}' has high cardinality "
                            f"({n_unique} unique values). "
                            "Consider frequency encoding, bucketing, or"
                            " dimensionality reduction."
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

    def _attach_guidance(self, col: str, n_unique: int, n_rows: int) -> None:
        """Generate EDA + ML guidance for a high-cardinality categorical column."""
        ratio: Literal[0] | float = n_unique / n_rows if n_rows else 0
        ratio_str: str = f"{ratio:.1%}"

        # Near-ID: >50% unique (high enough that it's probably an identifier
        # rather than a genuine categorical feature)
        is_near_id: bool = ratio > 0.5

        if is_near_id:
            level = "warn"
            eda_body: str = (
                f"{col} has {n_unique:,} unique values across {n_rows:,} rows "
                f"({ratio_str} uniqueness). At this level the column may be a "
                f"near-identifier — each value appears very rarely, making it "
                f"difficult to observe patterns across groups. Confirm whether "
                f"this is a meaningful categorical feature or effectively a key "
                f"column masquerading as a category."
            )
            ml_body: str = (
                f"{col} has {n_unique:,} unique values ({ratio_str} of rows). "
                f"One-hot encoding would produce {n_unique:,} sparse features, "
                f"causing extreme dimensionality and likely overfitting. "
                f"Target encoding or hashing are the practical options — "
                f"but verify this column is a genuine feature first, not an ID."
            )
            ml_actions: list[dict[str, str]] = [
                {
                    "action": "encode",
                    "method": "target_encoding",
                    "column": col,
                    "condition": "supervised context",
                },
                {
                    "action": "encode",
                    "method": "hash_encoding",
                    "column": col,
                    "condition": "unsupervised or high memory constraint",
                },
                {
                    "action": "drop",
                    "column": col,
                    "detail": "If confirmed to be an identifier",
                },
            ]
        else:
            level = "info"
            eda_body = (
                f"{col} has {n_unique:,} unique values across {n_rows:,} rows. "
                f"High cardinality means individual category frequencies are low — "
                f"bar charts will be crowded and group comparisons will be noisy. "
                f"Consider whether some categories can be grouped meaningfully, "
                f"or whether the column represents a genuinely fine-grained taxonomy."
            )
            ml_body = (
                f"{col} has {n_unique:,} unique values. One-hot encoding will "
                f"produce {n_unique:,} binary features, which is likely too many "
                f"for most models. Target encoding or frequency encoding are "
                f"more efficient. Group rare categories into an 'Other' bucket "
                f"before encoding to reduce noise."
            )
            ml_actions = [
                {
                    "action": "encode",
                    "method": "target_encoding",
                    "column": col,
                    "condition": "supervised context",
                },
                {
                    "action": "encode",
                    "method": "frequency_encoding",
                    "column": col,
                    "condition": "any",
                },
                {
                    "action": "group_rare",
                    "column": col,
                    "detail": "Collapse low-frequency categories into 'Other'",
                },
            ]

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=f"High Cardinality ({n_unique:,} unique values)",
            body=eda_body.strip(),
            actions=[],
            metric={
                "n_unique": n_unique,
                "n_rows": n_rows,
                "uniqueness_ratio": round(ratio, 4),
            },
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=level,
            title=f"High Cardinality — Encoding Required ({n_unique:,} values)",
            body=ml_body.strip(),
            actions=ml_actions,
            metric={
                "n_unique": n_unique,
                "n_rows": n_rows,
                "uniqueness_ratio": round(ratio, 4),
            },
        )
