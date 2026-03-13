# dsbf/eda/tasks/compute_entropy.py

from math import log2

import polars as pl
from scipy.stats import entropy as scipy_entropy

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import (
    TaskResult,
    add_reliability_warning,
    make_failure_result,
)
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Compute Entropy",
    description="Estimates entropy of columns to measure information content.",
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    domain="core",
    runtime_estimate="moderate",
    tags=["info", "distribution"],
    expected_semantic_types=["categorical", "text"],
)
class ComputeEntropy(BaseTask):
    """
    Computes the entropy of string-based columns to quantify categorical disorder.
    - Uses custom log2-based formula for Polars.
    - Uses scipy.stats.entropy for Pandas.
    """

    def run(self) -> None:
        results: dict[str, float] = {}

        # Use semantic typing to select relevant columns
        matched_cols, excluded = self.get_columns_by_intent()
        self._log(
            f"    Processing {len(matched_cols)} ['categorical', 'text'] column(s)",
            "debug",
        )

        try:
            df = self.input_data
            flags = self.ensure_reliability_flags()

            if is_polars(df):
                for col in matched_cols:
                    if df[col].dtype != pl.Utf8:
                        continue
                    try:
                        counts_df = df[col].value_counts()
                        counts = counts_df["count"]
                        total = counts.sum()
                        if total == 0:
                            continue  # Skip all-null or empty frequency
                        probs = [count / total for count in counts]
                        entropy_val = -sum(p * log2(p) for p in probs if p > 0)
                        results[col] = entropy_val
                    except Exception as e:
                        self._log(
                            f"    [ComputeEntropy] Failed on column {col}: {e}", "debug"
                        )
            else:
                for col in matched_cols:  # Only process matched columns
                    try:
                        counts = df[col].dropna().value_counts()
                        if counts.sum() == 0:
                            continue  # Skip empty frequency
                        results[col] = float(scipy_entropy(counts, base=2))
                    except Exception as e:
                        self._log(
                            f"    [ComputeEntropy] Failed on column {col}: {e}", "debug"
                        )

            result = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Computed entropy for {len(results)} columns."},
                data=results,
                plots={},
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Distributions",
                    "display_priority": "medium",
                    "excluded_columns": excluded,  # Now populated correctly
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys())
                    ),
                },
            )

            if flags["low_row_count"]:
                add_reliability_warning(
                    result,
                    level="heuristic_caution",
                    code="low_row_count_entropy",
                    description=(
                        "Entropy estimates may be unstable"
                        " with small sample sizes (N < 30)."
                    ),
                    recommendation=(
                        "Interpret entropy values cautiously"
                        " or validate with resampling."
                    ),
                )

            self.output = result

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
