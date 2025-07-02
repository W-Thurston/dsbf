# dsbf/eda/tasks/compute_entropy.py

from math import log2
from typing import Dict

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
)
class ComputeEntropy(BaseTask):
    """
    Computes the entropy of string-based columns to quantify categorical disorder.
    - Uses custom log2-based formula for Polars.
    - Uses scipy.stats.entropy for Pandas.
    """

    def run(self) -> None:
        results: Dict[str, float] = {}

        try:
            df = self.input_data
            flags = self.ensure_reliability_flags()

            if is_polars(df):
                for col in df.columns:
                    if df[col].dtype == pl.Utf8:
                        try:
                            counts_df = df[col].value_counts()
                            counts = counts_df["count"]
                            total = counts.sum()
                            probs = [count / total for count in counts]
                            entropy_val = -sum(p * log2(p) for p in probs if p > 0)
                            results[col] = entropy_val
                        except Exception as e:
                            self._log(
                                f"[ComputeEntropy] Failed on column {col}: {e}", "debug"
                            )
            else:
                for col in df.select_dtypes(include="object").columns:
                    try:
                        counts = df[col].value_counts()
                        results[col] = float(scipy_entropy(counts, base=2))
                    except Exception as e:
                        self._log(
                            f"[ComputeEntropy] Failed on column {col}: {e}", "debug"
                        )

            result = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Computed entropy for {len(results)} columns."},
                data=results,
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
            self.output = make_failure_result(self.name, e)
