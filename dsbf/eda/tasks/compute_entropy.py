# dsbf/eda/tasks/compute_entropy.py

from math import log2
from typing import Dict

import polars as pl
from scipy.stats import entropy as scipy_entropy

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


class ComputeEntropy(BaseTask):
    """
    Computes the entropy of string-based columns to quantify categorical disorder.
    - Uses custom log2-based formula for Polars.
    - Uses scipy.stats.entropy for Pandas.
    """

    def run(self) -> None:
        """
        Executes entropy computation on text-like columns.
        Produces a TaskResult with column-wise entropy scores.
        """
        results: Dict[str, float] = {}

        try:
            df = self.input_data

            if is_polars(df):
                # Polars backend: compute entropy manually
                for col in df.columns:
                    if df[col].dtype == pl.Utf8:
                        try:
                            counts_df = df[col].value_counts()
                            counts = counts_df["counts"]
                            total = counts.sum()
                            probs = [count / total for count in counts]
                            entropy_val = -sum(p * log2(p) for p in probs if p > 0)
                            results[col] = entropy_val
                        except Exception as e:
                            print(f"[ComputeEntropy] Failed on column {col}: {e}")
            else:
                # Pandas fallback: use scipy entropy
                for col in df.select_dtypes(include="object").columns:
                    try:
                        counts = df[col].value_counts()
                        results[col] = float(scipy_entropy(counts, base=2))
                    except Exception as e:
                        print(f"[ComputeEntropy] Failed on column {col}: {e}")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Computed entropy for {len(results)} columns.",
                data=results,
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during entropy computation: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
