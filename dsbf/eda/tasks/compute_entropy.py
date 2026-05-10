# dsbf/eda/tasks/compute_entropy.py

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
    description=(
        "Estimates Shannon entropy of categorical columns to measure "
        "information content and distributional uniformity."
    ),
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    domain="core",
    runtime_estimate="moderate",
    phase="eda",
    tags=["info", "distribution", "categorical"],
    expected_semantic_types=["categorical", "text"],
)
class ComputeEntropy(BaseTask):
    """
    Computes Shannon entropy (base 2) for all categorical and text columns.

    Entropy quantifies distributional disorder: a column where every value is
    the same has entropy 0; a column where all values are equally likely has
    maximum entropy (log2 of cardinality). High entropy indicates near-uniform
    distribution; low entropy indicates dominance by one or few values.

    Both Polars and Pandas paths use ``scipy.stats.entropy`` with base 2 for
    consistent results. The Polars path extracts value counts into a numpy
    array before calling scipy, avoiding a full conversion to pandas.

    A reliability warning is emitted when N < 30, since entropy estimates
    are unstable on small samples (rare categories may be underrepresented).

    Output is consumed by the frontend Distributions tab entropy bar chart.
    """

    def run(self) -> None:  # noqa: C901
        """
        Execute entropy computation and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        results: dict[str, float] = {}

        try:
            df = self.get_dataframe()
            flags: dict = self.ensure_reliability_flags()

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} ['categorical', 'text'] column(s)",
                "debug",
            )

            if not matched_cols:
                self.output = self.make_empty_result(
                    (
                        "No categorical or text columns found — entropy computation"
                        " skipped."
                    ),
                    excluded,
                )
                return

            if is_polars(df):
                for col in matched_cols:
                    # Only process string columns; numeric columns classified as
                    # categorical (e.g. TURNOVERS) are handled by the pandas path
                    # via select_dtypes, not here.
                    if df[col].dtype not in (pl.String, pl.Utf8):
                        self._log(
                            f"    [{self.name}] Skipping non-string column '{col}' "
                            f"(dtype: {df[col].dtype})",
                            "debug",
                        )
                        continue
                    try:
                        counts_df = df[col].drop_nulls().value_counts()
                        # Extract the count array as numpy for scipy - avoids a
                        # full DataFrame-to-pandas conversion for a single column.
                        counts_array = counts_df["count"].to_numpy()
                        if counts_array.sum() == 0:
                            continue
                        results[col] = float(scipy_entropy(counts_array, base=2))
                    except Exception as e:  # noqa: BLE001
                        self._log(
                            f"    [{self.name}] Failed on column '{col}': {e}",
                            "debug",
                        )
            else:
                for col in matched_cols:
                    try:
                        counts = df[col].dropna().value_counts()
                        if counts.sum() == 0:
                            continue
                        results[col] = float(scipy_entropy(counts, base=2))
                    except Exception as e:  # noqa: BLE001
                        self._log(
                            f"    [{self.name}] Failed on column '{col}': {e}",
                            "debug",
                        )

            result = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Computed entropy for {len(results)} columns."},
                data=results,
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Distributions",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            if flags["low_row_count"]:
                add_reliability_warning(
                    result,
                    level="heuristic_caution",
                    code="low_row_count_entropy",
                    description=(
                        "Entropy estimates may be unstable with small sample "
                        "sizes (N < 30). Rare categories may be underrepresented, "
                        "causing entropy to be underestimated."
                    ),
                    recommendation=(
                        "Interpret entropy values cautiously or validate with "
                        "resampling."
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
