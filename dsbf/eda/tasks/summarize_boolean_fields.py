from typing import Any, Dict

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory


def is_boolean_column(series) -> bool:
    """Identify columns that only contain True/False or nulls."""
    non_null_vals = series.dropna().unique()
    return set(non_null_vals).issubset({True, False})


@register_task(
    display_name="Summarize Boolean Fields",
    description="Summarizes frequency and distribution of boolean columns.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    tags=["boolean", "summary"],
    expected_semantic_types=["boolean"],
)
class SummarizeBooleanFields(BaseTask):
    """
    Summarizes boolean columns by computing proportions of True, False, and
    missing values.
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"Processing {len(matched_col)} 'boolean' column(s)", "debug")

            if is_polars(df):
                df = df.to_pandas()

            bool_cols = [col for col in df.columns if is_boolean_column(df[col])]
            result: Dict[str, Dict[str, float]] = {}

            for col in bool_cols:
                total = len(df[col])
                true_count = (df[col] == True).sum()  # noqa: E712
                false_count = (df[col] == False).sum()  # noqa: E712
                null_count = df[col].isnull().sum()

                result[col] = {
                    "pct_true": true_count / total,
                    "pct_false": false_count / total,
                    "pct_null": null_count / total,
                }

            plots: dict[str, dict[str, Any]] = {}

            if self.context and self.context.output_dir:
                for col in bool_cols:
                    series = df[col]
                    counts = series.value_counts(dropna=False).to_dict()
                    count_series = pd.Series(counts)

                    # Compose annotation
                    total = len(series)
                    annotations = []
                    for k, v in counts.items():
                        label = "Null" if pd.isna(k) else str(k)
                        pct = v / total * 100
                        annotations.append(f"{label}: {pct:.1f}%")

                    # Static + interactive
                    save_path = self.get_output_path(f"{col}_boolean_barplot.png")
                    static = PlotFactory.plot_barplot_static(count_series, save_path)
                    interactive = PlotFactory.plot_barplot_interactive(
                        count_series, annotations=annotations
                    )

                    plots[col] = {
                        "static": static["path"],
                        "interactive": interactive,
                    }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": (f"Summarized {len(result)} boolean columns.")},
                data=result,
                plots=plots,
                metadata={
                    "bool_columns": bool_cols,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Summary",
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
            self.output = make_failure_result(self.name, e)
