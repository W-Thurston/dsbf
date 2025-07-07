# dsbf/eda/tasks/detect_single_dominant_value.py

from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    display_name="Detect Single Dominant Value",
    description="Detects columns dominated by a single value.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    tags=["redundancy", "skew"],
)
class DetectSingleDominantValue(BaseTask):
    """
    Detects columns where a single value dominates the distribution,
    such as binary features with a heavy skew or categorical columns
    where nearly all values are the same.
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

            dominance_threshold = float(
                self.get_task_param("dominance_threshold") or 0.95
            )

            if is_polars(df):
                df = df.to_pandas()

            results: Dict[str, Dict[str, Any]] = {}

            for col in df.columns:
                series = df[col].dropna()
                if series.empty:
                    continue  # Skip all-null columns

                top_val = series.value_counts().index[0]
                proportion = series.value_counts(normalize=True).iloc[0]

                if proportion >= dominance_threshold:
                    self._log(
                        f"{col} has dominant value {top_val} at {proportion:.1%}",
                        "debug",
                    )
                    results[col] = {
                        "dominant_value": top_val,
                        "proportion": float(proportion),
                    }

            # Plotting
            plots: dict[str, dict[str, Any]] = {}

            for col in results:
                series = df[col].dropna()
                counts = series.value_counts()
                counts.name = col

                dominant_val = results[col]["dominant_value"]
                dominance = results[col]["proportion"]
                annotation = [f"Dominant value: '{dominant_val}' ({dominance:.1%})"]

                save_path = self.get_output_path(f"{col}_dominant_barplot.png")
                static = PlotFactory.plot_barplot_static(counts, save_path)
                interactive = PlotFactory.plot_barplot_interactive(counts)
                interactive["annotations"] = annotation

                plots[col] = {
                    "static": static["path"],
                    "interactive": interactive,
                }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Detected {len(results)} column(s) with dominant values."
                    )
                },
                data=results,
                plots=plots,
                metadata={"dominance_threshold": dominance_threshold},
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
