# dsbf/eda/tasks/identify_bottleneck_tasks.py

from typing import Any, Dict, cast

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    display_name="Identify Bottleneck Tasks",
    description="Ranks the top-N slowest tasks by runtime duration.",
    profiling_depth="full",
    stage="any",
    domain="core",
    runtime_estimate="fast",
    tags=["diagnostic", "runtime", "performance"],
)
class IdentifyBottleneckTasks(BaseTask):
    """
    Analyze task durations and identify top-N slowest bottlenecks.
    """

    def run(self) -> None:

        if self.context is None:
            raise RuntimeError("Context is not set for task.")

        durations_raw = self.context.get_metadata("task_durations", {})
        durations: Dict[str, float] = cast(Dict[str, float], durations_raw)
        if not durations:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary={"message": "No task durations available to analyze."},
            )
            return

        top_n = self.get_task_param("top_n", default=5)
        sorted_tasks = sorted(durations.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]

        summary = {
            "top_bottlenecks": [
                {"task": name, "duration_sec": round(duration, 4)}
                for name, duration in sorted_tasks
            ],
            "message": f"Top {top_n} slowest tasks identified.",
        }

        recommendations = []
        for task_info in summary["top_bottlenecks"]:
            if task_info["duration_sec"] > 5.0:
                recommendations.append(
                    f"Consider optimizing or parallelizing '{task_info['task']}'"
                    f" (took {task_info['duration_sec']}s)."
                )

        plots: dict[str, dict[str, Any]] = {}

        try:
            if sorted_tasks:
                durations_df = pd.DataFrame(
                    sorted_tasks, columns=["Task", "Duration (s)"]
                )

                # Reverse for horizontal barplot (longest task on top)
                durations_df = durations_df.iloc[::-1]

                series = pd.Series(
                    durations_df["Duration (s)"].values,
                    index=durations_df["Task"].values,
                    name="Duration (s)",
                )

                save_path = self.get_output_path("bottleneck_task_durations.png")
                static = PlotFactory.plot_barplot_static(
                    series, save_path=save_path, title="Top Bottleneck Tasks by Runtime"
                )
                interactive = PlotFactory.plot_barplot_interactive(
                    series,
                    title="Top Bottleneck Tasks by Runtime",
                    annotations=[
                        f"{row['Task']}: {row['Duration (s)']:.2f}s"
                        for _, row in durations_df.iterrows()
                    ],
                )

                plots["bottleneck_tasks"] = {
                    "static": static["path"],
                    "interactive": interactive,
                }
        except Exception as e:
            self._log(f"[PlotFactory] Skipped bottleneck barplot: {e}", level="debug")

        self.output = TaskResult(
            name=self.name,
            status="success",
            summary=summary,
            recommendations=recommendations,
            plots=plots,
        )
