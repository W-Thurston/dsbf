# dsbf/eda/tasks/log_resource_usage.py

from typing import Any, Dict, cast

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    display_name="Log Resource Usage",
    description="Summarizes overall runtime and per-task execution totals.",
    profiling_depth="full",
    stage="any",
    domain="core",
    runtime_estimate="fast",
    tags=["diagnostic", "runtime", "logging"],
)
class LogResourceUsage(BaseTask):
    def run(self) -> None:
        if self.context is None:
            raise RuntimeError("Context is not set.")

        # Safely cast durations and run_stats
        durations = cast(
            Dict[str, float], self.context.get_metadata("task_durations", {})
        )
        run_stats = self.context.get_metadata("run_stats") or {}

        total_time = (
            round(sum(durations.values()), 2)
            if durations
            else cast(float, run_stats.get("duration", 0.0))
        )

        task_count = len(durations)
        mean_task_time = round(total_time / task_count, 4) if task_count else None

        summary = {
            "task_count": task_count,
            "total_runtime_sec": total_time,
            "mean_task_time": mean_task_time,
            "task_durations": {
                k: round(v, 4) for k, v in sorted(durations.items(), key=lambda x: x[1])
            },
        }

        recommendations = []
        if total_time > 30:
            recommendations.append(
                "Consider caching static tasks if total time exceeds 30 seconds."
            )
        if mean_task_time and mean_task_time > 5:
            recommendations.append("Investigate tasks with long average runtime.")

        plots: dict[str, dict[str, Any]] = {}

        try:
            if durations:
                series = pd.Series(durations, name="Duration (s)").sort_values()

                save_path = self.get_output_path("per_task_runtime.png")
                static = PlotFactory.plot_barplot_static(
                    series, save_path=save_path, title="Per-Task Runtime"
                )
                interactive = PlotFactory.plot_barplot_interactive(
                    series,
                    title="Per-Task Runtime",
                    annotations=[f"{k}: {v:.2f}s" for k, v in series.items()],
                )

                plots["task_runtime"] = {
                    "static": static["path"],
                    "interactive": interactive,
                }

        except Exception as e:
            self._log(
                f"[PlotFactory] Skipped resource usage barplot: {e}", level="debug"
            )

        self.output = TaskResult(
            name=self.name,
            status="success",
            summary=summary,
            recommendations=recommendations,
            plots=plots,
        )
