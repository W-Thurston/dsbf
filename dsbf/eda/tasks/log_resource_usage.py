# dsbf/eda/tasks/log_resource_usage.py

from typing import cast

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult


@register_task(
    display_name="Log Resource Usage",
    description="Summarizes overall runtime and per-task execution totals.",
    profiling_depth="full",
    stage="any",
    phase="diagnostic",
    domain="core",
    runtime_estimate="fast",
    tags=["diagnostic", "runtime", "logging"],
    expected_semantic_types=["any"],
)
class LogResourceUsage(BaseTask):
    """
    Summarize runtime usage across all tasks in the current run.

    Reads per-task duration metadata from the analysis context and computes
    total and mean execution time. Emits a recommendation if the total runtime
    exceeds 30 seconds or if the mean per-task time exceeds 5 seconds.

    This task requires that task durations have been written to the context
    metadata before it runs (populated by the execution engine after each task).
    """

    def run(self) -> None:
        """
        Compute runtime statistics and populate self.output.

        Raises:
            RuntimeError: If no analysis context is attached.

        """
        matched_cols, excluded = self.get_columns_by_intent()
        self._log(f"    Processing {len(matched_cols)} column(s)", "debug")

        if self.context is None:
            raise RuntimeError("Context is not set.")

        durations: dict[str, float] = cast(
            "dict[str, float]",
            self.context.get_metadata("task_durations", {}),
        )
        run_stats: dict = self.context.get_metadata("run_stats") or {}

        total_time: float | int = (
            round(sum(durations.values()), 2)
            if durations
            else cast("float", run_stats.get("duration", 0.0))
        )

        task_count: int = len(durations)
        mean_task_time: float | None = (
            round(total_time / task_count, 4) if task_count else None
        )

        summary: dict[str, dict[str, float] | float | int | None] = {
            "task_count": task_count,
            "total_runtime_sec": total_time,
            "mean_task_time": mean_task_time,
            "task_durations": {
                k: round(v, 4) for k, v in sorted(durations.items(), key=lambda x: x[1])
            },
        }

        recommendations: list[str] = []
        if total_time > 30:  # noqa: PLR2004
            recommendations.append(
                "Consider caching static tasks if total time exceeds 30 seconds.",
            )
        if mean_task_time and mean_task_time > 5:  # noqa: PLR2004
            recommendations.append("Investigate tasks with long average runtime.")

        self.output = TaskResult(
            name=self.name,
            status="success",
            summary=summary,
            recommendations=recommendations,
            metadata={
                "suggested_viz_type": "bar",
                "recommended_section": "Diagnostics",
                "display_priority": "low",
                "excluded_columns": excluded,
                "column_types": self.get_column_type_info(
                    matched_cols + list(excluded.keys()),
                ),
            },
        )
