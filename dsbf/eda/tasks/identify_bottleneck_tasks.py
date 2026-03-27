# dsbf/eda/tasks/identify_bottleneck_tasks.py

from typing import cast

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult


@register_task(
    display_name="Identify Bottleneck Tasks",
    description="Ranks the top-N slowest tasks by runtime duration.",
    profiling_depth="full",
    stage="any",
    phase="diagnostic",
    domain="core",
    runtime_estimate="fast",
    tags=["diagnostic", "runtime", "performance"],
    expected_semantic_types=["any"],
)
class IdentifyBottleneckTasks(BaseTask):
    """
    Rank the top-N slowest tasks by runtime duration.

    Reads per-task duration metadata from the analysis context and returns the
    slowest N tasks sorted by descending duration. Emits a recommendation for
    any task exceeding 5 seconds.

    Returns a ``"failed"`` result rather than raising if no duration metadata
    is available - this allows the task to be included in the DAG without
    breaking the run when timing data has not been populated.

    Configurable parameters (via config["tasks"]["identify_bottleneck_tasks"]):
        top_n (int): Number of slowest tasks to report. Default: 5
    """

    def run(self) -> None:
        """
        Identify slowest tasks and populate self.output.

        Raises:
            RuntimeError: If no analysis context is attached.

        """
        matched_cols, excluded = self.get_columns_by_intent()
        self._log(f"    Processing {len(matched_cols)} column(s)", "debug")

        if self.context is None:
            raise RuntimeError("Context is not set for task.")

        durations: dict[str, float] = cast(
            "dict[str, float]",
            self.context.get_metadata("task_durations", {}),
        )
        if not durations:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary={"message": "No task durations available to analyze."},
            )
            return

        top_n = int(self.get_task_param("top_n") or 5)
        sorted_tasks: list[tuple[str, float]] = sorted(
            durations.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        bottlenecks: list[dict[str, float | str]] = [
            {"task": name, "duration_sec": round(duration, 4)}
            for name, duration in sorted_tasks
        ]

        recommendations: list[str] = [
            f"Consider optimizing or parallelizing '{t['task']}' "
            f"(took {t['duration_sec']}s)."
            for t in bottlenecks
            if t["duration_sec"] > 5.0  # noqa: PLR2004
        ]

        self.output = TaskResult(
            name=self.name,
            status="success",
            summary={
                "top_bottlenecks": bottlenecks,
                "message": f"Top {top_n} slowest tasks identified.",
            },
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
