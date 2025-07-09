# dsbf/eda/graph.py
"""
Execution Graph module for DSBF.

Defines `Task` and `ExecutionGraph` classes which manage DAG-style lazy execution of
EDA tasks.  Includes support for dependency resolution, error handling, and DAG
visualization.
"""

import os
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import networkx as nx
import psutil

from dsbf.core.base_task import BaseTask
from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.utils.dag_layout import assign_waterfall_positions, draw_dag, topo_sort_levels


class Task:
    def __init__(
        self, name: str, task_instance: BaseTask, requires: Optional[List[str]] = None
    ):
        self.name = name
        self.task_instance = task_instance
        self.requires = requires or []
        self.result: Optional[TaskResult] = None
        self.status = "pending"  # "success" or "failed"

    def run(self, context: AnalysisContext) -> TaskResult:
        try:
            result = context.run_task(self.task_instance)
            self.result = result
            self.status = "success"
        except Exception as e:
            self.status = "failed"
            raise RuntimeError(f"Task '{self.name}' failed: {e}") from e
        return self.result

    def __repr__(self):
        return (
            f"<Task name={self.name}, requires={self.requires}, status={self.status}>"
        )


class ExecutionGraph:
    def __init__(self, tasks: List[Task]):
        self.task_map = {task.name: task for task in tasks}
        self.graph = nx.DiGraph()

        # Build DAG structure
        for task in tasks:
            self.graph.add_node(task.name)
            for dep in task.requires:
                if dep not in self.task_map:
                    raise ValueError(
                        f"Task '{task.name}' depends on unknown task '{dep}'"
                    )
                self.graph.add_edge(dep, task.name)

        # Compute topological order
        _, self.node_levels = topo_sort_levels(self.graph)
        self.tasks_sorted = sorted(tasks, key=lambda t: self.node_levels[t.name])

    def run(
        self,
        context: AnalysisContext,
        log_fn: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, TaskResult]:
        task_outcomes = {
            "success": [],
            "failed": [],
            "skipped": [],
        }

        # Initialize task druations within AnalysisContext
        context.metadata.setdefault("task_durations", {})
        run_start = time.time()  # Global start time

        # Initialize memory tracker
        process = psutil.Process(os.getpid())
        context.metadata.setdefault("task_memory", {})
        limits = context.get_config("resource_limits") or {}
        max_memory = limits.get("max_memory_gb")
        max_runtime = limits.get("max_runtime_seconds")

        for task in self.tasks_sorted:
            deps = task.requires
            if deps:
                dep_statuses = [f"{dep}: {self.task_map[dep].status}" for dep in deps]
                if log_fn:
                    log_fn(
                        (
                            f"[{task.name}] Starting task (depends on: "
                            f"{', '.join(dep_statuses)})"
                        ),
                        "info",
                    )
            else:
                if log_fn:
                    log_fn(f"[{task.name}] Starting task (no dependencies)", "info")

            failed_deps = [
                dep for dep in deps if self.task_map[dep].status != "success"
            ]
            if failed_deps:
                task.status = "skipped"
                task_outcomes["skipped"].append(task.name)
                if log_fn:
                    log_fn(
                        f"[{task.name}] Skipped due to failed dependency: "
                        f"{failed_deps}",
                        "info",
                    )
                continue

            # Try running the task
            start_time = time.time()
            try:
                mem_before = process.memory_info().rss / 1e6  # in MB
                _ = task.run(context)

                mem_after = process.memory_info().rss / 1e6
                peak_mem = max(mem_before, mem_after)
                context.metadata["task_memory"][task.name] = peak_mem
                if max_memory and peak_mem > max_memory * 1024:
                    context._log(
                        (
                            f"[WARNING] Task '{task.name}' exceeded memory limit "
                            f"({peak_mem:.1f} MB > {max_memory} GB)"
                        ),
                        level="info",
                    )
                    if task.result:
                        task.result.metadata["memory_exceeded"] = True

                # Collect and log task duration
                duration = time.time() - start_time
                context.metadata["task_durations"][task.name] = duration

                if max_runtime and duration > max_runtime:
                    context._log(
                        (
                            f"[WARNING] Task '{task.name}' exceeded runtime limit "
                            f"({duration:.2f}s > {max_runtime}s)"
                        ),
                        level="info",
                    )
                    if task.result:
                        task.result.metadata["runtime_exceeded"] = (
                            duration > max_runtime
                        )

                if log_fn:
                    log_fn(
                        f"[{task.name}] Completed in {duration:.2f}s"
                        f" (status: {task.status})",
                        "info",
                    )
                task_outcomes["success"].append(task.name)

            except Exception as e:

                # Collect and log task duration
                duration = time.time() - start_time
                context.metadata["task_durations"][task.name] = duration

                error_metadata = {
                    "error_type": type(e).__name__,
                    "trace_summary": str(e),
                    "suggested_action": "Check logs or upstream outputs",
                }

                failed_result = TaskResult(
                    name=task.name,
                    status="failed",
                    summary={"message": "Task failed due to exception."},
                    error_metadata=error_metadata,
                )

                task.result = failed_result
                context.set_result(task.name, failed_result)

                if log_fn:
                    log_fn(
                        f"[{task.name}] Failed after {duration:.2f}s: "
                        f"{error_metadata['trace_summary']}",
                        "warn",
                    )

                task_outcomes["failed"].append(task.name)

        run_end = time.time()  # Global start time
        context.metadata["run_stats"] = {
            "start_time": datetime.fromtimestamp(run_start).isoformat(),
            "end_time": datetime.fromtimestamp(run_end).isoformat(),
            "total_tasks": len(self.tasks_sorted),
        }

        # Save results to context
        context.metadata["task_outcomes"] = task_outcomes

        # Global memory peak summary
        if context.metadata["task_memory"]:
            context.metadata["peak_memory_mb"] = max(
                context.metadata["task_memory"].values()
            )

        return context.results

    def visualize(
        self,
        status: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize the execution DAG with optional node status and save to file.

        Args:
            status (Optional[Dict[str, str]]): Task statuses
                (e.g., {"infer_types": "success"}).
            title (Optional[str]): Plot title.
            save_path (Optional[str]): Optional path to save the plot.
        """
        levels, _ = topo_sort_levels(self.graph)
        pos = assign_waterfall_positions(levels)
        draw_dag(
            self.graph,
            pos,
            status=status,
            title=title or "DSBF Execution DAG",
            save_path=save_path,
        )
