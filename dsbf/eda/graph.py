# dsbf/eda/graph.py
"""
Execution Graph module for DSBF.

Defines `Task` and `ExecutionGraph` classes which manage DAG-style lazy execution of
EDA tasks.  Includes support for dependency resolution, error handling, and DAG
visualization.
"""

import time
from typing import Callable, Dict, List, Optional

import networkx as nx

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

        for task in self.tasks_sorted:
            deps = task.requires
            if deps:
                dep_statuses = [f"{dep}: {self.task_map[dep].status}" for dep in deps]
                if log_fn:
                    log_fn(
                        (
                            f"[DEBUG] Starting task: {task.name}"
                            f" (depends on: {', '.join(dep_statuses)})"
                        ),
                        "debug",
                    )
            else:
                if log_fn:
                    log_fn(
                        f"[DEBUG] Starting task: {task.name} (no dependencies)", "debug"
                    )

            failed_deps = [
                dep for dep in deps if self.task_map[dep].status != "success"
            ]
            if failed_deps:
                task.status = "skipped"
                task_outcomes["skipped"].append(task.name)
                if log_fn:
                    log_fn(
                        (
                            f"Skipping task '{task.name}'"
                            f" due to failed dependency: {failed_deps}"
                        ),
                        "debug",
                    )
                continue

            # Try running the task
            start_time = time.time()
            try:
                _ = task.run(context)
                duration = time.time() - start_time
                if log_fn:
                    log_fn(
                        (
                            f"[DEBUG] Completed task: {task.name} in"
                            f" {duration:.2f}s (status: {task.status})"
                        ),
                        "debug",
                    )
                task_outcomes["success"].append(task.name)
            except Exception as e:
                duration = time.time() - start_time
                if log_fn:
                    log_fn(
                        (
                            f"[ERROR] Task {task.name} failed after"
                            f" {duration:.2f}s with error: {e}"
                        ),
                        "debug",
                    )
                task_outcomes["failed"].append(task.name)

        # Save results to context
        context.metadata["task_outcomes"] = task_outcomes
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
