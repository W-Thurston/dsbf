# dsbf/eda/graph.py
"""
Execution Graph module for DSBF.

Defines `Task` and `ExecutionGraph` classes which manage DAG-style lazy execution of
EDA tasks.  Includes support for dependency resolution, error handling, and DAG
visualization.
"""

from typing import Dict, List, Optional

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

    def run(self, context: AnalysisContext) -> Dict[str, TaskResult]:
        for task in self.tasks_sorted:
            task.run(context)  # calls AnalysisContext.run_task(task.task_instance)
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
