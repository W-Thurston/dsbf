# dsbf/eda/graph.py
"""
Execution Graph module for DSBF.

Defines `Task` and `ExecutionGraph` classes which manage DAG-style lazy execution of
EDA tasks.  Includes support for dependency resolution, error handling, and DAG
visualization.
"""

from typing import Dict, List, Optional, Tuple, cast

import networkx as nx

from dsbf.utils.dag_layout import assign_waterfall_positions, draw_dag, topo_sort_levels


class Task:
    def __init__(self, name: str, func, requires: Optional[List[str]] = None):
        """
        Represents a single task in the execution graph.

        Args:
            name (str): Unique task name.
            func (callable): Task function.
            requires (List[str], optional): List of task names this task depends on.
        """
        self.name = name
        self.func = func
        self.requires = requires or []
        self.result = None
        self.status = "pending"  # → "success" or "failed"

    def run(self, context: dict):
        """Executes the task using the given context for dependencies."""
        args = [context[dep] for dep in self.requires]
        try:
            self.result = self.func(*args)
            self.status = "success"
        except Exception as e:
            self.status = "failed"
            raise e
        context[self.name] = self.result
        return self.result


class ExecutionGraph:
    def __init__(self, tasks: List[Task]):
        """
        Container for managing and executing a DAG of tasks.

        Args:
            tasks (List[Task]): List of Task instances forming a DAG.
        """
        self.tasks = tasks

    def run(self) -> dict:
        """Runs all tasks in the graph in topological order."""
        context = {}
        for task in self.tasks:
            task.run(context)
        return context

    def visualize(self, output_path: str):
        """
        Visualizes the execution graph with node coloring by task status.

        Args:
            output_path (str): Path to save the visualization image.
        """
        G = nx.DiGraph()
        for task in self.tasks:
            G.add_node(task.name)
            for dep in task.requires:
                G.add_edge(dep, task.name)

        try:
            levels, _ = topo_sort_levels(G)
            pos = assign_waterfall_positions(levels)
        except Exception as e:
            print(f"[Warning] Layout fallback triggered: {e}")
            pos = nx.spring_layout(G, seed=42)

        status_dict = {task.name: task.status for task in self.tasks}
        pos = cast(Dict[str, Tuple[int, int]], pos)
        draw_dag(
            G,
            pos,
            status=status_dict,
            title="Task Execution Graph",
            save_path=output_path,
        )
