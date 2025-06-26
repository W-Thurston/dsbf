# dsbf/eda/graph.py
"""
Execution Graph module for DSBF.

Defines `Task` and `ExecutionGraph` classes which manage DAG-style lazy execution of
EDA tasks.  Includes support for dependency resolution, error handling, and DAG
visualization.
"""

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from networkx.drawing.nx_pydot import graphviz_layout


class Task:
    def __init__(self, name, func, requires=None):
        self.name = name
        self.func = func
        self.requires = requires or []
        self.result = None
        self.status = "pending"  # → "success" or "failed"

    def run(self, context):
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
    def __init__(self, tasks):
        self.tasks = tasks

    def run(self):
        context = {}
        for task in self.tasks:
            task.run(context)
        return context

    def visualize(self, output_path):
        G = nx.DiGraph()
        color_map = {"success": "#8BC34A", "failed": "#F44336", "pending": "#B0BEC5"}

        for task in self.tasks:
            G.add_node(task.name)
            for dep in task.requires:
                G.add_edge(dep, task.name)

        try:
            G.graph["graph"] = {"rankdir": "TB"}  # Left-to-right layout
            pos = graphviz_layout(G, prog="dot")
        except Exception as e:
            print(
                f"[Warning] Graphviz layout failed: {e}. Falling back to spring layout."
            )
            pos = nx.spring_layout(G, seed=42)

        node_colors = [color_map.get(task.status, "#B0BEC5") for task in self.tasks]

        plt.figure(figsize=(8, 10))  # Taller for top-down flow
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            edge_color="gray",
            node_size=2000,
            font_size=10,
            font_weight="bold",
            arrows=True,
        )

        legend_elements = [
            Patch(facecolor=color_map["success"], edgecolor="k", label="Success"),
            Patch(facecolor=color_map["failed"], edgecolor="k", label="Failed"),
            Patch(facecolor=color_map["pending"], edgecolor="k", label="Pending"),
        ]
        plt.legend(handles=legend_elements, loc="lower center", ncol=3, frameon=False)
        plt.title("Task Execution Graph")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
