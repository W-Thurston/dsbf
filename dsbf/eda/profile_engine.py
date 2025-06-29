# dsbf/eda/profile_engine.py

import os
from typing import Optional, Union

import networkx as nx
import pandas as pd
import polars as pl

from dsbf.config import load_default_config
from dsbf.core.base_engine import BaseEngine
from dsbf.core.context import AnalysisContext
from dsbf.eda.graph import ExecutionGraph, Task
from dsbf.eda.renderers.json_renderer import render as render_json
from dsbf.eda.task_loader import load_all_tasks
from dsbf.eda.task_registry import TASK_REGISTRY, get_all_task_specs
from dsbf.utils.data_loader import load_dataset
from dsbf.utils.task_utils import instantiate_task


class ProfileEngine(BaseEngine):
    """
    Orchestrates EDA profiling via task-based DAG execution.
    Loads data, constructs task graph, runs analysis, and exports report.
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or load_default_config()
        super().__init__(config)
        self.context: Optional[AnalysisContext] = None
        self.results: dict = {}
        self.inferred_stage: Optional[str] = None

    def get_result(self, task_name: str):
        return self.results.get(task_name)

    def get_all_results(self):
        return self.results

    def run(self):
        self._log("Starting profiling...", level="info")

        df = self._load_data()

        reference_path = self.config.get("engine", {}).get("reference_dataset_path")
        if reference_path and os.path.exists(reference_path):
            self._log(f"Loading reference dataset from: {reference_path}", level="info")
            reference_df = pd.read_csv(reference_path)
        else:
            reference_df = None
            if reference_path:
                self._log(
                    f"[WARNING] Reference path '{reference_path}' not found.",
                    level="info",
                )

        self.context = AnalysisContext(
            data=df,
            config=self.config,
            output_dir=self.output_dir,
            run_metadata=self.run_metadata,
            reference_data=reference_df,
        )

        # Load tasks into the global registry
        load_all_tasks()

        # Infer stage
        from dsbf.eda.stage_inference import infer_stage

        self.inferred_stage = infer_stage(df, self.config)
        self.context.stage = self.inferred_stage
        self.run_metadata["inferred_stage"] = self.inferred_stage
        self._log(f"Inferred data stage: {self.inferred_stage}", level="info")

        # Build graph and run tasks
        self._log("Building execution graph...", level="debug")
        graph = self.build_graph()
        self.results = graph.run(self.context, log_fn=self._log)

        # Optional DAG visualization
        if self.config.get("metadata", {}).get("visualize_dag", False):
            self._log("Visualizing task DAG...", level="debug")
            fig_path = os.path.join(self.fig_path, "dag.png")
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            status_dict = {name: result.status for name, result in self.results.items()}
            graph.visualize(save_path=fig_path, status=status_dict)

        # Export report
        self._log("Rendering JSON report...", level="debug")
        render_json(
            self.results,
            self.run_metadata,
            os.path.join(self.output_dir, "report.json"),
        )
        self.record_run()

    def _load_data(self) -> Union[pd.DataFrame, pl.DataFrame]:
        dataset_path = self.config.get("metadata", {}).get("dataset_path")
        dataset_name = self.config.get("metadata", {}).get("dataset_name", "iris")
        dataset_source = self.config.get("metadata", {}).get(
            "dataset_source", "sklearn"
        )

        backend = self.config.get("engine", {}).get("backend", "pandas")

        if dataset_path and os.path.exists(dataset_path):
            self._log(f"Loading dataset from: {dataset_path}", level="info")
            return pd.read_csv(dataset_path)

        self._log(
            f"Loading built-in dataset: {dataset_name} from {dataset_source}",
            level="info",
        )
        return load_dataset(name=dataset_name, source=dataset_source, backend=backend)

    def build_graph(self) -> ExecutionGraph:

        selected_depth = self.config.get("metadata", {}).get("profiling_depth", "full")
        PROFILING_DEPTH = {
            "basic": 1,
            "standard": 2,
            "full": 3,
        }

        # Optional: filter tasks based on profiling depth
        all_specs = get_all_task_specs()
        filtered_specs = [
            spec
            for spec in all_specs
            if spec.experimental is False
            and (
                selected_depth == "full"
                or PROFILING_DEPTH[spec.profiling_depth]
                <= PROFILING_DEPTH[selected_depth]
            )
        ]

        G = nx.DiGraph()
        for spec in filtered_specs:
            G.add_node(spec.name)
            for dep in spec.depends_on or []:
                if dep not in TASK_REGISTRY:
                    raise ValueError(
                        f"Task '{spec.name}' depends on unknown task '{dep}'"
                    )
                G.add_edge(dep, spec.name)

        sorted_names = list(nx.topological_sort(G))

        tasks = []
        for task_name in sorted_names:
            try:
                task_specific_cfg = self.config.get("tasks", {}).get(task_name, {})
                task_instance = instantiate_task(task_name, task_specific_cfg)
                requires = list(G.predecessors(task_name))
                tasks.append(
                    Task(name=task_name, task_instance=task_instance, requires=requires)
                )
            except KeyError:
                self._log(
                    f"[ERROR] Task '{task_name}' not found in registry.", level="error"
                )
                raise

        return ExecutionGraph(tasks)
