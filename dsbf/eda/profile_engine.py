# dsbf/eda/profile_engine.py

import os
from typing import Optional, Union

import pandas as pd
import polars as pl

from dsbf.core.base_engine import BaseEngine
from dsbf.core.context import AnalysisContext
from dsbf.eda.graph import ExecutionGraph, Task
from dsbf.eda.renderers.json_renderer import render as render_json
from dsbf.eda.task_loader import load_all_tasks
from dsbf.eda.task_registry import get_task_by_name
from dsbf.utils.data_loader import load_dataset


class ProfileEngine(BaseEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        self.context: Optional[AnalysisContext] = None

    def run(self):
        self._log("Starting profiling...", level="info")

        # --- Load dataset ---
        df = self._load_data()
        self.context = AnalysisContext(df, config=self.config)

        # --- Load and register all tasks ---
        load_all_tasks()

        # --- Infer stage ---
        from dsbf.eda.stage_inference import infer_stage

        self.inferred_stage = infer_stage(df, self.config)
        self.run_metadata["inferred_stage"] = self.inferred_stage
        self._log(f"Inferred data stage: {self.inferred_stage}", level="info")

        # --- Build graph ---
        self._log("Building execution graph...", level="debug")
        graph = self.build_graph(self.config)

        # --- Run graph ---
        results = graph.run(self.context)

        # --- Visualize DAG ---
        if self.config.get("visualize_dag", False):
            fig_path = os.path.join(self.fig_path, "dag.png")
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            graph.visualize(save_path=fig_path)

        # --- Render report ---
        self._log("Rendering JSON report...", level="debug")
        render_json(
            results, self.run_metadata, os.path.join(self.output_dir, "report.json")
        )
        self.record_run()

    def _load_data(self) -> Union[pd.DataFrame, pl.DataFrame]:
        dataset_path = self.config.get("dataset")
        dataset_name = self.config.get("dataset_name", "iris")
        dataset_source = self.config.get("dataset_source", "sklearn")
        backend = self.config.get("backend", "pandas")

        if dataset_path and os.path.exists(dataset_path):
            self._log(f"Loading dataset from: {dataset_path}", level="info")
            return pd.read_csv(dataset_path)

        self._log(
            f"Loading built-in dataset: {dataset_name} from {dataset_source}",
            level="info",
        )
        return load_dataset(name=dataset_name, source=dataset_source, backend=backend)

    def build_graph(self, config) -> ExecutionGraph:
        from dsbf.eda.depth_levels import (
            DEPTH_LEVELS,
        )

        depth = config.get("profiling_depth", "standard")
        task_specs = DEPTH_LEVELS.get(depth, [])

        tasks = []
        for task_name, deps in task_specs:
            task_cls = get_task_by_name(task_name)
            tasks.append(Task(name=task_name, task_instance=task_cls, requires=deps))

        return ExecutionGraph(tasks)
