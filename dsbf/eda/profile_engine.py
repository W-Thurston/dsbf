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
from dsbf.eda.stage_inference import infer_stage
from dsbf.eda.task_registry import (
    TASK_REGISTRY,
    get_all_task_specs,
    get_plugin_warnings,
    load_task_group,
    set_plugin_logger,
)
from dsbf.utils.config_validation import validate_config_and_graph
from dsbf.utils.data_loader import load_dataset
from dsbf.utils.data_utils import data_sampling
from dsbf.utils.report_utils import render_user_report, write_metadata_report
from dsbf.utils.task_utils import filter_tasks, instantiate_task


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
        self._log("Starting profiling...", level="stage")

        df = self._load_data()
        df, sampling_info = data_sampling(df, self.config, log_fn=self._log)

        if sampling_info:
            self.run_metadata["sampling"] = sampling_info

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
        task_groups = self.config.get("task_groups", ["core"])
        self._log(f"Loading task groups: {task_groups}", level="debug")

        set_plugin_logger(self._log)

        for group in task_groups:
            load_task_group(group)

        plugin_warnings = get_plugin_warnings()
        if plugin_warnings:
            self.context.set_metadata("plugin_warnings", plugin_warnings)
            self.run_metadata["plugin_warnings"] = plugin_warnings

        # Config + DAG validation
        self._log("Validating config, registry, and DAG...", level="stage")
        errors = validate_config_and_graph(self.config)
        strict = self.config.get("safety", {}).get("strict_mode", False)
        if errors:
            for err in errors:
                self._log(f"[CONFIG VALIDATION] {err}", level="warn")
            if strict:
                raise ValueError(
                    (
                        "Strict mode is enabled — halting due to "
                        f"{len(errors)} config/DAG issue(s)."
                    )
                )

        # Infer stage
        self.inferred_stage = infer_stage(df, self.config)
        self.context.stage = self.inferred_stage
        self.run_metadata["inferred_stage"] = self.inferred_stage
        self._log(f"Inferred data stage: {self.inferred_stage}", level="stage")

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

        # Export user-facing report (results only)
        self._log("Rendering JSON report...", level="debug")
        render_user_report(
            results=self.context.results,
            output_path=os.path.join(self.output_dir, "report.json"),
        )

        # Write separate runtime metadata
        write_metadata_report(self.context)

        self.record_run()

    def _load_data(self) -> Union[pd.DataFrame, pl.DataFrame]:
        dataset_path = self.config.get("metadata", {}).get("dataset_path")
        dataset_name = self.config.get("metadata", {}).get("dataset_name", "iris")
        dataset_source = self.config.get("metadata", {}).get(
            "dataset_source", "sklearn"
        )

        backend = self.config.get("engine", {}).get("backend", "pandas")

        if dataset_path and os.path.exists(dataset_path):
            self._log(f"Loading dataset from: {dataset_path}", level="stage")
            return pd.read_csv(dataset_path)

        self._log(
            f"Loading built-in dataset: {dataset_name} from {dataset_source}",
            level="stage",
        )
        return load_dataset(name=dataset_name, source=dataset_source, backend=backend)

    def build_graph(self) -> ExecutionGraph:

        selected_depth = self.config.get("metadata", {}).get("profiling_depth", "full")
        PROFILING_DEPTH = {
            "basic": 1,
            "standard": 2,
            "full": 3,
        }

        all_specs = get_all_task_specs()
        task_filters = self.config.get("task_selection", {})

        include_domains = task_filters.get("include_domains")
        exclude_stages = task_filters.get("exclude_stages")
        max_runtime = task_filters.get("max_runtime_estimate")

        criteria = {}
        if include_domains:
            criteria["domain"] = include_domains
        if max_runtime:
            runtime_order = ["fast", "moderate", "slow"]
            allowed_runtimes = runtime_order[: runtime_order.index(max_runtime) + 1]
            criteria["runtime_estimate"] = allowed_runtimes

        allowed_names = set(filter_tasks(criteria))

        # Remove tasks in excluded stages
        if exclude_stages:
            allowed_names = {
                name
                for name in allowed_names
                if TASK_REGISTRY[name].stage not in exclude_stages
            }

        # Fallback
        if not allowed_names:
            self._log(
                "[WARNING] No tasks matched filters — falling back to core domain",
                "warn",
            )
            allowed_names = {
                spec.name
                for spec in all_specs
                if spec.domain == "core" and not spec.experimental
            }

        filtered_specs = [
            spec
            for spec in all_specs
            if spec.name in allowed_names
            and not spec.experimental
            and (
                selected_depth == "full"
                or PROFILING_DEPTH[spec.profiling_depth]
                <= PROFILING_DEPTH[selected_depth]
            )
        ]
        self._log(f"Applying filters: {criteria}", "debug")
        self._log(
            f"Selected {len(filtered_specs)} tasks after filtering and depth checks",
            "debug",
        )

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
