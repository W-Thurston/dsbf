# dsbf/eda/profile_engine.py
"""
ProfileEngine for DSBF.

Implements a full EDA engine that loads data, infers stage, builds a profiling DAG,
runs tasks, visualizes the graph, and renders structured reports (Markdown, JSON).
"""

import json
import os

import pandas as pd

from dsbf.core.base_engine import BaseEngine
from dsbf.eda.graph import ExecutionGraph, Task
from dsbf.eda.renderers.json_renderer import render as render_json
from dsbf.eda.stage_inference import infer_stage
from dsbf.eda.tasks.categorical_length_stats import categorical_length_stats
from dsbf.eda.tasks.compute_correlations import compute_correlations
from dsbf.eda.tasks.compute_entropy import compute_entropy
from dsbf.eda.tasks.detect_constant_columns import detect_constant_columns
from dsbf.eda.tasks.detect_duplicates import detect_duplicates
from dsbf.eda.tasks.detect_high_cardinality import detect_high_cardinality
from dsbf.eda.tasks.detect_id_columns import detect_id_columns
from dsbf.eda.tasks.detect_skewness import detect_skewness
from dsbf.eda.tasks.infer_types import infer_types
from dsbf.eda.tasks.missingness_heatmap import missingness_heatmap
from dsbf.eda.tasks.missingness_matrix import missingness_matrix
from dsbf.eda.tasks.sample_head import sample_head
from dsbf.eda.tasks.sample_tail import sample_tail
from dsbf.eda.tasks.summarize_modes import summarize_modes
from dsbf.eda.tasks.summarize_nulls import summarize_nulls
from dsbf.eda.tasks.summarize_numeric import summarize_numeric
from dsbf.eda.tasks.summarize_text_fields import summarize_text_fields
from dsbf.eda.tasks.summarize_unique import summarize_unique
from dsbf.utils.data_loader import load_dataset


class ProfileEngine(BaseEngine):
    def run(self):
        self._log("Starting profiling...", level="info")

        dataset_path = self.config.get("dataset")
        dataset_source = self.config.get("dataset_source", "sklearn")
        dataset_name = self.config.get("dataset_name", "iris")
        backend = self.config.get("backend", "pandas")
        visualize_flag = self.config.get("visualize_dag", False)
        profiling_depth = self.config.get("profiling_depth", "standard")

        if dataset_path and os.path.exists(dataset_path):
            self._log(f"Loading dataset from: {dataset_path}", level="info")
            df = pd.read_csv(dataset_path)
        else:
            self._log(
                f"Loading built-in dataset: {dataset_name} from {dataset_source}",
                level="info",
            )
            df = load_dataset(name=dataset_name, source=dataset_source, backend=backend)

        self.inferred_stage = infer_stage(df, self.config)
        self.run_metadata["inferred_stage"] = self.inferred_stage
        self._log(f"Inferred data stage: {self.inferred_stage}", level="info")

        self._log("Building execution graph...", level="debug")
        graph = self.build_graph(df, profiling_depth)

        self._log("Running task graph...", level="info")
        results = graph.run()

        if visualize_flag:
            self._log("Visualizing DAG...", level="debug")
            fig_path = os.path.join(self.fig_path, "dag.png")
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            graph.visualize(fig_path)

        layout_name = self.config.get("layout_name", "default")
        self.layout_name = layout_name

        if layout_name.endswith(".json") and os.path.exists(layout_name):
            self._log(f"Using custom layout: {layout_name}", level="info")
            with open(layout_name, "r") as f:
                self.layout_spec = json.load(f)
        else:
            builtin_path = os.path.join("dsbf", "layouts", f"{layout_name}.json")
            if os.path.exists(builtin_path):
                self._log(f"Using built-in layout: {layout_name}", level="info")
                with open(builtin_path, "r") as f:
                    self.layout_spec = json.load(f)
            else:
                raise FileNotFoundError(
                    f"Layout '{layout_name}' not found as path or built-in file."
                )

        # Markdown Rendering
        # self._log("Rendering markdown report...", level="debug")
        # results["__stage_callout__"] = (
        #     f"This dataset appears to be in the **{self.inferred_stage}** stage "
        #     "based on null ratios, column types, and value distribution."
        # )
        # with open(os.path.join(self.output_dir, "report.md"), "w") as f:
        #     f.write("# Profile Summary\n")

        #     if "__stage_callout__" in results:
        #         f.write("\n> ⚙️ **Inferred Data Stage**: " +
        #                   self.inferred_stage + "\n")
        #         f.write("> " + results["__stage_callout__"] + "\n")
        #         del results["__stage_callout__"]

        #     for key, section in results.items():
        #         f.write(f"\n## {key}\n")
        #         f.write(f"```\n{section}\n```\n")

        # JSON Rendering
        self._log("Rendering JSON report...", level="debug")
        render_json(
            results, self.run_metadata, os.path.join(self.output_dir, "report.json")
        )

        self.run_metadata["inferred_stage"] = self.inferred_stage
        self.record_run()

    def build_graph(self, df, depth):
        DEPTH_LEVELS = {"basic": 0, "standard": 1, "full": 2}

        tasks = [
            Task("infer_types", lambda: infer_types(df)),
            Task(
                "summarize_nulls",
                lambda _: summarize_nulls(df),
                requires=["infer_types"],
            ),
            Task(
                "summarize_numeric",
                lambda _: summarize_numeric(df),
                requires=["infer_types"],
            ),
        ]

        if DEPTH_LEVELS[depth] >= 1:
            tasks.extend(
                [
                    Task(
                        "detect_constant_columns",
                        lambda _: detect_constant_columns(df),
                        requires=["infer_types"],
                    ),
                    Task(
                        "detect_duplicates",
                        lambda _: detect_duplicates(df),
                        requires=["infer_types"],
                    ),
                    Task(
                        "summarize_unique",
                        lambda _: summarize_unique(df),
                        requires=["infer_types"],
                    ),
                    Task(
                        "summarize_modes",
                        lambda _: summarize_modes(df),
                        requires=["infer_types"],
                    ),
                ]
            )

        if DEPTH_LEVELS[depth] >= 2:
            tasks.extend(
                [
                    Task(
                        "categorical_length_stats",
                        lambda _: categorical_length_stats(df),
                        requires=["infer_types"],
                    ),
                    Task(
                        "compute_correlations",
                        lambda _: compute_correlations(df),
                        requires=["infer_types"],
                    ),
                    Task(
                        "compute_entropy",
                        lambda _: compute_entropy(df),
                        requires=["infer_types"],
                    ),
                    Task(
                        "detect_high_cardinality",
                        lambda _: detect_high_cardinality(df),
                        requires=["infer_types"],
                    ),
                    Task(
                        "detect_id_columns",
                        lambda _: detect_id_columns(df),
                        requires=["infer_types"],
                    ),
                    Task(
                        "detect_skewness",
                        lambda _: detect_skewness(df),
                        requires=["infer_types"],
                    ),
                    Task(
                        "missingness_heatmap",
                        lambda _: missingness_heatmap(df, self.fig_path),
                        requires=["infer_types"],
                    ),
                    Task(
                        "missingness_matrix",
                        lambda _: missingness_matrix(df, self.fig_path),
                        requires=["infer_types"],
                    ),
                    Task(
                        "sample_head",
                        lambda _: sample_head(df),
                        requires=["infer_types"],
                    ),
                    Task(
                        "sample_tail",
                        lambda _: sample_tail(df),
                        requires=["infer_types"],
                    ),
                    Task(
                        "summarize_text_fields",
                        lambda _: summarize_text_fields(df),
                        requires=["infer_types"],
                    ),
                ]
            )

        return ExecutionGraph(tasks)
