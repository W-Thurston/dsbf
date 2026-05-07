# dsbf/core/base_task.py

import os
from abc import ABC, abstractmethod
from typing import Any, dict, list, tuple

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_registry import TaskSpec
from dsbf.eda.task_result import TaskResult
from dsbf.utils.logging_utils import DSBFLogger, get_log_fn, setup_logger


class BaseTask(ABC):
    """
    Abstract base class for all DSBF tasks.
    Enforces a standard interface and result format.
    """

    context: AnalysisContext | None = None

    def __init__(self, name: str | None = None, config: dict[str, Any] | None = None):
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.input_data: Any = None
        self.output: TaskResult | None = None

    def set_input(self, input_data: Any) -> None:
        """Set the input data for the task (usually a DataFrame or dict)."""
        self.input_data = input_data

    def get_output(self) -> TaskResult | None:
        """Retrieve the output TaskResult after run()."""
        return self.output

    @abstractmethod
    def run(self) -> None:
        """Execute the task. Must set self.output as a TaskResult."""
        pass

    def get_output_path(self, filename: str) -> str:
        """
        Construct the path to save a file (e.g., figure) inside the output directory.

        Args:
            filename (str): The name of the file to save.

        Returns:
            str: Full path to the output file.
        """
        if not self.context or not self.context.output_dir:
            raise RuntimeError("Context or output_dir is not set in this task.")
        fig_dir: str = os.path.join(self.context.output_dir, "figs")
        os.makedirs(fig_dir, exist_ok=True)
        return os.path.join(fig_dir, filename)

    def get_task_param(self, key: str, default=None):
        """
        Get a config parameter defined for this task under config["tasks"][task_name].
        """
        return self.config.get(key, default)

    def get_shared_param(self, block_name: str, key: str, default=None):
        """
        Get a config parameter from a named shared block config["tasks"][block_name].

        Used for cross-task shared configuration such as time_series settings,
        where multiple tasks read from the same config block rather than their
        own task-specific section.

        Args:
            block_name: The shared block name under config["tasks"].
            key: The parameter key within that block.
            default: Value to return if not found.

        Returns:
            The config value, or default if not found.

        """
        ctx: AnalysisContext | None = self.context
        if ctx and isinstance(ctx.config, dict):
            return ctx.config.get("tasks", {}).get(block_name, {}).get(key, default)
        return default

    def get_engine_param(self, key: str, default=None):
        """
        Get a value from the 'engine' section of the global config.
        """
        ctx: AnalysisContext | None = self.context
        if ctx and isinstance(ctx.config, dict):
            return ctx.config.get("engine", {}).get(key, default)
        return default

    def get_metadata_param(self, key: str, default=None):
        """
        Get a value from the 'metadata' section of the global config.
        """
        ctx: AnalysisContext | None = self.context
        if ctx and isinstance(ctx.config, dict):
            return ctx.config.get("metadata", {}).get(key, default)
        return default

    def _log(self, msg: str, level: str = "debug") -> None:
        """
        Structured logging that prefers the context logger if available.
        Automatically prepends the task name.
        """
        if self.context and hasattr(self.context, "_log"):
            self.context._log(msg, level=level, task_name=self.name)
        else:
            fallback: DSBFLogger = setup_logger("dsbf.task", "info")
            get_log_fn(fallback, level)(f"[{self.name}] {msg}")

    def ensure_reliability_flags(self) -> dict:
        """
        Ensure global reliability flags are computed and cached in context.

        Returns:
            dict: dictionary of reliability flags.
        """
        if self.context is None:
            raise RuntimeError("AnalysisContext is not set in this task.")

        if not self.context.reliability_flags:
            self.context.compute_reliability_flags(self.input_data)

        return self.context.reliability_flags

    def add_guidance(
        self,
        result: TaskResult,
        column: str,
        phase: str,
        level: str,
        title: str,
        body: str,
        actions: list[dict[str, Any]],
        metric: dict[str, Any],
    ) -> None:
        """
        Attach a guidance blurb for a specific column and phase to a TaskResult.

        Guidance blurbs are the authoritative, phase-scoped narrative for each
        signal detected by a task. They are stored in report.json and rendered
        by the dashboard - tasks are the single source of truth.

        EDA blurbs (phase="eda") describe data as-is: what was observed and what
        it means about the distribution. No modeling language, no action chips.

        ML blurbs (phase="ml") prescribe what to do before modeling: which models
        are affected, what transforms are recommended, as structured actions an
        agent or user can act on directly.

        Args:
            result (TaskResult): The task result to attach guidance to.
            column (str): The column this guidance applies to.
            phase (str): "eda" or "ml".
            level (str): Severity - "info", "warn", "error", or "good".
            title (str): Short descriptive title for the finding.
            body (str): Full self-contained narrative. Must include the observed
                metric value, the direction/nature of the issue, and the
                implication. Should make sense without surrounding context
                (for LLM/agent consumption).
            actions (list[dict]): Structured actions. Empty list for EDA blurbs.
                Each action dict should have at minimum an "action" key.
                Example: {"action": "transform", "method": "log1p", "column": col}
            metric (dict): The observed metric values that triggered this blurb.
                Always include the raw numeric values, not just labels.
                Example: {"skewness": 2.84, "mean": 312.4, "median": 287.0}
        """
        if result.guidance is None:
            result.guidance = {}
        if column not in result.guidance:
            result.guidance[column] = {"eda": [], "ml": []}

        result.guidance[column][phase].append(
            {
                "phase": phase,
                "column": column,
                "level": level,
                "title": title,
                "body": body,
                "actions": actions,
                "metric": metric,
            }
        )

    def set_ml_signals(
        self,
        result: TaskResult,
        score: float,
        tags: list[str],
        recommendation: str,
    ) -> None:
        """
        Attach ML impact metadata to a TaskResult.

        Args:
            result (TaskResult): The task result object to modify.
            score (float): ML impact score between 0.0 and 1.0.
            tags (list[str]): Tags that describe the issue or remedy.
            recommendation (str): User-facing recommendation or note.
        """
        result.ml_impact_score = score
        result.recommendation_tags = tags
        if result.recommendations is None:
            result.recommendations = []
        result.recommendations.append(recommendation)

    def get_expected_types(self) -> list[str]:
        """
        Retrieve the expected semantic types from the task's registry entry.

        Returns:
            list[str]: list of expected analysis-intent dtypes (e.g., ['continuous'])
        """
        from dsbf.eda.task_registry import TASK_REGISTRY, _to_snake_case

        snake_name: str = _to_snake_case(self.__class__.__name__)
        spec: TaskSpec | None = TASK_REGISTRY.get(snake_name)
        return spec.expected_semantic_types or [] if spec else []

    def get_columns_by_intent(
        self, expected_types: list[str] | None = None
    ) -> tuple[list[str], dict[str, str]]:
        """
        Retrieve a list of columns whose analysis_intent_dtype matches the
        expected types, and return a dict of excluded columns with their
        inferred types for reporting.

        Args:
            expected_types (list[str] or None): list of allowed semantic types
                for the task. If None, will fall back to the task's registered
                expected_semantic_types.

        Returns:
            tuple[list[str], dict[str, str]]:
                - list of matching column names
                - dict of excluded columns with their mismatched types
        """
        if not self.context:
            return [], {}

        semantic_types: dict = self.context.get_metadata("semantic_types", {}) or {}
        _ = self.context.get_metadata("inferred_dtypes", {}) or {}

        if expected_types is None:
            expected_types = self.get_expected_types()

        matched: list = []
        excluded: dict = {}

        for col, intent_type in semantic_types.items():
            if "any" in expected_types or intent_type in expected_types:
                matched.append(col)
            else:
                excluded[col] = intent_type

        return matched, excluded

    # ── DataFrame access helpers ──────────────────────────────────────────────

    def get_dataframe_pandas(self) -> "pd.DataFrame":
        """
        Return the input DataFrame as a pandas DataFrame.

        If the input is a Polars DataFrame it is converted to pandas.
        If it is already pandas it is returned as-is (zero copy).

        Use this in tasks that require scipy, statsmodels, sklearn, or
        seaborn — libraries that do not accept Polars natively.

        Returns:
            pandas DataFrame.
        """
        from dsbf.utils.backend import is_polars

        df = self.input_data
        if is_polars(df):
            return df.to_pandas()
        return df

    def get_dataframe(self):
        """
        Return the input DataFrame in its original backend format.

        Unlike ``get_dataframe_pandas()``, this never converts — the
        caller receives a Polars DataFrame when Polars was used to load
        the data, and a pandas DataFrame otherwise.

        Use this in tasks that are pure aggregations (null counts,
        value counts, shape checks, etc.) and have been written to
        handle both backends via ``is_polars()`` branching.

        Returns:
            pandas or Polars DataFrame, unchanged.
        """
        return self.input_data

    def setup_run(
        self,
        log_label: str | None = None,
    ) -> "tuple[Any, list[str], dict[str, str]]":
        """
        Handle the common opening sequence for tasks that need pandas.

        Combines the three lines that appear at the top of almost every
        ``run()`` method into a single call:

        1. Retrieve ``self.input_data``
        2. Convert to pandas if the input is Polars
        3. Call ``get_columns_by_intent()``
        4. Emit a debug log with the column count

        This is for tasks that require pandas (scipy, statsmodels,
        sklearn, seaborn dependencies). Tasks that can stay Polars-
        native should use ``setup_run_native()`` instead.

        Args:
            log_label: Human-readable type label for the debug log,
                e.g. ``"'continuous'"`` or ``"['categorical', 'text']"``.
                Defaults to ``"eligible"``.

        Returns:
            tuple of (df_pandas, matched_cols, excluded).

        Example::

            df, matched_cols, excluded = self.setup_run("'continuous'")
        """
        from dsbf.utils.backend import is_polars

        df = self.input_data
        if is_polars(df):
            df = df.to_pandas()

        matched_cols, excluded = self.get_columns_by_intent()
        label: str = log_label or "eligible"
        self._log(
            f"    Processing {len(matched_cols)} {label} column(s)",
            "debug",
        )
        return df, matched_cols, excluded

    def setup_run_native(
        self,
        log_label: str | None = None,
    ) -> "tuple[Any, list[str], dict[str, str]]":
        """
        Handle the common opening sequence for tasks that stay backend-native.

        Like ``setup_run()`` but does NOT convert the DataFrame. The
        caller receives the input in its original format (Polars or
        pandas) and is responsible for branching on ``is_polars(df)``
        where the two APIs differ.

        Use this for pure aggregation tasks (null counts, value counts,
        shape checks, string length stats, etc.) that do not need
        scipy, statsmodels, sklearn, or seaborn.

        Args:
            log_label: Human-readable type label for the debug log.
                Defaults to ``"eligible"``.

        Returns:
            tuple of (df_native, matched_cols, excluded).

        Example::

            df, matched_cols, excluded = self.setup_run_native()
            if is_polars(df):
                pass  # Polars computation
            else:
                pass  # Pandas computation
        """
        df = self.input_data
        matched_cols, excluded = self.get_columns_by_intent()
        label: str = log_label or "eligible"
        self._log(
            f"    Processing {len(matched_cols)} {label} column(s)",
            "debug",
        )
        return df, matched_cols, excluded

    def make_empty_result(
        self,
        message: str,
        excluded: dict[str, str] | None = None,
    ) -> TaskResult:
        """
        Build a success TaskResult for the zero-eligible-columns case.

        Every task that filters columns by semantic type needs to handle
        the case where no columns match. This produces a consistent
        ``status="success"`` result rather than an error, since "no
        eligible columns" is a valid dataset state (e.g. an all-
        categorical dataset passed to a continuous-only task).

        Args:
            message: Summary message describing why no computation
                was performed, e.g.
                ``"No continuous columns found — VIF not computed."``.
            excluded: dict of excluded columns from
                ``get_columns_by_intent()``. Included in metadata when
                provided.

        Returns:
            TaskResult with empty data and the provided summary message.

        Example::

            df, matched_cols, excluded = self.setup_run("'continuous'")
            if not matched_cols:
                self.output = self.make_empty_result(
                    "No continuous columns found.", excluded
                )
                return
        """
        meta: dict[str, Any] = {}
        if excluded is not None:
            meta["excluded_columns"] = excluded
        return TaskResult(
            name=self.name,
            status="success",
            summary={"message": message},
            data={},
            metadata=meta,
        )

    def get_column_type_info(self, columns: list[str]) -> dict[str, dict[str, str]]:
        """
        Returns a dictionary mapping each column name to its inferred and
         analysis-intent dtypes.

        Args:
            columns (list[str]): list of column names to include

        Returns:
            dict[str, dict[str, str]]: {
                column_name: {
                    "inferred_dtype": ...,
                    "analysis_intent_dtype": ...
                },
                ...
            }
        """
        if not self.context:
            return {}

        semantic_types: dict = self.context.get_metadata("semantic_types", {}) or {}
        inferred_types: dict = self.context.get_metadata("inferred_dtypes", {}) or {}

        return {
            col: {
                "inferred_dtype": inferred_types.get(col, "unknown"),
                "analysis_intent_dtype": semantic_types.get(col, "unknown"),
            }
            for col in columns
        }
