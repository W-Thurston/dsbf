# dsbf/core/base_task.py

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.utils.logging_utils import get_log_fn, setup_logger


class BaseTask(ABC):
    """
    Abstract base class for all DSBF tasks.
    Enforces a standard interface and result format.
    """

    context: Optional[AnalysisContext] = None

    def __init__(
        self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ):
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.input_data: Any = None
        self.output: Optional[TaskResult] = None

    def set_input(self, input_data: Any) -> None:
        """Set the input data for the task (usually a DataFrame or dict)."""
        self.input_data = input_data

    def get_output(self) -> Optional[TaskResult]:
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
        fig_dir = os.path.join(self.context.output_dir, "figs")
        os.makedirs(fig_dir, exist_ok=True)
        return os.path.join(fig_dir, filename)

    def get_task_param(self, key: str, default=None):
        """
        Get a config parameter defined for this task under config["tasks"][task_name].
        """
        return self.config.get(key, default)

    def get_engine_param(self, key: str, default=None):
        """
        Get a value from the 'engine' section of the global config.
        """
        ctx = self.context
        if ctx and isinstance(ctx.config, dict):
            return ctx.config.get("engine", {}).get(key, default)
        return default

    def get_metadata_param(self, key: str, default=None):
        """
        Get a value from the 'metadata' section of the global config.
        """
        ctx = self.context
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
            fallback = setup_logger("dsbf.task", "info")
            get_log_fn(fallback, level)(f"[{self.name}] {msg}")

    def ensure_reliability_flags(self) -> Dict:
        """
        Ensure global reliability flags are computed and cached in context.

        Returns:
            Dict: Dictionary of reliability flags.
        """
        if self.context is None:
            raise RuntimeError("AnalysisContext is not set in this task.")

        if not self.context.reliability_flags:
            self.context.compute_reliability_flags(self.input_data)

        return self.context.reliability_flags

    def set_ml_signals(
        self,
        result: TaskResult,
        score: float,
        tags: List[str],
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

    def get_expected_types(self) -> List[str]:
        """
        Retrieve the expected semantic types from the task's registry entry.

        Returns:
            List[str]: List of expected analysis-intent dtypes (e.g., ['continuous'])
        """
        from dsbf.eda.task_registry import TASK_REGISTRY, _to_snake_case

        snake_name = _to_snake_case(self.__class__.__name__)
        spec = TASK_REGISTRY.get(snake_name)
        return spec.expected_semantic_types or [] if spec else []

    def get_columns_by_intent(
        self, expected_types: Optional[List[str]] = None
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Retrieve a list of columns whose analysis_intent_dtype matches the
        expected types, and return a dict of excluded columns with their
        inferred types for reporting.

        Args:
            expected_types (list[str] or None): List of allowed semantic types
                for the task. If None, will fall back to the task's registered
                expected_semantic_types.

        Returns:
            Tuple[list[str], Dict[str, str]]:
                - List of matching column names
                - Dict of excluded columns with their mismatched types
        """
        if not self.context:
            return [], {}

        semantic_types = self.context.get_metadata("semantic_types", {}) or {}
        _ = self.context.get_metadata("inferred_dtypes", {}) or {}

        if expected_types is None:
            expected_types = self.get_expected_types()

        matched = []
        excluded = {}

        for col, intent_type in semantic_types.items():
            if intent_type in expected_types:
                matched.append(col)
            else:
                excluded[col] = intent_type

        return matched, excluded

    def get_column_type_info(self, columns: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Returns a dictionary mapping each column name to its inferred and
         analysis-intent dtypes.

        Args:
            columns (List[str]): List of column names to include

        Returns:
            Dict[str, Dict[str, str]]: {
                column_name: {
                    "inferred_dtype": ...,
                    "analysis_intent_dtype": ...
                },
                ...
            }
        """
        if not self.context:
            return {}

        semantic_types = self.context.get_metadata("semantic_types", {}) or {}
        inferred_types = self.context.get_metadata("inferred_dtypes", {}) or {}

        return {
            col: {
                "inferred_dtype": inferred_types.get(col, "unknown"),
                "analysis_intent_dtype": semantic_types.get(col, "unknown"),
            }
            for col in columns
        }
