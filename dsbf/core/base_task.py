# dsbf/core/base_task.py

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult


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

    def _log(self, msg: str, level: str = "info") -> None:
        """
        Safe logging wrapper that uses context._log() if available.
        Prevents Pylance 'Optional' warnings.
        """
        if self.context and hasattr(self.context, "_log"):
            self.context._log(msg, level)

    def ensure_reliability_flags(self) -> Dict:
        if self.context is None:
            raise RuntimeError("AnalysisContext is not set in this task.")

        if not self.context.reliability_flags:
            self.context.compute_reliability_flags(self.input_data)

        return self.context.reliability_flags

    def set_ml_signals(
        self,
        result: TaskResult,
        score: float,
        tags: list[str],
        recommendation: str,
    ) -> None:
        """
        Attach ML impact metadata to a TaskResult.
        """
        result.ml_impact_score = score
        result.recommendation_tags = tags
        if result.recommendations is None:
            result.recommendations = []
        result.recommendations.append(recommendation)
