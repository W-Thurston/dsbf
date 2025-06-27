# dsbf/core/base_task.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult


class BaseTask(ABC):
    """
    Abstract base class for all DSBF tasks.
    Enforces a standard interface and result format.
    """

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

    # def run_with_context(self, context: AnalysisContext) -> TaskResult:
    #     """
    #     Convenience method to execute this task within an AnalysisContext.
    #     Preferred pattern is to call `context.run_task(task)` directly.
    #     """
    #     self.set_input(context.df)
    #     self.config.update(context.config)
    #     self.run()

    #     result = self.get_output()
    #     if result is None:
    #         raise RuntimeError(f"Task '{self.name}' did not produce a TaskResult.")

    #     context.set_result(self.name, result)
    #     return result
