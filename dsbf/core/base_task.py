# dsbf/core/base_task.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

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
