# dsbf/core/context.py

from typing import TYPE_CHECKING, Any, Dict, Optional

from dsbf.eda.task_result import TaskResult

if TYPE_CHECKING:
    from dsbf.core.base_task import BaseTask

MESSAGE_VERBOSITY_LEVELS = {"quiet": 0, "info": 1, "debug": 2}


class AnalysisContext:
    """
    Shared context object for DSBF runs.
    Holds the main input dataframe, global config, task results, and metadata.
    """

    def __init__(
        self,
        data: Any,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        reference_data: Optional[Any] = None,
    ):
        self.data = data
        self.config = config or {}
        self.output_dir = output_dir
        self.run_metadata = run_metadata or {}
        self.reference_data = reference_data

        self.results: Dict[str, TaskResult] = {}
        self.metadata: Dict[str, Any] = {}
        self.stage: Optional[str] = None

    from dsbf.core.base_engine import (  # Import this if not already
        MESSAGE_VERBOSITY_LEVELS,
    )

    def _log(self, msg: str, level: str = "info") -> None:
        """
        Log messages conditionally based on configured verbosity.
        """
        verbosity_str = self.config.get("metadata", {}).get("message_verbosity", "info")
        verbosity_level = MESSAGE_VERBOSITY_LEVELS.get(verbosity_str, 1)
        msg_level = MESSAGE_VERBOSITY_LEVELS.get(level, 1)

        if verbosity_level >= msg_level:
            print(f"[AnalysisContext] {msg}")

    def get_config(self, key: str, default=None):
        return self.config.get(key, default)

    def set_result(self, task_name: str, result: TaskResult):
        self.results[task_name] = result

    def get_result(self, task_name: str) -> Optional[TaskResult]:
        return self.results.get(task_name)

    def has_result(self, task_name: str) -> bool:
        return task_name in self.results

    def set_metadata(self, key: str, value: Any):
        self.metadata[key] = value

    def get_metadata(self, key: str, default=None):
        return self.metadata.get(key, default)

    def validate(self):
        if self.data is None:
            raise ValueError("No data loaded into context.")
        if not hasattr(self.data, "shape"):
            raise TypeError("Expected a DataFrame-like object with `.shape`.")

    def __repr__(self):
        return (
            f"<AnalysisContext data=({type(self.data).__name__}), "
            f"config_keys={list(self.config.keys())}, "
            f"tasks={list(self.results.keys())}>"
        )

    def run_task(self, task: "BaseTask") -> TaskResult:
        task.set_input(self.data)

        task.context = self

        task.run()

        result = task.get_output()
        if result is None:
            raise RuntimeError(f"Task '{task.name}' did not produce a TaskResult.")

        self.set_result(task.name, result)
        return result
