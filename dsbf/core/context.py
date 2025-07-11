# dsbf/core/context.py

from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars
from dsbf.utils.logging_utils import DSBFLogger, get_log_fn, setup_logger
from dsbf.utils.reliability_stats import compute_reliability_flags as compute_flags

if TYPE_CHECKING:
    from dsbf.core.base_task import BaseTask


class AnalysisContext:
    """
    Shared context object for DSBF runs.
    Holds the main input dataframe, global config, task results, and metadata.

    Attributes:
        data (Any): The input dataframe (Pandas or Polars).
        config (dict): Full config dictionary (engine, metadata, task-level).
        results (dict[str, TaskResult]): Stores each task's final output.
        metadata (dict): Flexible key-value store for:
            - 'semantic_types': analysis-intent column types
            - 'inferred_dtypes': raw type inference results
            - 'task_durations': per-task runtime (sec)
            - 'plugin_warnings': plugin validation messages
            - Any custom task-level or engine-level signals
    """

    def __init__(
        self,
        data: Any,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        reference_data: Optional[Any] = None,
    ):
        """
        Initialize shared context object for a single DSBF profiling run.
        Stores raw data, configuration, outputs, and metadata across tasks.
        """
        self.data = data
        self.config = config or {}
        self.output_dir = output_dir
        self.run_metadata = run_metadata or {}
        self.reference_data = reference_data

        self.results: Dict[str, TaskResult] = {}  # Stores outputs by task name
        self.metadata: Dict[str, Any] = {}  # Shared metadata from tasks or engine
        self.stage: Optional[str] = None  # Inferred data stage (raw, cleaned, etc.)
        self.reliability_flags: Dict[str, Any] = {}  # Cached global reliability info

        self.logger: DSBFLogger = setup_logger(
            "dsbf.context",
            self.config.get("metadata", {}).get("message_verbosity", "info"),
        )

    def _log(
        self, msg: str, level: str = "info", task_name: Optional[str] = None
    ) -> None:
        """
        Structured logging for context operations using DSBF verbosity levels.
        Optionally prefixes the message with a task name.
        """
        log_fn = get_log_fn(self.logger, level)
        prefix = f"[{task_name}] " if task_name else ""
        log_fn(prefix + msg)

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

        # import statement here to prevent cyclical imports warning
        from dsbf.utils.task_utils import validate_task_result

        # Inject task input and context
        task.set_input(self.data)
        task.context = self
        task.run()
        result = task.get_output()

        if result is None:
            raise RuntimeError(f"Task '{task.name}' did not produce a TaskResult.")

        # Validate result before storing it
        if not validate_task_result(result):
            msg = f"[{task.name}] TaskResult validation failed"
            self._log(msg, "warn")

        self.set_result(task.name, result)
        return result

    def compute_reliability_flags(self, df: pd.DataFrame | pl.DataFrame) -> None:
        if self.reliability_flags:
            return

        if is_polars(df):
            df = cast(pl.DataFrame, df).to_pandas()
        df = cast(pd.DataFrame, df)

        self.reliability_flags = compute_flags(df)
