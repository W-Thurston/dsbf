# dsbf/core/context.py

import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import median_abs_deviation, skew

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars
from dsbf.utils.logging_utils import DSBFLogger, setup_logger

if TYPE_CHECKING:
    from dsbf.core.base_task import BaseTask


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
        self.reliability_flags: Dict[str, Any] = {}

    def _log(self, msg: str, level: str = "info") -> None:
        """
        Structured logging for context operations using DSBF verbosity levels.
        """

        verbosity = self.config.get("metadata", {}).get("message_verbosity", "info")
        logger: DSBFLogger = setup_logger("dsbf.context", verbosity)

        level_map = {
            "warn": logger.warning,
            "stage": logger.stage,
            "info": logger.info2,
            "debug": logger.debug,
        }
        log_fn = level_map.get(level, logger.info2)
        log_fn(msg)

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
        from dsbf.utils.task_utils import validate_task_result

        task.set_input(self.data)
        task.context = self
        task.run()
        result = task.get_output()

        if result is None:
            raise RuntimeError(f"Task '{task.name}' did not produce a TaskResult.")

        # Validate result before storing it
        if not validate_task_result(result):
            self._log(f"[{task.name}] ⚠️ TaskResult validation failed", "debug")

        self.set_result(task.name, result)
        return result

    def compute_reliability_flags(self, df: pd.DataFrame | pl.DataFrame) -> None:
        if self.reliability_flags:  # Already computed
            return

        if is_polars(df):
            df = cast(pl.DataFrame, df).to_pandas()

        df = cast(pd.DataFrame, df)
        numeric_df = df.select_dtypes(include=np.number).dropna()

        n_rows = len(numeric_df)
        stds = numeric_df.std().to_dict()
        means = numeric_df.mean().to_dict()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            skew_vals = dict(
                zip(numeric_df.columns, skew(numeric_df, nan_policy="omit", bias=False))
            )

        # Robust outlier detection using median and MAD
        mad = {
            col: median_abs_deviation(numeric_df[col], nan_policy="omit")
            for col in numeric_df.columns
        }
        mad = {k: (v if v != 0 else 1e-8) for k, v in mad.items()}
        robust_z = pd.DataFrame(
            {
                col: (numeric_df[col] - numeric_df[col].median()) / mad[col]
                for col in numeric_df.columns
            }
        )
        has_outliers = (robust_z.abs() > 3).any().any()

        low_var_cols = [
            col for col, std in stds.items() if std is not None and std < 1e-8
        ]

        self.reliability_flags = {
            "n_rows": n_rows,
            "low_row_count": n_rows < 30,
            "extreme_outliers": has_outliers,
            "high_skew": any(abs(s) > 2 for s in skew_vals.values() if s is not None),
            "zero_variance_cols": low_var_cols,
            "skew_vals": skew_vals,
            "stds": stds,
            "means": means,
        }
