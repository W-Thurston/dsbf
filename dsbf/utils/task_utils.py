from typing import Any, Dict, Optional

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_loader import load_all_tasks
from dsbf.eda.task_registry import TASK_REGISTRY

load_all_tasks()


def instantiate_task(
    task_name: str,
    task_specific_cfg: Optional[Dict[str, Any]] = None,
) -> BaseTask:
    spec = TASK_REGISTRY[task_name]
    return spec.cls(name=task_name, config=task_specific_cfg)


def is_integer_polars(series):
    import polars as pl

    return hasattr(series, "dtype") and series.dtype in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }


def is_diagnostic_task(task_name: str) -> bool:
    """
    Identify whether a task is a diagnostic/runtime/system-level task
    rather than a data EDA task.

    Args:
        task_name (str): The name of the task (from context.results)

    Returns:
        bool: True if task is diagnostic, else False
    """
    normalized = task_name.lower().replace("_", "")
    return normalized.startswith("identifybottleneck") or normalized.startswith(
        "logresourceusage"
    )
