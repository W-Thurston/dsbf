# dsbf/utils/task_utils.py

from typing import Any, Dict, Optional

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_loader import load_all_tasks
from dsbf.eda.task_registry import TASK_REGISTRY

load_all_tasks()


def instantiate_task(
    task_name: str,
    task_specific_cfg: Optional[Dict[str, Any]] = None,
) -> BaseTask:
    """
    Look up a task by name and return an instantiated task with a validated config.
    If no config is provided but the task defines a schema, the default config is used.

    Args:
        task_name (str): The registered name of the task.
        task_specific_cfg (Optional[dict]): Overrides from config["tasks"][task_name].

    Returns:
        BaseTask: An instance of the task, with config applied.
    """
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
