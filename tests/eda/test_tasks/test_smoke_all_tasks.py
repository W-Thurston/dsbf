# tests/test_tasks/test_smoke_all_tasks.py

import importlib
import inspect

import polars as pl

from dsbf.eda.task_result import TaskResult

# List of task module paths (without .py)
task_modules = [
    "dsbf.eda.tasks.compute_entropy",
    "dsbf.eda.tasks.detect_constant_columns",
    "dsbf.eda.tasks.compute_correlations",
    "dsbf.eda.tasks.detect_duplicates",
    "dsbf.eda.tasks.detect_id_columns",
    "dsbf.eda.tasks.detect_high_cardinality",
    "dsbf.eda.tasks.detect_skewness",
    "dsbf.eda.tasks.missingness_heatmap",
    "dsbf.eda.tasks.infer_types",
    "dsbf.eda.tasks.sample_head",
]

# Simple test dataframe
df = pl.DataFrame(
    {
        "a": [1, 2, 3, 4, 5],
        "b": [5, 4, 3, 2, 1],
        "c": ["x", "y", "z", "x", "y"],
        "d": [None, 2, 3, None, 5],
    }
)


def test_all_tasks_smoke():
    for module_path in task_modules:
        mod = importlib.import_module(module_path)
        task_fns = [f for name, f in inspect.getmembers(mod, inspect.isfunction)]
        for fn in task_fns:
            result = fn(df)
            assert isinstance(
                result, TaskResult
            ), f"{fn.__name__} did not return TaskResult"
            assert result.status in (
                "success",
                "skipped",
            ), f"{fn.__name__} returned status {result.status}"
