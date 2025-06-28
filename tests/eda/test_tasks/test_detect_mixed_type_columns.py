# test/eda/test_tasks/test_detect_mixed_type_columns.py

import polars as pl

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_mixed_type_columns import DetectMixedTypeColumns


def test_all_same_type_column():
    df = pl.DataFrame({"col1": [1, 2, 3, 4, 5]})
    context = AnalysisContext(data=df, config={})
    task = DetectMixedTypeColumns(name="detect_mixed_type_columns", config={})
    task.set_input(df)
    task.context = context
    task.run()

    result = task.get_output()
    assert result is not None
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.summary["num_mixed_type_columns"] == 0
    assert result.summary["columns"] == []


def test_detects_mixed_type_column():
    df = pl.DataFrame(
        {"col1": pl.Series("col1", [1, 2, "three", 4.0, None], dtype=pl.Object)}
    )
    config = {
        "tasks": {
            "detect_mixed_type_columns": {"min_ratio": 0.1, "ignore_null_type": True}
        }
    }
    context = AnalysisContext(data=df, config=config)
    task = DetectMixedTypeColumns(
        name="detect_mixed_type_columns",
        config=config["tasks"]["detect_mixed_type_columns"],
    )
    task.set_input(df)
    task.context = context
    task.run()

    result = task.get_output()
    assert result is not None
    assert result.status == "success"
    assert result.summary["num_mixed_type_columns"] == 1
    assert "col1" in result.summary["columns"]
    assert result.data is not None
    assert "str" in result.data["col1"]["type_counts"]


def test_ignores_minor_type_if_below_threshold():
    df = pl.DataFrame(
        {"col1": pl.Series("col1", [1] * 98 + ["x"] * 2, dtype=pl.Object)}
    )
    config = {
        "tasks": {
            "detect_mixed_type_columns": {
                "min_ratio": 0.05,  # 2/100 == 2% < 5%
                "ignore_null_type": True,
            }
        }
    }
    context = AnalysisContext(data=df, config=config)
    task = DetectMixedTypeColumns(
        name="detect_mixed_type_columns",
        config=config["tasks"]["detect_mixed_type_columns"],
    )
    task.set_input(df)
    task.context = context
    task.run()

    result = task.get_output()
    assert result is not None
    assert result.status == "success"
    assert result.summary["num_mixed_type_columns"] == 0


def test_detects_mixed_with_nulls_ignored():
    df = pl.DataFrame({"col1": pl.Series("col1", [1, "two", 3, None], dtype=pl.Object)})
    config = {
        "tasks": {
            "detect_mixed_type_columns": {
                "min_ratio": 0.1,
                "ignore_null_type": True,  # Count None as a type
            }
        }
    }
    context = AnalysisContext(data=df, config=config)
    task = DetectMixedTypeColumns(
        name="detect_mixed_type_columns",
        config=config["tasks"]["detect_mixed_type_columns"],
    )
    task.set_input(df)
    task.context = context
    task.run()

    result = task.get_output()
    assert result is not None
    assert result.status == "success"
    assert result.summary["num_mixed_type_columns"] == 1
    assert result.data is not None
    assert "NoneType" not in result.data["col1"]["type_counts"]


def test_skips_non_object_column():
    df = pl.DataFrame({"col1": [1, 2, 3, 4]})
    df = df.with_columns(pl.col("col1").cast(pl.Int64))
    context = AnalysisContext(data=df, config={})
    task = DetectMixedTypeColumns(name="detect_mixed_type_columns", config={})
    task.set_input(df)
    task.context = context
    task.run()
    result = task.get_output()
    assert result is not None
    assert result.status == "success"
    assert result.summary["num_mixed_type_columns"] == 0
