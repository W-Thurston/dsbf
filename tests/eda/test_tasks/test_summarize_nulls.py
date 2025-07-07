# tests/eda/test_tasks/test_summarize_nulls.py

from pathlib import Path

import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_nulls import SummarizeNulls
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_nulls_expanded_output(tmp_path):
    df = pd.DataFrame(
        {"a": [None, 1, None, 1], "b": [1, None, None, 1], "c": [1, 1, 1, 1]}
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeNulls,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert "null_patterns" in result.data
    assert isinstance(result.data["null_patterns"], dict)
    assert "high_null_columns" in result.data
    assert any(col in ["a", "b"] for col in result.data["high_null_columns"])


def test_summarize_nulls_all_null_column(tmp_path):
    df = pd.DataFrame({"a": [None, None, None]})

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeNulls,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert "a" in result.data.get("high_null_columns", [])


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_summarize_nulls_with_plots(tmp_path):
    df = pd.DataFrame(
        {
            "city": ["NY", None, "SF", "NY"],
            "state": [None, None, "CA", None],
            "zip": [10001, 10002, None, 10003],
            "notes": [None, None, None, None],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeNulls,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is not None
    assert "null_counts" in result.plots

    # Check static plot
    static_path: Path = result.plots["null_counts"]["static"]
    assert static_path.exists()
    static_path.unlink()

    # Check annotations
    interactive = result.plots["null_counts"]["interactive"]
    assert isinstance(interactive, dict)
    assert "annotations" in interactive
    assert any("fully null" in ann.lower() for ann in interactive["annotations"])
