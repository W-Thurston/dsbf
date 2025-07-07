# tests/eda/test_tasks/test_summarize_unique.py

from pathlib import Path

import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_unique import SummarizeUnique
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_unique_expected_output():
    df = pd.DataFrame(
        {"a": [1, 2, 2, 3], "b": ["x", "x", "y", "z"], "c": [True, False, True, True]}
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeUnique,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["a"] == 3
    assert result.data["b"] == 3
    assert result.data["c"] == 2


def test_summarize_unique_empty_column():
    df = pd.DataFrame({"empty": [None, None, None, None]})

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeUnique,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.data["empty"] == 0


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_summarize_unique_with_plots(tmp_path):
    df = pd.DataFrame(
        {
            "city": ["NY", "LA", "SF", "NY"],
            "state": ["NY", "CA", "CA", "NY"],
            "zip": [10001, 90001, 94101, 10001],
            "constant": ["yes"] * 4,
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeUnique,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is not None
    assert "unique_counts" in result.plots

    # Check static plot
    static_path: Path = result.plots["unique_counts"]["static"]
    assert static_path.exists()
    static_path.unlink()

    # Check interactive plot
    interactive = result.plots["unique_counts"]["interactive"]
    assert isinstance(interactive, dict)
    assert "annotations" in interactive
    assert any("constant" in ann.lower() for ann in interactive["annotations"])
