# tests/eda/test_tasks/test_summarize_text_fields.py

from pathlib import Path

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_text_fields import SummarizeTextFields
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_text_fields_expected_output(tmp_path):
    df = pd.DataFrame(
        {
            "comments": ["Great job!", "Well done", "Needs improvement", None],
            "codes": ["A12", "B34", "C56", "D78"],
            "numeric": [1, 2, 3, 4],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeTextFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "comments" in result.data
    assert "codes" in result.data
    assert "numeric" not in result.data


def test_summarize_text_fields_diverse_cases(tmp_path):
    df = pd.DataFrame(
        {
            "weird_text": ["👍", "", "🚀" * 50, None],
            "numeric": [1, 2, 3, 4],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeTextFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert "weird_text" in result.data


def test_summarize_text_fields_with_plots(tmp_path):
    df = pd.DataFrame(
        {
            "feedback": ["Good", "Excellent service", "Bad", "Okay", "Could be better"],
            "other": [1, 2, 3, 4, 5],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeTextFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.plots is not None
    assert "feedback" in result.plots

    static = result.plots["feedback"]["static"]
    assert isinstance(static, Path)
    assert static.exists()
    static.unlink()

    interactive = result.plots["feedback"]["interactive"]
    assert isinstance(interactive, dict)
    assert "annotations" in interactive
    assert any("Avg" in a for a in interactive["annotations"])


def test_summarize_text_fields_all_null(tmp_path):
    df = pd.DataFrame({"col": [None, None, None]})
    ctx, task = make_ctx_and_task(
        task_cls=SummarizeTextFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots == {}
