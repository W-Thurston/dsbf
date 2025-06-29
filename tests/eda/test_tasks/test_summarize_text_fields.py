# tests/test_tasks/test_summarize_text_fields.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_text_fields import SummarizeTextFields
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_text_fields_expected_output():
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
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "comments" in result.data
    assert "codes" in result.data
    assert "numeric" not in result.data


def test_summarize_text_fields_diverse_cases():
    df = pd.DataFrame(
        {
            "weird_text": ["👍", "", "🚀" * 50, None],
            "numeric": [1, 2, 3, 4],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeTextFields,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert "weird_text" in result.data
