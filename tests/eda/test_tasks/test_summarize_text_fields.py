# tests/test_tasks/test_summarize_text_fields.py

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_text_fields import SummarizeTextFields


def test_summarize_text_fields_expected_output():
    df = pd.DataFrame(
        {
            "comments": ["Great job!", "Well done", "Needs improvement", None],
            "codes": ["A12", "B34", "C56", "D78"],
            "numeric": [1, 2, 3, 4],
        }
    )

    context = AnalysisContext(df)
    result = context.run_task(SummarizeTextFields())

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "comments" in result.data
    assert "codes" in result.data
    assert "numeric" not in result.data
