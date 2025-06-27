# tests/test_tasks/test_summarize_value_counts.py

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_value_counts import SummarizeValueCounts


def test_summarize_value_counts_expected_output():
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a", "a", "c", "b", "c", "c", "c"],
            "num": [1, 2, 1, 3, 2, 2, 1, 3, 3],
            "misc": [None, None, "x", "x", "x", "y", "y", "y", "y"],
        }
    )

    context = AnalysisContext(df)
    result = context.run_task(SummarizeValueCounts(config={"top_k": 2}))

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "cat" in result.data
    assert isinstance(result.data["cat"], dict)
    assert len(result.data["cat"]) <= 2
    assert "a" in result.data["cat"] or "c" in result.data["cat"]
