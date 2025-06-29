# tests/test_tasks/test_summarize_boolean_fields.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_boolean_fields import SummarizeBooleanFields
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_boolean_fields_expected_output():
    df = pd.DataFrame(
        {
            "flag_1": [True, False, True, True, False, None],
            "flag_2": [False, False, False, False, False, False],
            "nonbool": [1, 2, 3, 4, 5, 6],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeBooleanFields,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "flag_1" in result.data
    assert "pct_true" in result.data["flag_1"]
    assert result.data["flag_1"]["pct_null"] > 0
    assert "nonbool" not in result.data
