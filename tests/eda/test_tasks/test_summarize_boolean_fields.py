# tests/eda/test_tasks/test_summarize_boolean_fields.py

from pathlib import Path

import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_boolean_fields import SummarizeBooleanFields
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_boolean_fields_expected_output(tmp_path):
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
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "flag_1" in result.data
    assert "pct_true" in result.data["flag_1"]
    assert result.data["flag_1"]["pct_null"] > 0
    assert "nonbool" not in result.data


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_summarize_boolean_fields_with_plots(tmp_path):
    df = pd.DataFrame(
        {
            "flag_1": [True, False, True, True, False, None],
            "flag_2": [False, False, False, False, False, False],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeBooleanFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is not None
    assert "flag_1" in result.plots
    assert "flag_2" in result.plots

    # Static file exists
    static_path: Path = result.plots["flag_1"]["static"]
    assert static_path.exists()
    static_path.unlink()

    # Interactive plot contains annotated percentages
    annotations = result.plots["flag_1"]["interactive"].get("annotations", [])
    assert any("True:" in ann or "False:" in ann for ann in annotations)
