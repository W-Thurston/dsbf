# tests/eda/test_tasks/test_infer_types.py

import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.infer_types import InferTypes
from tests.helpers.context_utils import make_ctx_and_task


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_infer_types_expanded_output():
    df = pd.DataFrame(
        {
            "numeric": [1, 2, 3],
            "binary": [0, 1, 0],
            "bool": [True, False, True],
            "datetime_str": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "text": ["a", "b", "c"],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=InferTypes,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert result.data["numeric"]["inferred_dtype"] in ("int64", "float64")
    assert result.data["numeric"]["analysis_intent_dtype"] == "continuous"
    assert result.data["binary"]["analysis_intent_dtype"] == "categorical"
    assert result.data["bool"]["analysis_intent_dtype"] == "categorical"
    assert result.data["datetime_str"]["analysis_intent_dtype"] == "datetime"
    assert result.data["text"]["analysis_intent_dtype"] in ("id", "text", "categorical")
