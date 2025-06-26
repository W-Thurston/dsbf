# tests/test_tasks/test_infer_types.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.infer_types import InferTypes


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

    task = InferTypes()
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert result.data["numeric"]["tag"] == "numeric"
    assert result.data["binary"]["tag"] == "binary"
    assert result.data["bool"]["tag"] == "boolean"
    assert result.data["datetime_str"]["tag"] in ("likely_datetime_string", "datetime")
    assert result.data["text"]["tag"] == "string"
