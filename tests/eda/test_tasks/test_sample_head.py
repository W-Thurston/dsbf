# tests/test_tasks/test_sample_head.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.sample_head import SampleHead


def test_sample_head_expected_output():
    df = pd.DataFrame({"a": list(range(10))})

    task = SampleHead(n=3)
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert len(result.data["sample"]["a"]) == 3
    assert result.data["sample"]["a"] == [0, 1, 2]
