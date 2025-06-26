# tests/test_tasks/test_sample_tail.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.sample_tail import SampleTail


def test_sample_tail_expected_output():
    df = pd.DataFrame({"a": list(range(10))})

    task = SampleTail(n=3)
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert len(result.data["sample"]["a"]) == 3
    assert result.data["sample"]["a"] == [7, 8, 9]
