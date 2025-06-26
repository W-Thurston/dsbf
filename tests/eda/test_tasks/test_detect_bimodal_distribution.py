# tests/test_tasks/test_detect_bimodal_distribution.py

import numpy as np
import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_bimodal_distribution import detect_bimodal_distribution


def test_detect_bimodal_distribution_expected_output():
    # Generate a bimodal distribution
    x1 = np.random.normal(0, 1, 100)
    x2 = np.random.normal(5, 1, 100)
    df = pd.DataFrame(
        {"bimodal": np.concatenate([x1, x2]), "uniform": np.random.uniform(0, 1, 200)}
    )

    result = detect_bimodal_distribution(df, bic_threshold=5.0)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    flags = result.data["bimodal_flags"]
    assert "bimodal" in flags
