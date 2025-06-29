# tests/test_tasks/test_detect_bimodal_distribution.py

import numpy as np
import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_bimodal_distribution import DetectBimodalDistribution
from tests.helpers.context_utils import make_ctx_and_task


def test_detect_bimodal_distribution_expected_output():
    """
    Test that DetectBimodalDistribution flags a bimodal distribution
    using Gaussian Mixture Models.
    """
    # Create synthetic bimodal and unimodal distributions
    np.random.seed(42)
    x1 = np.random.normal(0, 1, 100)
    x2 = np.random.normal(5, 1, 100)
    df = pd.DataFrame(
        {
            "bimodal": np.concatenate([x1, x2]),
            "uniform": np.random.uniform(0, 1, 200),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectBimodalDistribution,
        current_df=df,
        task_overrides={"bic_threshold": 5.0},
    )
    result = ctx.run_task(task)

    assert result is not None, "No TaskResult returned"
    assert isinstance(result, TaskResult)
    assert result.status == "success"

    assert result.data is not None
    flags = result.data.get("bimodal_flags", {})
    scores = result.data.get("bic_scores", {})

    assert "bimodal" in flags
    assert "uniform" in flags
    assert isinstance(flags["bimodal"], bool)
    assert isinstance(scores["bimodal"]["bic_1_component"], float)
    assert isinstance(scores["bimodal"]["bic_2_components"], float)
