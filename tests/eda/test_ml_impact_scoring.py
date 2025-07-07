# tests/eda/test_ml_impact_scoring.py

import polars as pl
import pytest

from dsbf.config import load_default_config
from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.utils.task_utils import instantiate_task


# Sample small DataFrame for testing
@pytest.fixture
def test_data():
    return pl.DataFrame(
        {
            "num": [1, 2, 2, 100, 5],
            "cat": ["a", "b", "c", "d", "e"],
            "bool": [True, False, True, False, True],
            "target": [0, 1, 0, 1, 1],
            "high_card": [f"id_{i}" for i in range(5)],
            "constant": ["x"] * 5,
        }
    )


# Tasks expected to emit ML impact signals
ML_TASKS = [
    "detect_skewness",
    "detect_encoded_columns",
    "detect_high_cardinality",
    "detect_constant_columns",
    "detect_mixed_type_columns",
    "detect_class_imbalance",
    "detect_collinear_features",
    "detect_near_zero_variance",
    "detect_data_leakage",
    "detect_outliers",
    "detect_feature_drift",
    "detect_target_drift",
    "suggest_numerical_binning",
    "suggest_categorical_encoding",
]


@pytest.mark.parametrize("task_name", ML_TASKS)
@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_ml_impact_scoring_fields(task_name, test_data, tmp_path):
    config = load_default_config()

    # Inject needed config if missing
    config.setdefault("tasks", {})
    config["tasks"].setdefault(task_name, {})
    config["tasks"][task_name]["target_column"] = "target"
    config["tasks"][task_name]["target"] = "target"

    ctx = AnalysisContext(
        data=test_data,
        config=config,
        reference_data=test_data,
        output_dir=str(tmp_path),
    )

    task = instantiate_task(task_name, config["tasks"].get(task_name, {}))
    task.context = ctx
    task.set_input(test_data)

    try:
        task.run()
        result = task.get_output()

        assert isinstance(
            result, TaskResult
        ), f"{task_name} did not return a TaskResult."

        # If scoring is emitted, check its structure
        if result.ml_impact_score is not None:
            assert (
                0.0 <= result.ml_impact_score <= 1.0
            ), f"{task_name} score out of range"
            assert isinstance(
                result.recommendation_tags, list
            ), f"{task_name} tags missing or invalid"
            assert (
                isinstance(result.recommendations, list) and result.recommendations
            ), f"{task_name} recommendations missing"

    except Exception as e:
        pytest.fail(f"{task_name} raised an error: {e}")
