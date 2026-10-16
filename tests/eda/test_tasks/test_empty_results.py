# tests/eda/test_tasks/test_empty_results.py
"""
Tasks that report a fixed set of ``data`` keys must keep those keys when no
column is eligible, so the dashboard and dependent tasks can read them.
"""

from typing import Any

import pandas as pd
import pytest

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import TASK_REGISTRY
from dsbf.eda.task_result import TaskResult
from tests.helpers.context_utils import make_ctx_and_task

pytestmark = pytest.mark.unit

# Registry name -> top-level keys of ``data`` for a result with no eligible columns.
EMPTY_DATA_KEYS: dict[str, set[str]] = {
    "compute_mutual_information": {"mi_scores"},
    "detect_bimodal_distribution": {"bimodal_flags", "bic_scores"},
    "detect_collinear_features": {"vif_scores", "collinear_columns"},
    "detect_constant_columns": {"constant_columns"},
    "detect_data_leakage": {"leakage_pairs"},
    "detect_duplicate_columns": {"duplicate_column_pairs"},
    "detect_duplicates": {"duplicate_count"},
    "detect_near_zero_variance": {"low_variance_columns"},
    "detect_zeros": {"zero_counts", "zero_percentages", "zero_flags"},
    "fuzzy_duplicate_detection": {"__dataset__"},
    "imputation_strategy_suggestions": {"suggestions"},
    "missingness_heatmap": {
        "missing_cells",
        "missing_columns",
        "column_count",
        "row_count",
    },
    "sample_head": {"sample"},
    "sample_tail": {"sample"},
    "suggest_categorical_encoding": {"encoding_suggestions"},
    "suggest_dtype_optimizations": {"suggestions"},
    "suggest_numerical_binning": {"binning_suggestions"},
    "summarize_dataset_shape": {
        "num_rows",
        "num_columns",
        "null_cell_percentage",
        "approx_memory_MB",
        "column_memory_bytes",
        "column_memory_MB",
    },
}


class _Probe(BaseTask):
    def run(self) -> None:
        self.output = self.make_empty_result("nothing to do")


def test_make_empty_result_defaults_to_empty_data() -> None:
    result = _Probe(name="probe").make_empty_result("nothing to do")

    assert result.status == "success"
    assert result.data == {}
    assert result.summary == {"message": "nothing to do"}


def test_make_empty_result_keeps_the_given_data_and_excluded_columns() -> None:
    result = _Probe(name="probe").make_empty_result(
        "nothing to do",
        {"a": "text"},
        data={"flags": {}, "pairs": []},
    )

    assert result.data == {"flags": {}, "pairs": []}
    assert result.metadata["excluded_columns"] == {"a": "text"}


@pytest.mark.parametrize("task_name", sorted(EMPTY_DATA_KEYS))
def test_empty_result_keeps_the_tasks_data_keys(task_name: str, tmp_path) -> None:
    spec = TASK_REGISTRY[task_name]
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    ctx, task = make_ctx_and_task(
        task_cls=spec.cls,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    # No column has a semantic type, so no column is eligible for any task.
    ctx.set_metadata("semantic_types", {})
    task.set_input(df)
    result: TaskResult = ctx.run_task(task)

    data: dict[str, Any] = result.data
    assert result.status == "success"
    assert set(data) == EMPTY_DATA_KEYS[task_name]
