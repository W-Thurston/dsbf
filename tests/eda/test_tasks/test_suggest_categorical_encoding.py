# tests/eda/test_tasks/test_suggest_categorical_encoding.py

from typing import TYPE_CHECKING

import pandas as pd

from dsbf.eda.tasks.suggest_categorical_encoding import SuggestCategoricalEncoding
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_low_cardinality_gets_one_hot(tmp_path) -> None:
    """Columns with ≤ 10 unique values must receive one-hot encoding suggestion."""
    df = pd.DataFrame({"color": ["red", "blue", "green", "red", "blue"] * 4})

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestCategoricalEncoding,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SuggestCategoricalEncoding)

    assert result.status == "success"
    suggestions = result.data["encoding_suggestions"]
    assert "color" in suggestions
    assert suggestions["color"]["suggested_encoding"] == "one-hot"


def test_high_cardinality_gets_frequency_encoding(tmp_path) -> None:
    """Columns with > 50 unique values must receive frequency (high-cardinality)."""
    df = pd.DataFrame({"city": [f"city_{i}" for i in range(100)]})

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestCategoricalEncoding,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    # Inject semantic type - 100% unique strings classified as 'id' by infer_types
    ctx.set_metadata("semantic_types", {"city": "categorical"})

    task = SuggestCategoricalEncoding()
    task.set_input(df)
    task.context = ctx
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    suggestions = result.data["encoding_suggestions"]
    assert "city" in suggestions
    assert "high-cardinality" in suggestions["city"]["suggested_encoding"]


def test_guidance_attached_for_all_columns(tmp_path) -> None:
    """EDA and ML guidance must be attached for each suggested column."""
    df = pd.DataFrame({"flag": ["A", "B", "A", "C"] * 5})

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestCategoricalEncoding,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SuggestCategoricalEncoding)

    assert result.status == "success"
    assert result.guidance is not None
    assert "flag" in result.guidance
    assert len(result.guidance["flag"]["eda"]) > 0
    assert len(result.guidance["flag"]["ml"]) > 0


def test_no_plots_generated(tmp_path) -> None:
    """Encoding suggestion task must not generate plots."""
    df = pd.DataFrame({"x": ["A", "B", "A"] * 5})

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestCategoricalEncoding,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SuggestCategoricalEncoding)

    assert result.status == "success"
    assert result.plots is None
