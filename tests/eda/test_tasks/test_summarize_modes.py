# tests/eda/test_tasks/test_summarize_modes.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_modes import SummarizeModes
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_modes_expected_output(tmp_path):
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "y", "x", "z"]})

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeModes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    # Accept either scalar or multimodal list
    a_mode = result.data["a"]
    assert a_mode == 1 or (isinstance(a_mode, list) and 1 in a_mode)

    b_mode = result.data["b"]
    assert b_mode == "x" or (isinstance(b_mode, list) and "x" in b_mode)


def test_summarize_modes_multimodal_case(tmp_path):
    df = pd.DataFrame({"x": [1, 1, 2, 2, 3]})

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeModes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert isinstance(result.data["x"], list)
    assert set(result.data["x"]) >= {1, 2}


def test_summarize_modes_with_plots(tmp_path):
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "red", "green", "red"],
            "shape": ["circle", "square", "circle", "triangle", "circle"],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeModes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is not None
    assert "color" in result.plots

    # Check static plot
    static_path = result.plots["color"]["static"]
    assert static_path.exists()
    static_path.unlink()

    # Check annotation
    interactive = result.plots["color"]["interactive"]
    assert any("Most frequent" in a for a in interactive["annotations"])


def test_summarize_modes_numeric_column_only(tmp_path):
    df = pd.DataFrame({"value": [1, 2, 2, 3, 3]})
    ctx, task = make_ctx_and_task(
        task_cls=SummarizeModes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)
    assert result.data is not None
    assert "value" in result.data
    assert result.plots == {}


def test_summarize_modes_constant_column(tmp_path):
    df = pd.DataFrame({"x": ["same"] * 5})
    ctx, task = make_ctx_and_task(
        task_cls=SummarizeModes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)
    assert result.data is not None
    assert "x" in result.data
    assert result.plots == {}


def test_summarize_modes_empty_df(tmp_path):
    df = pd.DataFrame()
    ctx, task = make_ctx_and_task(
        task_cls=SummarizeModes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)
    assert result.status == "success"
    assert result.data == {}
    assert result.plots == {}
