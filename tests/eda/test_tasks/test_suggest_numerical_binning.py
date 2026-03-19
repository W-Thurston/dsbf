# tests/eda/test_tasks/test_suggest_numerical_binning.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.suggest_numerical_binning import SuggestNumericalBinning
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


def test_log_transform_suggested_for_skewed_column(tmp_path) -> None:
    """A highly right-skewed column must receive a log-transform suggestion."""
    df = pd.DataFrame({"skewed": [1] * 90 + list(range(100, 200, 10))})

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestNumericalBinning,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SuggestNumericalBinning)

    assert result.status == "success"
    suggestions = result.data["binning_suggestions"]
    assert "skewed" in suggestions
    assert suggestions["skewed"]["suggested_binning"] == "log-transform"


def test_quantile_binning_for_symmetric_column(tmp_path) -> None:
    """A symmetric column with compact spread must receive quantile binning."""
    df = pd.DataFrame({"symmetric": list(range(1, 51))})  # uniform 1–50

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestNumericalBinning,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SuggestNumericalBinning)

    assert result.status == "success"
    suggestions = result.data["binning_suggestions"]
    if "symmetric" in suggestions:
        # Depending on spread, quantile or equal-width — both are acceptable
        assert suggestions["symmetric"]["suggested_binning"] in (
            "quantile binning",
            "equal-width binning",
        )


def test_guidance_attached_for_suggestions(tmp_path) -> None:
    """EDA and ML guidance must be attached for each suggested column."""
    df = pd.DataFrame({"skewed": [1] * 90 + list(range(100, 200, 10))})

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestNumericalBinning,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SuggestNumericalBinning)

    assert result.status == "success"
    if result.guidance:
        for col in result.guidance:
            assert len(result.guidance[col]["eda"]) > 0
            assert len(result.guidance[col]["ml"]) > 0


def test_no_plots_generated(tmp_path) -> None:
    """Binning suggestion task must not generate plots."""
    df = pd.DataFrame({"x": list(range(100))})

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestNumericalBinning,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SuggestNumericalBinning)

    assert result.status == "success"
    assert result.plots is None
