# tests/eda/test_tasks/test_compute_entropy.py

from pathlib import Path

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.compute_entropy import ComputeEntropy
from tests.helpers.context_utils import make_ctx_and_task


def test_compute_entropy_expected_output(tmp_path):
    """
    Test that ComputeEntropy returns entropy values for text columns.
    """
    df = pd.DataFrame(
        {
            "cat": ["a", "a", "b", "b", "b", "c", "c", "c", "c"],
            "num": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Should be ignored
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeEntropy,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result is not None, "No TaskResult returned"
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "cat" in result.data
    assert result.data["cat"] > 0.0


def test_compute_entropy_with_plots(tmp_path):
    df = pd.DataFrame(
        {
            "color": [
                "red",
                "red",
                "blue",
                "blue",
                "blue",
                "green",
                "green",
                "green",
                "green",
            ],
            "constant": ["x"] * 9,
            "numeric": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Should be skipped
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeEntropy,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.plots is not None

    for col in ["color", "constant"]:
        assert col in result.plots
        plot_entry = result.plots[col]

        # Validate static plot file
        static_path = plot_entry["static"]
        assert isinstance(static_path, Path)
        assert static_path.exists()
        static_path.unlink()

        # Validate interactive plot + annotations
        interactive = plot_entry["interactive"]
        assert isinstance(interactive, dict)
        assert "annotations" in interactive
        assert any("Entropy" in a for a in interactive["annotations"])
