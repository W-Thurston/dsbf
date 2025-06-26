# dsbf/eda/tasks/missingness_heatmap.py

import os
from typing import Any

import matplotlib.pyplot as plt

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def missingness_heatmap(df: Any, output_dir: str = "outputs") -> TaskResult:
    """
    Generates a missingness heatmap and saves it to disk.

    Args:
        df (DataFrame): Input Polars or Pandas DataFrame.
        output_dir (str): Directory where the image will be saved.

    Returns:
        TaskResult: File path to saved heatmap or error message.
    """
    try:
        import missingno as msno

        if is_polars(df):
            df = df.to_pandas()

        fig_path = os.path.join(output_dir, "missingness_heatmap.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)

        msno.heatmap(df)
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()

        return TaskResult(
            name="missingness_heatmap",
            status="success",
            summary="Saved missingness heatmap to disk.",
            data={"image_path": fig_path},
        )

    except Exception as e:
        return TaskResult(
            name="missingness_heatmap",
            status="failed",
            summary=f"missingness_heatmap failed: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
