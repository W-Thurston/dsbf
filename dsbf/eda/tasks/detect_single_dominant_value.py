# dsbf/eda/tasks/detect_single_dominant_value.py

from typing import Any, Dict

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def detect_single_dominant_value(
    df: Any, dominance_threshold: float = 0.95
) -> TaskResult:
    """
    Detects columns where a single value dominates the distribution.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.
        dominance_threshold (float): Proportion above which a single value is
            considered dominant.

    Returns:
        TaskResult: Flagged columns and their dominant value stats.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        results: Dict[str, Dict[str, Any]] = {}
        for col in df.columns:
            vc = df[col].value_counts(dropna=False, normalize=True)
            if not vc.empty and vc.iloc[0] >= dominance_threshold:
                results[col] = {
                    "dominant_value": vc.index[0],
                    "proportion": float(vc.iloc[0]),
                }

        return TaskResult(
            name="detect_single_dominant_value",
            status="success",
            summary=f"Detected {len(results)} column(s) with dominant values.",
            data=results,
            metadata={"dominance_threshold": dominance_threshold},
        )

    except Exception as e:
        return TaskResult(
            name="detect_single_dominant_value",
            status="failed",
            summary=f"Error during dominant value detection: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
