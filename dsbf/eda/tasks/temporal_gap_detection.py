# dsbf/eda/tasks/temporal_gap_detection.py

from typing import TYPE_CHECKING, Any

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

if TYPE_CHECKING:
    from pandas import Series

# ── Gap analysis ──────────────────────────────────────────────────────────────


def _analyse_gaps(series: pd.Series) -> dict[str, Any] | None:
    """
    Compute gap statistics for a sorted datetime Series.

    Args:
        series: Non-null datetime Series (will be sorted internally).

    Returns:
        Dict with gap statistics, or None if fewer than 2 observations.

    """
    sorted_s: Series = series.sort_values().reset_index(drop=True)
    if len(sorted_s) < 2:
        return None

    gaps: Series[float] = sorted_s.diff().dropna()
    gaps_days = gaps.dt.total_seconds() / 86_400

    # Detect the dominant (most common) gap — the expected interval
    # Use mode of rounded gap in days; fall back to median
    try:
        dominant_gap_days = float(gaps_days.round(0).mode().iloc[0])
    except Exception:
        dominant_gap_days = float(gaps_days.median())

    # A gap is "large" if it is more than 2x the dominant gap
    large_threshold: float = max(dominant_gap_days * 2, 1.0)
    large_gaps = gaps_days[gaps_days > large_threshold]

    large_gap_details: list[dict[str, float | str]] = []
    for idx in large_gaps.index[:10]:  # cap at 10 examples
        before = sorted_s.iloc[idx - 1]
        after = sorted_s.iloc[idx]
        large_gap_details.append(
            {
                "before": str(before)[:10],
                "after": str(after)[:10],
                "gap_days": round(float(gaps_days.iloc[idx - 1]), 1),
            },
        )

    return {
        "n_observations": len(sorted_s),
        "date_min": str(sorted_s.iloc[0])[:10],
        "date_max": str(sorted_s.iloc[-1])[:10],
        "total_range_days": round(float(gaps_days.sum()), 1),
        "dominant_gap_days": round(dominant_gap_days, 1),
        "median_gap_days": round(float(gaps_days.median()), 1),
        "mean_gap_days": round(float(gaps_days.mean()), 2),
        "max_gap_days": round(float(gaps_days.max()), 1),
        "min_gap_days": round(float(gaps_days.min()), 1),
        "large_gap_count": len(large_gaps),
        "large_gap_threshold_days": round(large_threshold, 1),
        "large_gap_details": large_gap_details,
    }


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="temporal_gap_detection",
    display_name="Temporal Gap Detection",
    description=(
        "Identifies missing time steps and irregular gaps in datetime columns. "
        "Flags periods where the interval between consecutive timestamps "
        "substantially exceeds the dominant (most common) gap."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["datetime", "gaps", "time_series", "quality"],
    expected_semantic_types=["datetime"],
)
class TemporalGapDetection(BaseTask):
    """
    Detect missing time steps and irregular gaps in datetime columns.

    For each datetime column, sorts the timestamps, computes inter-observation
    gaps, and identifies the dominant gap (the most common interval). Any gap
    more than 2x the dominant is flagged as a large gap.

    **Why this matters:**
    Gaps in time series data cause ACF/PACF and decomposition analyses to
    produce misleading results. Forward-fill imputation silently propagates
    stale values across gaps. Knowing where gaps occur — and how large they
    are — is prerequisite for any temporal analysis.

    **Dominant gap detection:**
    The dominant gap is the mode of rounded gap lengths in days. For datasets
    with a clear regular frequency (daily, weekly, monthly), this identifies
    the expected interval. For irregular event data, the median is used as
    fallback.

    **Large gap threshold:**
    A gap is flagged as large if it exceeds ``max(2 x dominant_gap, 1 day)``.
    For daily data, this flags any gap of ≥ 2 days; for weekly data, gaps
    ≥ 2 weeks; etc.

    EDA guidance is emitted for columns with large gaps, describing the gap
    pattern and its implications for downstream time series analysis.

    Configurable parameters (via config["tasks"]["temporal_gap_detection"]):
        min_n (int): Minimum non-null observations to analyse. Default: 10
    """

    def run(self) -> None:
        """
        Execute temporal gap detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'datetime' column(s)",
                "debug",
            )

            min_n_raw: Any | None = self.get_task_param("min_n")
            min_n: int = int(min_n_raw) if min_n_raw is not None else 10

            semantic_types: dict[str, str] = {}
            if self.context:
                semantic_types = self.context.get_metadata("semantic_types") or {}

            datetime_cols: list[str] = [
                col
                for col in df.columns
                if semantic_types.get(col, "") == "datetime"
                or pd.api.types.is_datetime64_any_dtype(df[col])
            ]

            if not datetime_cols:
                self._log("    No datetime columns found.", "debug")
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    summary={
                        "message": "No datetime columns found.",
                        "columns_analysed": 0,
                    },
                    data={},
                    metadata={
                        "suggested_viz_type": "scatter",
                        "recommended_section": "Distributions",
                        "display_priority": "medium",
                        "excluded_columns": excluded,
                        "column_types": self.get_column_type_info(
                            matched_cols + list(excluded.keys()),
                        ),
                    },
                )
                return

            results: dict[str, dict[str, Any]] = {}

            for col in datetime_cols:
                series = pd.to_datetime(df[col], errors="coerce").dropna()

                if len(series) < min_n:
                    self._log(
                        f"    '{col}' skipped: only {len(series)} non-null values.",
                        "debug",
                    )
                    continue

                gap_stats: dict[str, Any] | None = _analyse_gaps(series)
                if gap_stats is None:
                    continue

                results[col] = gap_stats
                self._log(
                    f"    '{col}': dominant gap={gap_stats['dominant_gap_days']}d, "
                    f"{gap_stats['large_gap_count']} large gap(s).",
                    "debug",
                )

            cols_with_gaps: int = sum(
                1 for v in results.values() if v["large_gap_count"] > 0
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Temporal gap analysis on {len(results)} column(s); "
                        f"{cols_with_gaps} have large gap(s)."
                    ),
                    "columns_analysed": len(results),
                    "columns_with_gaps": cols_with_gaps,
                },
                data=results,
                metadata={
                    "min_n": min_n,
                    "suggested_viz_type": "scatter",
                    "recommended_section": "Distributions",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, col_data in results.items():
                if col_data["large_gap_count"] > 0:
                    self._attach_guidance(col, col_data)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, col_data: dict[str, Any]) -> None:
        """
        Generate EDA guidance for a datetime column with large gaps.

        Args:
            col: Column name.
            col_data: Gap analysis result dict.

        """
        n_gaps = col_data["large_gap_count"]
        threshold = col_data["large_gap_threshold_days"]
        dominant = col_data["dominant_gap_days"]
        max_gap = col_data["max_gap_days"]
        date_range: str = f"{col_data['date_min']} to {col_data['date_max']}"
        n = col_data["n_observations"]

        examples = col_data["large_gap_details"][:3]
        example_str: str = "; ".join(
            f"{g['before']} → {g['after']} ({g['gap_days']:.0f}d)" for g in examples
        )

        body: str = (
            f"'{col}' has {n_gaps} gap(s) exceeding {threshold:.0f} days "
            f"(2x the dominant interval of {dominant:.0f} day(s)) "
            f"across {n:,} observations from {date_range}. "
            f"Largest gap: {max_gap:.0f} day(s). "
            f"Examples: {example_str}. "
            f"Gaps in sequential data cause ACF/PACF autocorrelation estimates "
            f"to be misleading, forward-fill imputation to silently propagate "
            f"stale values, and rolling window aggregations to span across "
            f"structurally different periods. Investigate whether gaps represent "
            f"genuine absence of events, data collection failures, or expected "
            f"calendar breaks (weekends, holidays)."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="warn" if n_gaps >= 5 else "info",
            title=(
                f"Temporal Gaps: {n_gaps} gap(s) > "
                f"{threshold:.0f}d (dominant={dominant:.0f}d)"
            ),
            body=body.strip(),
            actions=[
                {
                    "action": "investigate_gaps",
                    "column": col,
                    "detail": (
                        f"Check whether {n_gaps} gap(s) > {threshold:.0f}d "
                        "are data collection failures or expected breaks"
                    ),
                },
                {
                    "action": "resample_or_interpolate",
                    "column": col,
                    "detail": (
                        "Resample to regular frequency or interpolate across "
                        "gaps before temporal analysis"
                    ),
                },
            ],
            metric={
                "large_gap_count": n_gaps,
                "dominant_gap_days": dominant,
                "max_gap_days": max_gap,
                "large_gap_threshold_days": threshold,
                "n_observations": n,
            },
        )
