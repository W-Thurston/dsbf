# dsbf/eda/tasks/_time_series_utils.py
#
# Shared utilities for all time series tasks.
#
# Every time series task must:
#   1. Check that time_series is enabled in config
#   2. Validate that datetime_index_column is configured and present
#   3. Warn if group_by_column is set (not yet supported)
#   4. Sort the DataFrame by the datetime index
#   5. Optionally check for large gaps from temporal_gap_detection context
#
# All of this lives here so the task files stay thin and the logic is tested
# in one place.

from typing import Any

import pandas as pd

from dsbf.utils.backend import is_polars

# ── Public result types ───────────────────────────────────────────────────────


class TimeSeriesSetupError(Exception):
    """
    Raised when time series preconditions are not met.

    Caught by each task's run() method and converted to a graceful no-op
    result with a clear message rather than a hard failure.
    """


class TimeSeriesConfig:
    """
    Validated, parsed time series configuration.

    Attributes:
        enabled: Whether time series analysis is enabled.
        index_col: Name of the datetime index column.
        value_cols: List of continuous columns to analyse. Empty = all.
        frequency: Configured frequency string (e.g. "D", "W"). None = infer.
        group_by_col: Reserved group column (None = not set).

    """

    def __init__(
        self,
        enabled: bool,
        index_col: str | None,
        value_cols: list[str],
        frequency: str | None,
        group_by_col: str | None,
    ) -> None:
        self.enabled = enabled
        self.index_col = index_col
        self.value_cols = value_cols
        self.frequency = frequency
        self.group_by_col = group_by_col


# ── Config reading ────────────────────────────────────────────────────────────


def read_time_series_config(task_instance: Any) -> TimeSeriesConfig:
    enabled = task_instance.get_shared_param("time_series", "enabled", False)
    index_col = (
        task_instance.get_shared_param("time_series", "datetime_index_column") or None
    )
    value_cols = list(
        task_instance.get_shared_param("time_series", "value_columns") or [],
    )
    frequency = task_instance.get_shared_param("time_series", "frequency") or None
    group_by_col = (
        task_instance.get_shared_param("time_series", "group_by_column") or None
    )

    return TimeSeriesConfig(
        enabled=bool(enabled),
        index_col=index_col,
        value_cols=value_cols,
        frequency=frequency,
        group_by_col=group_by_col,
    )


# ── Validation ────────────────────────────────────────────────────────────────


def validate_time_series_config(
    ts_config: TimeSeriesConfig,
    df: pd.DataFrame,
    task_instance: Any,
) -> None:
    """
    Validate the time series configuration against the DataFrame.

    Raises ``TimeSeriesSetupError`` with a clear message for each failure
    condition. Does not raise for the group_by_column warning - instead
    logs a warning and continues.

    Args:
        ts_config: Parsed time series config.
        df: Source DataFrame (pandas).
        task_instance: Task instance for logging.

    Raises:
        TimeSeriesSetupError: If time series is disabled, no index column
            is configured, or the index column is not in the DataFrame.

    """
    if not ts_config.enabled:
        raise TimeSeriesSetupError(
            "Time series analysis is not enabled. "
            "To enable it, set time_series.enabled: true and "
            "time_series.datetime_index_column: <column_name> in your config. "
            "DSBF does not guess which datetime column is the time axis - "
            "the risk of silently analysing the wrong column is too high."
        )

    if not ts_config.index_col:
        raise TimeSeriesSetupError(
            "time_series.datetime_index_column is not set. "
            "Specify the column that represents the time axis. "
            "DSBF will not guess - multiple datetime columns may be present "
            "and selecting the wrong one produces misleading results."
        )

    if ts_config.index_col not in df.columns:
        msg: str = (
            f"Configured datetime_index_column '{ts_config.index_col}' "
            f"is not present in the DataFrame. "
            f"Available columns: {list(df.columns)[:10]}."
        )
        raise TimeSeriesSetupError(msg)

    # group_by_column: reserved but not yet supported - warn, don't fail
    if ts_config.group_by_col:
        task_instance._log(
            f"    time_series.group_by_column='{ts_config.group_by_col}' is set "
            "but grouped/panel time series is not yet supported. "
            "Analysis will proceed on the full dataset. "
            "The group_by_column setting is reserved for a future release.",
            "warn",
        )


# ── DataFrame preparation ─────────────────────────────────────────────────────


def prepare_time_series_df(
    df: pd.DataFrame,
    ts_config: TimeSeriesConfig,
    task_instance: Any,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Sort the DataFrame by the datetime index and select value columns.

    Also validates the index column can be parsed as datetime (attempting
    conversion if needed) and checks for and logs any large gaps detected
    by temporal_gap_detection in context.

    Args:
        df: Source DataFrame (pandas, already converted from Polars if needed).
        ts_config: Validated time series config.
        task_instance: Task instance for logging and context access.

    Returns:
        Tuple of (sorted DataFrame, list of value column names to analyse).

    Raises:
        TimeSeriesSetupError: If the index column cannot be parsed as datetime
            or has fewer than 3 non-null values.

    """
    index_col = ts_config.index_col

    # Ensure datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(df[index_col]):
        try:
            df = df.copy()
            df[index_col] = pd.to_datetime(df[index_col], errors="coerce")
            n_null = int(df[index_col].isna().sum())
            if n_null > 0:
                task_instance._log(
                    f"    '{index_col}': {n_null} value(s) could not be parsed "
                    "as datetime and were set to NaT.",
                    "warn",
                )
        except Exception as e:
            msg: str = f"Could not parse '{index_col}' as datetime: {e}"
            raise TimeSeriesSetupError(msg) from e

    n_valid = int(df[index_col].notna().sum())
    if n_valid < 3:
        msg = (
            f"'{index_col}' has only {n_valid} non-null timestamp(s). "
            "At least 3 are required for time series analysis."
        )
        raise TimeSeriesSetupError(msg)

    # Sort by the index, drop rows with null timestamps
    df = df.dropna(subset=[index_col]).sort_values(index_col).reset_index(drop=True)

    # Select value columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Exclude the index column if it somehow appears in numeric cols
    numeric_cols = [c for c in numeric_cols if c != index_col]

    if ts_config.value_cols:
        # User-specified: validate each is present and numeric
        missing: list[str] = [c for c in ts_config.value_cols if c not in df.columns]
        non_numeric: list[str] = [
            c for c in ts_config.value_cols if c in df.columns and c not in numeric_cols
        ]
        if missing:
            task_instance._log(
                f"    Configured value_columns not in DataFrame: {missing}. "
                "They will be skipped.",
                "warn",
            )
        if non_numeric:
            task_instance._log(
                f"    Configured value_columns are not numeric: {non_numeric}. "
                "They will be skipped.",
                "warn",
            )
        value_cols: list[str] = [
            c for c in ts_config.value_cols if c in df.columns and c in numeric_cols
        ]
    else:
        value_cols = numeric_cols

    if not value_cols:
        raise TimeSeriesSetupError(
            "No numeric value columns available for time series analysis. "
            "Specify value_columns in config or ensure the DataFrame has "
            "numeric columns other than the datetime index.",
        )

    # Log gap warnings from temporal_gap_detection if available
    _log_gap_warnings(df, index_col, ts_config, task_instance)

    return df, value_cols


def _log_gap_warnings(
    df: pd.DataFrame,
    index_col: str,
    ts_config: TimeSeriesConfig,
    task_instance: Any,
) -> None:
    """
    Log a warning if temporal_gap_detection found large gaps in the index column.

    Large gaps cause ACF/PACF and STL results to be misleading - the analysis
    spans structurally different periods. This is logged as a warning so users
    can make an informed decision about whether to proceed.

    Args:
        df: Sorted DataFrame.
        index_col: Datetime index column name.
        ts_config: Time series config.
        task_instance: Task instance for context and logging.
    """
    if not task_instance.context:
        return

    gap_result = task_instance.context.results.get("temporal_gap_detection")
    if gap_result and gap_result.status == "success":
        col_gaps = gap_result.data.get(index_col, {})
        n_large_gaps = col_gaps.get("large_gap_count", 0)
        if n_large_gaps > 0:
            dominant = col_gaps.get("dominant_gap_days", "?")
            threshold = col_gaps.get("large_gap_threshold_days", "?")
            task_instance._log(
                f"    Warning: temporal_gap_detection found {n_large_gaps} "
                f"gap(s) > {threshold}d (dominant interval: {dominant}d) in "
                f"'{index_col}'. ACF/PACF, stationarity, and decomposition "
                f"results may be misleading - gaps cause the analysis to span "
                f"structurally different periods. Consider resampling to a "
                f"regular frequency or interpolating gaps first.",
                "warn",
            )


# ── Frequency inference ───────────────────────────────────────────────────────


def infer_frequency(
    series: pd.Series,
    configured: str | None,
    task_instance: Any,
) -> str | None:
    """
    Infer or validate the time series frequency.

    If a frequency is configured, returns it unchanged. Otherwise attempts
    to infer it from the dominant gap in the sorted datetime series.

    Args:
        series: Sorted datetime Series (the index column values).
        configured: Configured frequency string or None.
        task_instance: Task instance for logging.

    Returns:
        Frequency string (e.g. "D", "W", "ME") or None if inference fails.

    """
    if configured:
        task_instance._log(f"    Using configured frequency: '{configured}'.", "debug")
        return configured

    try:
        inferred: str | None = pd.infer_freq(series)
        if inferred:
            task_instance._log(f"    Inferred frequency: '{inferred}'.", "debug")
            return inferred
    except Exception:
        pass

    # Fallback: derive from median gap
    gaps = series.diff().dropna()
    if gaps.empty:
        return None

    median_days = gaps.dt.total_seconds().median() / 86_400

    # Map common gap ranges to pandas frequency aliases
    freq_map: list[tuple[float, float, str]] = [
        (0.9, 1.1, "D"),
        (6.5, 7.5, "W"),
        (13.5, 14.5, "2W"),
        (28.0, 32.0, "ME"),
        (85.0, 95.0, "QE"),
        (360.0, 370.0, "YE"),
    ]
    for low, high, alias in freq_map:
        if low <= median_days <= high:
            task_instance._log(
                f"    Frequency inferred from median gap ({median_days:.1f}d): "
                f"'{alias}'.",
                "debug",
            )
            return alias

    task_instance._log(
        f"    Could not infer standard frequency (median gap: "
        f"{median_days:.1f}d). Proceeding without frequency alias.",
        "debug",
    )
    return None


# ── Conversion helper ─────────────────────────────────────────────────────────


def to_pandas(df: Any) -> pd.DataFrame:
    """
    Convert a Polars or pandas DataFrame to pandas.

    Args:
        df: Input DataFrame.

    Returns:
        pandas DataFrame.

    """
    if is_polars(df):
        return df.to_pandas()
    return df


# ── No-op result builder ──────────────────────────────────────────────────────


def make_disabled_result(task_name: str, message: str) -> dict:
    """
    Build a minimal success result dict for when TS analysis is disabled/skipped.

    Args:
        task_name: Name of the calling task.
        message: Explanation message for the user.

    Returns:
        Dict suitable for passing to ``TaskResult()``.

    """
    return {
        "summary": {
            "message": message,
            "time_series_enabled": False,
        },
        "data": {},
        "metadata": {
            "suggested_viz_type": None,
            "recommended_section": "Time Series",
            "display_priority": "low",
        },
    }
