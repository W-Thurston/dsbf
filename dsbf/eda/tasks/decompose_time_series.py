# dsbf/eda/tasks/decompose_time_series.py

from typing import TYPE_CHECKING, Any

import numpy as np
from statsmodels.tsa.seasonal import STL

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.eda.tasks._time_series_utils import (
    TimeSeriesConfig,
    TimeSeriesSetupError,
    infer_frequency,
    make_disabled_result,
    prepare_time_series_df,
    read_time_series_config,
    to_pandas,
    validate_time_series_config,
)

if TYPE_CHECKING:
    from pandas import DataFrame, Series

# Frequency alias → likely seasonal period mapping
# Used when seasonal_period is not configured and cannot be inferred
_FREQ_TO_PERIOD: dict[str, int] = {
    "D": 7,  # daily → weekly seasonality
    "B": 5,  # business daily → weekly
    "W": 52,  # weekly → annual
    "2W": 26,  # biweekly → annual
    "ME": 12,  # monthly → annual
    "MS": 12,
    "QE": 4,  # quarterly → annual
    "QS": 4,
    "YE": 1,  # annual → no meaningful seasonality
    "YS": 1,
}


def _infer_seasonal_period(
    frequency: str | None,
    n: int,
    task_instance: Any,
) -> int | None:
    """
    Infer a reasonable seasonal period from the series frequency.

    Args:
        frequency: Frequency alias (e.g. "D", "ME") or None.
        n: Number of observations.
        task_instance: Task instance for logging.

    Returns:
        Integer seasonal period, or None if not determinable.

    """
    if frequency is None:
        task_instance._log(
            "    No frequency available - cannot infer seasonal period. "
            "Set time_series.frequency or time_series.tasks.stl_decomposition"
            ".seasonal_period in config.",
            "warn",
        )
        return None

    # Normalise aliases (pandas sometimes returns e.g. "ME", sometimes "M")
    for alias, period in _FREQ_TO_PERIOD.items():
        if frequency.upper().startswith(alias):
            if period == 1:
                task_instance._log(
                    f"    Frequency '{frequency}' maps to period=1 - "
                    "STL requires period >= 2. Skipping decomposition.",
                    "warn",
                )
                return None
            # Need at least 2 full cycles
            if n < period * 2:
                task_instance._log(
                    f"    n={n} < 2 x period={period}. Series is too short "
                    f"for reliable seasonal decomposition with "
                    f"frequency='{frequency}'.",
                    "warn",
                )
                return None
            task_instance._log(
                f"    Inferred seasonal period {period} from frequency '{frequency}'.",
                "debug",
            )
            return period

    task_instance._log(
        f"    Unknown frequency alias '{frequency}' - cannot infer seasonal period.",
        "warn",
    )
    return None


@register_task(
    name="decompose_time_series",
    display_name="Time Series Decomposition (STL)",
    description=(
        "Decomposes each value column into trend, seasonal, and residual "
        "components using STL (Seasonal-Trend decomposition using LOESS). "
        "Requires time_series.datetime_index_column to be set in config."
    ),
    depends_on=["infer_types", "temporal_gap_detection"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="moderate",
    tags=["time_series", "decomposition", "stl", "trend", "seasonal"],
    expected_semantic_types=["continuous"],
)
class DecomposeTimeSeries(BaseTask):
    """
    Decompose time series into trend, seasonal, and residual components.

    Uses STL (Seasonal-Trend decomposition using LOESS) which is more robust
    than classical additive/multiplicative decomposition because:

    - It handles any seasonality period, not just monthly/quarterly
    - The LOESS smoother is robust to outliers (``robust=True``)
    - The seasonal component can change over time

    **Output format:**
    Each column result contains ``timestamps``, ``observed``, ``trend``,
    ``seasonal``, and ``residual`` arrays of equal length, ready for direct
    frontend rendering. All arrays are indexed by position matching the
    sorted, gap-filtered DataFrame.

    **Seasonal period:**
    Determined in priority order:
    1. ``time_series.tasks.stl_decomposition.seasonal_period`` in config
    2. Inferred from the series frequency (e.g. daily → 7, monthly → 12)
    3. If neither is available, the column is skipped with a clear message

    **Strength metrics:**
    Trend strength and seasonal strength are computed from the variance
    decomposition (Wang, Smith & Hyndman 2006):

        trend_strength     = max(0, 1 - Var(residual) / Var(trend + residual))
        seasonal_strength  = max(0, 1 - Var(residual) / Var(seasonal + residual))

    Values near 1.0 indicate strong components; near 0.0 indicates the
    component explains little variance.

    Configurable parameters
    (via config["tasks"]["time_series"]["tasks"]["stl_decomposition"]):
        seasonal_period (int): Seasonal period. Default: inferred from frequency.
        robust (bool): Use robust LOESS (resistant to outliers). Default: True
    """

    def run(self) -> None:
        """
        Execute STL decomposition and populate self.output.

        Raises:
            Exception: Re-raised if a context is present.

        """
        try:
            df: DataFrame = to_pandas(self.input_data)

            ts_config: TimeSeriesConfig = read_time_series_config(self)

            if not ts_config.enabled:
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    **make_disabled_result(
                        self.name,
                        "Time series analysis is not enabled. Set "
                        "time_series.enabled: true and "
                        "time_series.datetime_index_column in config.",
                    ),
                )
                return

            try:
                validate_time_series_config(ts_config, df, self)
                df, value_cols = prepare_time_series_df(df, ts_config, self)
            except TimeSeriesSetupError as e:
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    **make_disabled_result(self.name, str(e)),
                )
                return

            ts_tasks_cfg: dict = self.get_shared_param("time_series", "tasks") or {}
            stl_cfg: dict = (
                ts_tasks_cfg.get("stl_decomposition", {}) if ts_tasks_cfg else {}
            )

            configured_period = stl_cfg.get("seasonal_period")
            if configured_period is not None:
                configured_period = int(configured_period)
            robust = bool(stl_cfg.get("robust", True))

            frequency: str | None = infer_frequency(
                df[ts_config.index_col],
                ts_config.frequency,
                self,
            )

            # Determine seasonal period
            if configured_period is not None:
                seasonal_period: int = configured_period
                self._log(
                    f"    Using configured seasonal period: {seasonal_period}.",
                    "debug",
                )
            else:
                seasonal_period = _infer_seasonal_period(frequency, len(df), self)

            if seasonal_period is None:
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    **make_disabled_result(
                        self.name,
                        "STL decomposition requires a seasonal period. "
                        "Set time_series.tasks.stl_decomposition.seasonal_period "
                        "in config (e.g. 7 for daily/weekly, 12 for monthly/annual). "
                        "DSBF could not determine the period automatically from "
                        f"the inferred frequency '{frequency}'.",
                    ),
                )
                return

            # Timestamps as ISO strings for JSON serialisation
            # timestamps = (
            #     df[ts_config.index_col].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
            # )

            results: dict[str, dict[str, Any]] = {}

            for col in value_cols:
                series: Series = df[col].dropna()
                n: int = len(series)

                if n < seasonal_period * 2:
                    self._log(
                        f"    '{col}' skipped: n={n} < 2 x period={seasonal_period}.",
                        "debug",
                    )
                    continue

                # Align timestamps to non-null series index
                col_timestamps = (
                    df.loc[series.index, ts_config.index_col]
                    .dt.strftime("%Y-%m-%dT%H:%M:%S")
                    .tolist()
                )

                try:
                    stl = STL(
                        series.values,
                        period=seasonal_period,
                        robust=robust,
                    )
                    res = stl.fit()
                except Exception as e:
                    self._log(f"    '{col}' STL failed: {e}", "warn")
                    continue

                trend = res.trend
                seasonal = res.seasonal
                residual = res.resid
                observed = series.to_numpy()

                # Strength metrics (Wang, Smith & Hyndman 2006)
                var_resid = float(np.var(residual, ddof=1))
                var_trend_resid = float(np.var(trend + residual, ddof=1))
                var_seasonal_resid = float(np.var(seasonal + residual, ddof=1))

                trend_strength = float(
                    max(
                        0.0,
                        (
                            1.0 - var_resid / var_trend_resid
                            if var_trend_resid > 0
                            else 0.0
                        ),
                    ),
                )
                seasonal_strength = float(
                    max(
                        0.0,
                        (
                            1.0 - var_resid / var_seasonal_resid
                            if var_seasonal_resid > 0
                            else 0.0
                        ),
                    ),
                )

                results[col] = {
                    "timestamps": col_timestamps,
                    "observed": [round(float(v), 6) for v in observed],
                    "trend": [round(float(v), 6) for v in trend],
                    "seasonal": [round(float(v), 6) for v in seasonal],
                    "residual": [round(float(v), 6) for v in residual],
                    "trend_strength": round(trend_strength, 4),
                    "seasonal_strength": round(seasonal_strength, 4),
                    "seasonal_period": seasonal_period,
                    "robust": robust,
                    "n": n,
                }

                self._log(
                    f"    '{col}': trend_strength={trend_strength:.3f}, "
                    f"seasonal_strength={seasonal_strength:.3f}",
                    "debug",
                )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"STL decomposition completed for {len(results)} column(s). "
                        f"Seasonal period: {seasonal_period}."
                    ),
                    "columns_decomposed": len(results),
                    "seasonal_period": seasonal_period,
                    "index_column": ts_config.index_col,
                    "frequency": frequency,
                    "robust": robust,
                },
                data=results,
                metadata={
                    "index_column": ts_config.index_col,
                    "frequency": frequency,
                    "seasonal_period": seasonal_period,
                    "robust": robust,
                    "suggested_viz_type": "line",
                    "recommended_section": "Time Series",
                    "display_priority": "high",
                    "column_types": self.get_column_type_info(value_cols),
                },
            )

            for col, col_data in results.items():
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
        Generate EDA guidance for STL decomposition findings.

        Args:
            col: Column name.
            col_data: Decomposition result dict.

        """
        trend_strength = col_data["trend_strength"]
        seasonal_strength = col_data["seasonal_strength"]
        period = col_data["seasonal_period"]
        n = col_data["n"]

        def _strength_label(v: float) -> str:
            if v >= 0.7:
                return "strong"
            if v >= 0.4:
                return "moderate"
            if v >= 0.1:
                return "weak"
            return "negligible"

        trend_label: str = _strength_label(trend_strength)
        seasonal_label: str = _strength_label(seasonal_strength)

        body: str = (
            f"STL decomposition of '{col}' (n={n:,}, period={period}): "
            f"trend strength={trend_strength:.3f} ({trend_label}), "
            f"seasonal strength={seasonal_strength:.3f} ({seasonal_label}).\n\n"
        )

        if trend_strength >= 0.4:
            body += (
                f"A {trend_label} trend component explains a substantial portion "
                f"of the variance. This confirms that the series is not stationary "
                f"in mean - differencing or detrending will be needed before "
                f"fitting ARIMA or other stationary-assumption models.\n\n"
            )
        else:
            body += (
                "The trend component is weak, suggesting the long-run level "
                "of the series is relatively stable.\n\n"
            )

        if seasonal_strength >= 0.4:
            body += (
                f"A {seasonal_label} seasonal component with period={period} "
                f"is present. Seasonal differencing or seasonal ARIMA (SARIMA) "
                f"may be appropriate. The seasonal component can be inspected "
                f"in the Time Series tab."
            )
        else:
            body += (
                "The seasonal component is weak - standard ARIMA (non-seasonal) "
                "models may be adequate."
            )

        level = "info"
        if trend_strength >= 0.4 and seasonal_strength >= 0.4:
            level = "warn"

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=(
                f"STL: trend={trend_label} ({trend_strength:.3f}), "
                f"seasonal={seasonal_label} ({seasonal_strength:.3f})"
            ),
            body=body.strip(),
            actions=[
                {
                    "action": "seasonal_difference",
                    "detail": (
                        f"df[col].diff({period}).dropna() - removes period={period} "
                        "seasonality before modelling"
                    ),
                    "condition": (
                        "if seasonal_strength >= 0.4 "
                        f"(currently {seasonal_strength:.3f})"
                    ),
                },
                {
                    "action": "detrend_or_difference",
                    "detail": (
                        "Apply first differencing df[col].diff().dropna() "
                        "or subtract the STL trend component before modelling"
                    ),
                    "condition": (
                        f"if trend_strength >= 0.4 (currently {trend_strength:.3f})"
                    ),
                },
            ],
            metric={
                "trend_strength": trend_strength,
                "seasonal_strength": seasonal_strength,
                "trend_label": trend_label,
                "seasonal_label": seasonal_label,
                "seasonal_period": period,
                "n": n,
            },
        )
