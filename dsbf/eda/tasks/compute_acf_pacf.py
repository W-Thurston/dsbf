# dsbf/eda/tasks/compute_acf_pacf.py

from typing import TYPE_CHECKING, Any

from statsmodels.tsa.stattools import acf, pacf

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


@register_task(
    name="compute_acf_pacf",
    display_name="ACF / PACF",
    description=(
        "Computes autocorrelation (ACF) and partial autocorrelation (PACF) "
        "functions for each configured value column. Requires "
        "time_series.datetime_index_column to be set in config."
    ),
    depends_on=["infer_types", "temporal_gap_detection"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["time_series", "autocorrelation", "acf", "pacf"],
    expected_semantic_types=["continuous"],
)
class ComputeACFPACF(BaseTask):
    """
    Compute ACF and PACF for each value column in the time series.

    **ACF (Autocorrelation Function):** Measures the correlation of a series
    with a lagged version of itself. Includes both direct and indirect effects.
    Useful for identifying moving average (MA) order in ARIMA models.

    **PACF (Partial Autocorrelation Function):** Measures the correlation at
    lag k after removing the effects of all shorter lags. Useful for
    identifying autoregressive (AR) order.

    **Output format:**
    Each column result contains ``lags``, ``acf_values``, ``pacf_values``,
    and ``confidence_bands`` lists ready for direct frontend rendering.
    Significant lags (where the value exceeds the confidence band) are
    identified and included in guidance.

    **Confidence bands:**
    Computed at the configured ``alpha`` level using Bartlett's formula
    (ACF) and the standard asymptotic formula (PACF). Values outside
    ±1.96/√n (at α=0.05) are considered statistically significant.

    **Gap warning:**
    If ``temporal_gap_detection`` found large gaps in the index column,
    a warning is logged. ACF/PACF computed on gapped series treats
    irregular observations as if they were equally spaced - this is
    incorrect and can produce misleading results.

    Configurable parameters (via config["tasks"]["time_series"]["tasks"]["acf_pacf"]):
        max_lags (int): Maximum number of lags to compute. Default: 40
        alpha (float): Confidence interval level. Default: 0.05
    """

    def run(self) -> None:
        """
        Execute ACF/PACF computation and populate self.output.

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
                        "time_series.datetime_index_column in config to "
                        "run ACF/PACF analysis.",
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

            # Task-level params from the nested time_series.tasks.acf_pacf block
            ts_tasks_cfg: dict = self.get_shared_param("time_series", "tasks") or {}
            acf_cfg: dict = ts_tasks_cfg.get("acf_pacf", {}) if ts_tasks_cfg else {}

            max_lags_raw = acf_cfg.get("max_lags", 40)
            max_lags = int(max_lags_raw)
            alpha_raw = acf_cfg.get("alpha", 0.05)
            alpha = float(alpha_raw)

            frequency: str | None = infer_frequency(
                df[ts_config.index_col],
                ts_config.frequency,
                self,
            )

            results: dict[str, dict[str, Any]] = {}

            for col in value_cols:
                series: Series = df[col].dropna()
                n: int = len(series)

                if n < max_lags + 2:
                    actual_lags: int = max(2, n // 2)
                    self._log(
                        f"    '{col}': n={n} < max_lags+2 - "
                        f"reducing to {actual_lags} lags.",
                        "debug",
                    )
                else:
                    actual_lags = max_lags

                try:
                    acf_vals, acf_confint = acf(
                        series.values,
                        nlags=actual_lags,
                        alpha=alpha,
                        fft=True,
                    )
                    pacf_vals, pacf_confint = pacf(
                        series.values,
                        nlags=actual_lags,
                        alpha=alpha,
                        method="ywm",
                    )
                except Exception as e:  # noqa: BLE001
                    self._log(f"    '{col}' ACF/PACF failed: {e}", "warn")
                    continue

                lags: list[int] = list(range(len(acf_vals)))

                # Confidence bands: symmetric around zero
                # confint shape is (nlags+1, 2) - lower/upper relative to acf value
                acf_lower: list[float] = [
                    round(float(acf_confint[i, 0] - acf_vals[i]), 6)
                    for i in range(len(acf_vals))
                ]
                acf_upper: list[float] = [
                    round(float(acf_confint[i, 1] - acf_vals[i]), 6)
                    for i in range(len(acf_vals))
                ]
                pacf_lower: list[float] = [
                    round(float(pacf_confint[i, 0] - pacf_vals[i]), 6)
                    for i in range(len(pacf_vals))
                ]
                pacf_upper: list[float] = [
                    round(float(pacf_confint[i, 1] - pacf_vals[i]), 6)
                    for i in range(len(pacf_vals))
                ]

                # Significant lags: where |value| exceeds upper CI bound
                # Skip lag 0 (always 1.0 for ACF)
                acf_sig: list[int] = [
                    i
                    for i in range(1, len(acf_vals))
                    if abs(acf_vals[i]) > abs(acf_upper[i])
                ]
                pacf_sig: list[int] = [
                    i
                    for i in range(1, len(pacf_vals))
                    if abs(pacf_vals[i]) > abs(pacf_upper[i])
                ]

                results[col] = {
                    "lags": lags,
                    "acf_values": [round(float(v), 6) for v in acf_vals],
                    "pacf_values": [round(float(v), 6) for v in pacf_vals],
                    "acf_confidence_lower": acf_lower,
                    "acf_confidence_upper": acf_upper,
                    "pacf_confidence_lower": pacf_lower,
                    "pacf_confidence_upper": pacf_upper,
                    "acf_significant_lags": acf_sig,
                    "pacf_significant_lags": pacf_sig,
                    "n": n,
                    "max_lags": actual_lags,
                    "alpha": alpha,
                }

                self._log(
                    f"    '{col}': {len(acf_sig)} significant ACF lag(s), "
                    f"{len(pacf_sig)} significant PACF lag(s).",
                    "debug",
                )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"ACF/PACF computed for {len(results)} column(s) "
                        f"against '{ts_config.index_col}'."
                    ),
                    "columns_computed": len(results),
                    "index_column": ts_config.index_col,
                    "max_lags": max_lags,
                    "frequency": frequency,
                },
                data=results,
                metadata={
                    "index_column": ts_config.index_col,
                    "frequency": frequency,
                    "max_lags": max_lags,
                    "alpha": alpha,
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
        Generate EDA guidance for ACF/PACF findings.

        Args:
            col: Value column name.
            col_data: ACF/PACF result dict for this column.

        """
        acf_sig = col_data["acf_significant_lags"]
        pacf_sig = col_data["pacf_significant_lags"]
        n = col_data["n"]
        alpha = col_data["alpha"]

        if not acf_sig and not pacf_sig:
            body: str = (
                f"'{col}' shows no statistically significant autocorrelation "
                f"at any lag up to {col_data['max_lags']} "
                f"(n={n:,}, α={alpha}). "
                f"This is consistent with white noise - past values do not "
                f"predict future values. Simple models (mean, random walk) "
                f"may be adequate."
            )
            level = "info"
        else:
            # Characterise the pattern from ACF/PACF profile
            if acf_sig and pacf_sig:
                # Both have significant lags - ARMA structure likely
                pattern: str = (
                    f"Both ACF and PACF have significant lags "
                    f"(ACF: {acf_sig[:5]}, PACF: {pacf_sig[:5]}). "
                    f"This is consistent with an ARMA process. "
                    f"The number of significant PACF lags suggests AR order; "
                    f"the ACF pattern suggests MA order."
                )
            elif pacf_sig and not acf_sig:
                pattern = (
                    f"PACF cuts off after lag {max(pacf_sig)} with ACF "
                    f"decaying gradually. This is consistent with a pure "
                    f"AR({max(pacf_sig)}) process."
                )
            else:
                pattern = (
                    f"ACF cuts off after lag {max(acf_sig)} with PACF "
                    f"decaying gradually. This is consistent with a pure "
                    f"MA({max(acf_sig)}) process."
                )

            body = (
                f"'{col}' shows significant autocorrelation (n={n:,}, α={alpha}). "
                f"{pattern} "
                f"Inspect the ACF/PACF chart in the Time Series tab to confirm "
                f"the pattern visually before selecting a model order. "
                f"These interpretations are guidelines - real data often shows "
                f"mixed or ambiguous patterns."
            )
            level = "info"

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=(
                f"ACF/PACF: {len(acf_sig)} significant ACF lag(s), "
                f"{len(pacf_sig)} significant PACF lag(s)"
            ),
            body=body.strip(),
            actions=[],
            metric={
                "acf_significant_lags": acf_sig,
                "pacf_significant_lags": pacf_sig,
                "n": n,
                "max_lags": col_data["max_lags"],
            },
        )
