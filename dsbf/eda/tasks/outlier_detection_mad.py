# dsbf/eda/tasks/outlier_detection_mad.py

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

if TYPE_CHECKING:
    from pandas import Series

# ── MAD scoring ───────────────────────────────────────────────────────────────


def _mad_score(series: pd.Series) -> pd.Series | None:
    """
    Compute the Modified Z-score for each value using Median Absolute Deviation.

    The Modified Z-score is defined as:

        MZS(x) = 0.6745 x (x - median) / MAD

    where MAD = median(|x - median(x)|) and 0.6745 is the consistency
    factor that makes MAD equivalent to the standard deviation for a normal
    distribution.

    Unlike the standard Z-score, the Modified Z-score uses the median and
    MAD rather than the mean and std — both of which are corrupted by the
    very outliers being detected. It is therefore robust to outliers and
    appropriate for skewed or contaminated distributions.

    Args:
        series: Non-null numeric Series.

    Returns:
        Series of absolute Modified Z-scores, or None if MAD is zero
        (constant or near-constant column).

    """
    median: float = series.median()
    mad: float = (series - median).abs().median()

    if mad == 0:
        return None

    return (0.6745 * (series - median) / mad).abs()


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="outlier_detection_mad",
    display_name="Outlier Detection (MAD)",
    description=(
        "Detects outliers using the Modified Z-score (Median Absolute Deviation). "
        "Robust to skewed distributions and existing outliers — unlike standard "
        "Z-score, the median and MAD are not corrupted by the values being detected."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["outliers", "numeric", "robust", "mad"],
    expected_semantic_types=["continuous"],
)
class OutlierDetectionMAD(BaseTask):
    """
    Detect outliers using the Modified Z-score (Median Absolute Deviation).

    The Modified Z-score is defined as:

        MZS(x) = 0.6745 x (x - median) / MAD

    where MAD = median(|x - median(x)|).

    **Why MAD over standard Z-score:**
    The standard Z-score uses the mean and standard deviation, both of which
    are heavily influenced by outliers. A column with extreme values will have
    an inflated std, causing the Z-scores of the outliers to appear smaller
    than they actually are relative to the bulk of the data. MAD-based scoring
    is entirely robust to this masking effect.

    **Threshold:**
    Iglewicz and Hoaglin (1993) recommend |MZS| > 3.5 as the standard
    threshold for flagging outliers. This is configurable via ``threshold``.

    **Comparison with detect_outliers:**
    ``detect_outliers`` uses IQR fences and standard Z-score. This task uses
    the more robust MAD method. The two tasks are complementary — a value
    flagged by both methods is more likely a genuine outlier than one flagged
    by only one.

    Columns where MAD = 0 (constant or near-constant after median subtraction)
    are skipped — the Modified Z-score is undefined for such columns.

    Configurable parameters (via config["tasks"]["outlier_detection_mad"]):
        threshold (float): |MZS| above which a value is flagged as an outlier.
            Default: 3.5 (Iglewicz & Hoaglin 1993 recommendation)
        min_n (int): Minimum non-null values to analyse a column. Default: 10
    """

    def run(self) -> None:
        """
        Execute MAD-based outlier detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'continuous' column(s)",
                "debug",
            )

            threshold_raw: Any | None = self.get_task_param("threshold")
            threshold: float = (
                float(threshold_raw) if threshold_raw is not None else 3.5
            )

            min_n_raw: Any | None = self.get_task_param("min_n")
            min_n: int = int(min_n_raw) if min_n_raw is not None else 10

            numeric_df = df.select_dtypes(include=np.number)
            results: dict[str, dict[str, Any]] = {}

            for col in numeric_df.columns:
                series = numeric_df[col].dropna()

                if len(series) < min_n:
                    self._log(
                        f"    '{col}' skipped: only {len(series)} non-null values.",
                        "debug",
                    )
                    continue

                scores: Series | None = _mad_score(series)

                if scores is None:
                    self._log(
                        f"    '{col}' skipped: MAD is zero (near-constant column).",
                        "debug",
                    )
                    continue

                outlier_mask: Series[bool] = scores > threshold
                outlier_count = int(outlier_mask.sum())
                outlier_pct: float = round(outlier_count / len(series), 4)

                # Store the top outlier values (by MZS score) for inspection
                top_outliers = (
                    series[outlier_mask]
                    .reindex(scores[outlier_mask].nlargest(10).index)
                    .tolist()
                )
                top_outliers: list[float] = [round(float(v), 6) for v in top_outliers]

                median_val: float = round(float(series.median()), 6)
                mad_val: float = round(
                    float((series - series.median()).abs().median()),
                    6,
                )
                max_score: float = round(float(scores.max()), 4)

                results[col] = {
                    "outlier_count": outlier_count,
                    "outlier_pct": outlier_pct,
                    "threshold": threshold,
                    "median": median_val,
                    "mad": mad_val,
                    "max_modified_z_score": max_score,
                    "top_outlier_values": top_outliers,
                    "n": len(series),
                }

                self._log(
                    f"    '{col}': {outlier_count} outlier(s) "
                    f"({outlier_pct:.1%}), max MZS={max_score:.2f}",
                    "debug",
                )

            cols_with_outliers: int = sum(
                1 for v in results.values() if v["outlier_count"] > 0
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"MAD outlier detection run on {len(results)} column(s); "
                        f"{cols_with_outliers} have outlier(s) at "
                        f"|MZS| > {threshold}."
                    ),
                    "columns_analysed": len(results),
                    "columns_with_outliers": cols_with_outliers,
                    "threshold": threshold,
                },
                data=results,
                metadata={
                    "threshold": threshold,
                    "method": "modified_z_score",
                    "consistency_factor": 0.6745,
                    "suggested_viz_type": "scatter",
                    "recommended_section": "Distributions",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, col_results in results.items():
                if col_results["outlier_count"] > 0:
                    self._attach_guidance(col, col_results)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, col_results: dict[str, Any]) -> None:
        """
        Generate EDA and ML guidance for a column with MAD outliers.

        Args:
            col: Column name.
            col_results: Result dict for this column.

        """
        count = col_results["outlier_count"]
        pct = col_results["outlier_pct"]
        threshold = col_results["threshold"]
        max_score = col_results["max_modified_z_score"]
        median = col_results["median"]
        mad = col_results["mad"]
        top_vals = col_results["top_outlier_values"][:5]
        n = col_results["n"]

        level: Literal["info", "warn"] = "warn" if pct >= 0.05 else "info"

        eda_body: str = (
            f"'{col}' has {count} outlier(s) ({pct:.1%} of {n:,} values) "
            f"with |Modified Z-score| > {threshold} (Iglewicz & Hoaglin 1993 "
            f"threshold). The column median is {median:.4g} and MAD is "
            f"{mad:.4g}. Maximum Modified Z-score: {max_score:.2f}. "
            f"Most extreme values: {top_vals}. "
            f"Unlike IQR or standard Z-score, the MAD method is robust to "
            f"existing outliers — these flagged values are genuinely anomalous "
            f"relative to the bulk of the distribution, not artefacts of "
            f"an inflated standard deviation. Investigate whether they represent "
            f"genuine rare events, data entry errors, or a separate sub-population."
        )

        ml_body: str = (
            f"'{col}' has {count} MAD outlier(s) ({pct:.1%}). "
            f"Distance-based models (KNN, SVM), linear models, and PCA are "
            f"sensitive to extreme values — these {count} point(s) may "
            f"disproportionately influence decision boundaries and coefficient "
            f"estimates. Consider Winsorising at the 1st/99th percentile or "
            f"applying a robust scaler before training. Tree-based models "
            f"(Random Forest, XGBoost) are largely unaffected."
        )

        metric: dict[str, Any] = {
            "outlier_count": count,
            "outlier_pct": pct,
            "threshold": threshold,
            "max_modified_z_score": max_score,
            "median": median,
            "mad": mad,
            "n": n,
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=(
                f"MAD Outliers: {count} value(s) ({pct:.1%}) with |MZS| > {threshold}"
            ),
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=level,
            title=f"Robust Outliers Detected ({count} value(s))",
            body=ml_body.strip(),
            actions=[
                {
                    "action": "winsorise",
                    "column": col,
                    "detail": "Cap at 1st/99th percentile to reduce outlier influence",
                },
                {
                    "action": "robust_scale",
                    "method": "RobustScaler",
                    "column": col,
                    "detail": "Scale using median and IQR instead of mean and std",
                },
            ],
            metric=metric,
        )
