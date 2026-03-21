# dsbf/eda/tasks/compute_kurtosis.py

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

# ── Kurtosis classification ────────────────────────────────────────────────────
#
# Fisher (excess) kurtosis is used throughout: a normal distribution has
# excess kurtosis = 0. Positive values indicate heavier tails than normal
# (leptokurtic), negative values indicate lighter tails (platykurtic).
#
# Guidance thresholds:
#   |k| < 1.0   → mesokurtic — normal-like tails, no guidance needed
#   k >= 1.0    → mildly leptokurtic — info
#   k >= 3.0    → strongly leptokurtic — warn (meaningful outlier risk)
#   k <= -1.0   → platykurtic — info (thin tails, uniform-like distribution)


def _classify_kurtosis(k: float) -> str:
    """
    Classify excess kurtosis into a tail-behaviour label.

    Args:
        k: Fisher excess kurtosis (normal = 0).

    Returns:
        One of ``"strongly_leptokurtic"``, ``"mildly_leptokurtic"``,
        ``"mesokurtic"``, or ``"platykurtic"``.

    """
    if k >= 3.0:  # noqa: PLR2004
        return "strongly_leptokurtic"
    if k >= 1.0:
        return "mildly_leptokurtic"
    if k <= -1.0:
        return "platykurtic"
    return "mesokurtic"


@register_task(
    name="compute_kurtosis",
    display_name="Compute Kurtosis",
    description=(
        "Computes Fisher excess kurtosis for continuous numeric columns. "
        "Positive values indicate heavier tails than normal (leptokurtic); "
        "negative values indicate lighter tails (platykurtic)."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["distribution", "kurtosis", "statistics", "numeric"],
    expected_semantic_types=["continuous"],
)
class ComputeKurtosis(BaseTask):
    """
    Compute Fisher excess kurtosis for all continuous numeric columns.

    Uses ``scipy.stats.kurtosis`` with ``fisher=True`` (excess kurtosis,
    normal distribution = 0) and ``bias=False`` (unbiased estimator, suited
    for samples rather than populations).

    Kurtosis complements skewness:

    - Skewness measures distributional *asymmetry*.
    - Kurtosis measures *tail weight* relative to a normal distribution.

    A column can be perfectly symmetric (skewness ≈ 0) but still have fat
    tails (high positive kurtosis), which signals elevated outlier risk.

    Classification thresholds (Fisher excess kurtosis):

    - ``mesokurtic``:          |k| < 1.0 — normal-like tails
    - ``mildly_leptokurtic``:  1.0 ≤ k < 3.0 — moderately heavier tails
    - ``strongly_leptokurtic``: k ≥ 3.0 — fat tails, significant outlier risk
    - ``platykurtic``:          k ≤ -1.0 — thin tails, uniform-like

    EDA guidance blurbs are emitted for leptokurtic and platykurtic columns.
    Mesokurtic columns produce no guidance — their tail behaviour is unremarkable.

    ML guidance is emitted for strongly leptokurtic columns, noting that
    distance-based models (KNN, SVM) and regularised regression are sensitive
    to fat-tailed features.

    Columns with fewer than 4 non-null values are skipped — kurtosis requires
    at least 4 observations to produce a meaningful estimate.
    """

    def run(self) -> None:
        """
        Execute kurtosis computation and populate self.output.

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

            numeric_df = df.select_dtypes(include=np.number)
            kurtosis_results: dict[str, dict] = {}

            for col in numeric_df.columns:
                series = numeric_df[col].dropna()

                if len(series) < 4:  # noqa: PLR2004
                    self._log(
                        f"    '{col}' skipped: fewer than 4 non-null values "
                        f"({len(series)}).",
                        "debug",
                    )
                    continue

                try:
                    # fisher=True: excess kurtosis (normal = 0)
                    # bias=False:  unbiased sample estimator
                    k = float(scipy_kurtosis(series.values, fisher=True, bias=False))
                except Exception as e:  # noqa: BLE001
                    self._log(
                        f"    '{col}' kurtosis failed: {type(e).__name__} - {e}",
                        "debug",
                    )
                    continue

                classification: str = _classify_kurtosis(k)
                kurtosis_results[col] = {
                    "kurtosis": round(k, 4),
                    "classification": classification,
                }
                self._log(f"    '{col}': kurtosis={k:.4f} ({classification})", "debug")

            notable: list[str] = [
                col
                for col, v in kurtosis_results.items()
                if v["classification"] != "mesokurtic"
            ]
            self._log(
                f"    {len(notable)} column(s) with notable kurtosis "
                f"({len(kurtosis_results)} total computed).",
                "debug",
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed kurtosis for {len(kurtosis_results)} numeric "
                        f"columns; {len(notable)} with notable tail behaviour."
                    ),
                    "notable_count": len(notable),
                },
                data=kurtosis_results,
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Distributions",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, info in kurtosis_results.items():
                if info["classification"] != "mesokurtic":
                    self._attach_guidance(col, info["kurtosis"], info["classification"])

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, k: float, classification: str) -> None:
        """
        Generate EDA and ML guidance for a column with notable kurtosis.

        Args:
            col: Column name.
            k: Fisher excess kurtosis value.
            classification: One of the four classification labels from
                ``_classify_kurtosis``.

        """
        k_str: str = f"{k:.2f}"

        if classification == "strongly_leptokurtic":
            eda_level = "warn"
            eda_title: str = f"Fat Tails — Strongly Leptokurtic (kurtosis {k_str})"
            eda_body: str = (
                f"'{col}' has an excess kurtosis of {k_str}, well above the normal "
                f"distribution baseline of 0. Fat-tailed distributions concentrate "
                f"most observations near the centre but produce extreme values far "
                f"more often than a normal distribution would. Standard deviation "
                f"understates the true spread — check the 95th and 99th percentiles "
                f"for a better sense of tail extent. Outlier detection and "
                f"distributional tests will be heavily influenced by these extremes."
            )
            ml_level = "warn"
            ml_title: str = f"Fat-Tailed Feature — Outlier Risk (kurtosis {k_str})"
            ml_body: str = (
                f"'{col}' has strongly leptokurtic distribution (kurtosis {k_str}). "
                f"Fat tails mean outliers are structurally common, not anomalous. "
                f"Distance-based models (KNN, SVM) and regularised regression "
                f"(Ridge, Lasso) are sensitive to extreme values — consider "
                f"Winsorising at the 1st/99th percentile or applying a log1p or "
                f"Box-Cox transform before training. Tree-based models are "
                f"invariant to monotonic transforms and generally unaffected."
            )
            ml_actions: list[dict[str, str]] = [
                {
                    "action": "winsorise",
                    "column": col,
                    "detail": "Cap at 1st/99th percentile to reduce tail influence",
                },
                {
                    "action": "transform",
                    "method": "log1p or Box-Cox",
                    "column": col,
                    "detail": "Reduce tail weight for linear and distance-based models",
                },
            ]
            emit_ml = True

        elif classification == "mildly_leptokurtic":
            eda_level = "info"
            eda_title = f"Slightly Heavy Tails — Leptokurtic (kurtosis {k_str})"
            eda_body = (
                f"'{col}' has an excess kurtosis of {k_str}, indicating moderately "
                f"heavier tails than a normal distribution. Extreme values are more "
                f"common than the standard deviation alone suggests. Check the "
                f"histogram and 99th percentile before concluding that outliers "
                f"are anomalous — at this kurtosis level they may simply be "
                f"characteristic of the distribution."
            )
            ml_level = "info"
            ml_title = f"Moderately Leptokurtic (kurtosis {k_str})"
            ml_body = (
                f"'{col}' has mildly heavy tails (kurtosis {k_str}). At this level "
                f"the impact on most models is limited, but linear models and "
                f"distance-based algorithms may still benefit from a Winsorise or "
                f"log transform if extreme values are driving predictions."
            )
            ml_actions = [
                {
                    "action": "monitor",
                    "column": col,
                    "detail": "Check residuals after fitting linear models",
                },
            ]
            emit_ml = True

        else:  # platykurtic
            eda_level = "info"
            eda_title = f"Thin Tails — Platykurtic (kurtosis {k_str})"
            eda_body = (
                f"'{col}' has an excess kurtosis of {k_str}, below the normal "
                f"distribution baseline of 0. Platykurtic distributions have "
                f"fewer extreme values than normal — the data is more uniformly "
                f"spread across its range with less concentration in the centre "
                f"and tails. This is common in bounded or discretised features. "
                f"Outlier detection methods calibrated for normal distributions "
                f"will produce fewer flags than expected."
            )
            ml_level = "info"
            ml_title = f"Thin-Tailed Feature (kurtosis {k_str})"
            ml_body = (
                f"'{col}' has a platykurtic distribution (kurtosis {k_str}). "
                f"Thin tails generally cause fewer modelling issues than fat tails. "
                f"Normality-based models may be mildly affected; tree-based "
                f"models are unaffected."
            )
            ml_actions = []
            emit_ml = True

        metric: dict[str, float | str] = {
            "kurtosis": k,
            "classification": classification,
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=eda_level,
            title=eda_title,
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )

        if emit_ml:
            self.add_guidance(
                result=self.output,
                column=col,
                phase="ml",
                level=ml_level,
                title=ml_title,
                body=ml_body.strip(),
                actions=ml_actions,
                metric=metric,
            )
