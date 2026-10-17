# dsbf/eda/tasks/detect_skewness.py

from typing import Literal

import numpy as np
from scipy.stats import skew

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.reco_engine import get_recommendation_tip

# Skewness thresholds (absolute value).
# |skew| <= 0.5: symmetric - no guidance emitted.
# 0.5 < |skew| <= 1.0: mild - info level.
# 1.0 < |skew| <= 2.0: moderate - warn level, transform recommended.
# |skew| > 2.0: heavy - warn level, transform strongly recommended.
_SKEW_MILD = 0.5
_SKEW_MOD = 1.0
_SKEW_HEAVY = 2.0


@register_task(
    display_name="Detect Skewness",
    description="Computes skewness of numeric columns.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    phase="eda",
    tags=["distribution", "skew"],
    expected_semantic_types=["continuous"],
)
class DetectSkewness(BaseTask):
    """
    Computes Fisher skewness for all numeric columns tagged as 'continuous'.

    Skewness quantifies the asymmetry of a distribution. For each column with
    notable asymmetry (|skew| > 0.5), this task generates two guidance blurbs:

    - **EDA blurb**: describes the distribution as observed - what the skew value
      means about the shape of the data, with no modeling language or action chips.
    - **ML blurb**: prescribes what to do before modeling - which model families
      are affected, what transforms are recommended, expressed as structured actions
      a user or agent can act on directly.

    Both blurbs are self-contained (column name, metric value, and context are
    explicit) so they can be consumed meaningfully without surrounding context
    by a downstream LLM, agent, or rendering layer.

    Symmetric columns (|skew| ≤ 0.5) receive no blurb - a clean result does not
    need a finding.
    """

    def run(self) -> None:  # noqa: C901
        """
        Execute skewness computation and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, numeric_cols, excluded = self.setup_run_native("'continuous'")
            column_stats: dict[str, dict[str, float]] = {}

            if not numeric_cols:
                self.output = self.make_empty_result(
                    "No continuous columns found — skewness detection skipped.",
                    excluded,
                )
                return

            if is_polars(df):
                df_sel = df.select(numeric_cols) if numeric_cols else df
                for col in df_sel.columns:
                    series = df_sel[col].drop_nulls().to_numpy()
                    if series.size == 0:
                        self._log(
                            f"    '{col}' skipped: empty after drop_nulls()",
                            "debug",
                        )
                        continue
                    mean = float(np.mean(series))
                    std = float(np.std(series))
                    skew_val: float = (
                        float(np.mean(((series - mean) / std) ** 3))
                        if std != 0
                        else 0.0
                    )
                    column_stats[col] = {
                        "skew": skew_val,
                        "mean": mean,
                        "median": float(np.median(series)),
                        "std": std,
                    }
            else:
                numeric_df = (
                    df[numeric_cols]
                    if numeric_cols
                    else df.select_dtypes(include="number")
                )
                for col in numeric_df.columns:
                    series = numeric_df[col].dropna()
                    if series.empty:
                        self._log(f"    '{col}' skipped: empty after dropna()", "debug")
                        continue
                    # Constant column has undefined skewness - treat as 0.
                    skew_val = 0.0 if series.nunique() == 1 else float(skew(series))
                    column_stats[col] = {
                        "skew": skew_val,
                        "mean": float(series.mean()),
                        "median": float(series.median()),
                        "std": float(series.std()),
                    }

            # data stores the raw skew float per column for downstream consumption.
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed skewness for {len(column_stats)} numeric column(s)."
                    ),
                },
                data={col: stats["skew"] for col, stats in column_stats.items()},
                metadata={
                    "suggested_viz_type": "histogram",
                    "recommended_section": "Distributions",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        numeric_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, stats in column_stats.items():
                self._attach_guidance(col, stats)

            # ML impact scoring - report the first column with meaningful skew.
            if self.get_engine_param("enable_impact_scoring", True):
                for col, stats in column_stats.items():
                    abs_skew: float = abs(stats["skew"])
                    if abs_skew <= _SKEW_MOD:
                        continue
                    score: float = 0.6 if abs_skew <= _SKEW_HEAVY else 0.8
                    tip: str | None = get_recommendation_tip(
                        self.name,
                        {"skew": stats["skew"]},
                    )
                    self.set_ml_signals(
                        result=self.output,
                        score=score,
                        tags=["transform"],
                        recommendation=tip
                        or (
                            f"Column '{col}' has high skew "
                            f"(skew = {stats['skew']:.2f}). "
                            "Consider transforming it."
                        ),
                    )
                    self.output.summary["column"] = col
                    break

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, stats: dict[str, float]) -> None:
        """
        Generate and attach EDA and ML guidance blurbs for a skewed column.

        Columns with |skew| ≤ 0.5 are symmetric and receive no blurb.
        Thresholds: mild (0.5-1.0), moderate (1.0-2.0), heavy (>2.0).

        Args:
            col: Column name.
            stats: Dict containing ``skew``, ``mean``, ``median``, and ``std``.

        """
        skew_val: float = stats["skew"]
        mean: float | None = stats.get("mean")
        median: float | None = stats.get("median")
        abs_skew: float = abs(skew_val)

        if abs_skew <= _SKEW_MILD:
            return  # Symmetric - no blurb needed

        direction: Literal["left (negative)", "right (positive)"] = (
            "right (positive)" if skew_val > 0 else "left (negative)"
        )
        tail_dir: Literal["higher", "lower"] = "higher" if skew_val > 0 else "lower"
        bulk_dir: Literal["higher", "lower"] = "lower" if skew_val > 0 else "higher"
        dir_word: Literal["Left", "Right"] = "Right" if skew_val > 0 else "Left"

        metric: dict[str, float] = {"skewness": round(skew_val, 4)}
        if mean is not None:
            metric["mean"] = round(mean, 4)
        if median is not None:
            metric["median"] = round(median, 4)

        if abs_skew <= _SKEW_MOD:
            level = "info"
            title: str = f"Mild {dir_word} Skew ({skew_val:+.2f})"

            eda_body: str = (
                f"'{col}' has a mild {direction} skew (skewness = {skew_val:.2f}). "
                f"The distribution is slightly asymmetric, with a minor lean toward "
                f"{tail_dir} values. The mean and median are close, so either is a "
                f"reasonable summary of the centre. Glance at the histogram to confirm "
                f"no unusual clustering or gaps are hiding behind the mild asymmetry."
            )

            ml_body: str = (
                f"'{col}' has mild skewness ({skew_val:.2f}). Most models will handle "
                f"this without transformation. If using linear or distance-based "
                f"models, monitor residuals after fitting - a transform may help "
                f"marginally. Tree-based models are unaffected."
            )
            ml_actions: list[dict] = [
                {
                    "action": "monitor",
                    "column": col,
                    "detail": "Check residual plots after fitting linear models",
                },
            ]

        elif abs_skew <= _SKEW_HEAVY:
            level = "warn"
            title = f"Moderate {dir_word} Skew ({skew_val:+.2f})"

            eda_body = (
                f"'{col}' shows moderate {direction} skewness "
                f"(skewness = {skew_val:.2f}). Values are concentrated toward the "
                f"{bulk_dir} end of the range, with a tail extending toward {tail_dir} "
                f"values. The median is likely more descriptive than the mean here. "
                f"Check for outliers or data quality issues to confirm the skew is "
                f"genuine rather than driven by a small number of anomalous values."
            )

            ml_body = (
                f"Skewness of {skew_val:.2f} in '{col}' will affect linear models "
                f"(linear/logistic regression) and distance-based models (SVM, KNN) "
                f"by distorting coefficient scale and distance calculations. "
                f"Tree-based models (Random Forest, XGBoost) are largely robust. "
                f"A transform is recommended before fitting linear or distance-based "
                f"models."
            )
            if skew_val > 0:
                ml_actions = [
                    {
                        "action": "transform",
                        "method": "log1p",
                        "column": col,
                        "condition": "values >= 0",
                    },
                    {
                        "action": "transform",
                        "method": "sqrt",
                        "column": col,
                        "condition": "values >= 0",
                    },
                    {
                        "action": "transform",
                        "method": "yeo-johnson",
                        "column": col,
                        "condition": "any",
                    },
                ]
            else:
                ml_actions = [
                    {
                        "action": "transform",
                        "method": "reflect_then_log",
                        "column": col,
                        "condition": "left-skewed",
                    },
                    {
                        "action": "transform",
                        "method": "yeo-johnson",
                        "column": col,
                        "condition": "any",
                    },
                ]

        else:
            level = "warn"
            title = f"Heavy {dir_word} Skew ({skew_val:+.2f})"

            eda_body = (
                f"'{col}' has heavy {direction} skewness (skewness = {skew_val:.2f}). "
                f"The bulk of values cluster near the {bulk_dir} end with a long tail "
                f"extending toward {tail_dir} values. The mean is significantly "
                f"distorted by the tail - the median is a much more honest description "
                f"of the typical value. Inspect the tail values directly: check "
                f"whether they represent genuine data, outliers, or data entry errors "
                f"before drawing conclusions about this column."
            )

            ml_body = (
                f"Heavy skewness of {skew_val:.2f} in '{col}' will significantly "
                f"distort linear model coefficients and KNN/SVM distance calculations. "
                f"Tree-based models are robust but may still benefit from "
                f"transformation for interpretability. Transformation is strongly "
                f"recommended before using any non-tree model."
            )
            if skew_val > 0:
                ml_actions = [
                    {
                        "action": "transform",
                        "method": "log1p",
                        "column": col,
                        "condition": "values >= 0",
                    },
                    {
                        "action": "transform",
                        "method": "box-cox",
                        "column": col,
                        "condition": "values > 0",
                    },
                    {
                        "action": "transform",
                        "method": "yeo-johnson",
                        "column": col,
                        "condition": "any",
                    },
                    {
                        "action": "winsorize",
                        "column": col,
                        "detail": "Cap at 1st/99th percentile to reduce tail influence",
                    },
                ]
            else:
                ml_actions = [
                    {
                        "action": "transform",
                        "method": "yeo-johnson",
                        "column": col,
                        "condition": "any",
                    },
                    {
                        "action": "transform",
                        "method": "reflect_then_log",
                        "column": col,
                        "condition": "left-skewed",
                    },
                    {
                        "action": "winsorize",
                        "column": col,
                        "detail": "Cap at 1st/99th percentile to reduce tail influence",
                    },
                ]

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=title,
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=level,
            title=title,
            body=ml_body.strip(),
            actions=ml_actions,
            metric=metric,
        )
