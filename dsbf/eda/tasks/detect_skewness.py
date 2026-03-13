# dsbf/eda/tasks/detect_skewness.py

from typing import Any

import numpy as np
from scipy.stats import skew

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.reco_engine import get_recommendation_tip

# Skewness thresholds (absolute value)
_SKEW_MILD = 0.5  # below: symmetric, no guidance needed
_SKEW_MOD = 1.0  # mild → moderate boundary
_SKEW_HEAVY = 2.0  # moderate → heavy boundary


@register_task(
    display_name="Detect Skewness",
    description="Computes skewness of numeric columns.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    tags=["distribution", "skew"],
    expected_semantic_types=["continuous"],
)
class DetectSkewness(BaseTask):
    """
    Computes skewness for all numeric columns tagged as 'continuous'.

    Skewness quantifies the asymmetry of a distribution. For each column
    with notable asymmetry, this task generates two guidance blurbs:

    - EDA guidance: describes the distribution as observed - what the skew
      value means about the shape of the data, with no modeling language.
    - ML guidance: prescribes what to do before modeling - which model
      families are affected, what transforms are recommended, expressed
      as structured actions an agent or user can act on.

    Both blurbs are self-contained (column name, metric value, and context
    are explicit) so they can be consumed meaningfully without surrounding
    context by a downstream LLM, agent, or rendering layer.
    """

    def run(self) -> None:
        try:
            df: Any = self.input_data
            # col -> {skew, mean, median, std} - collected during computation
            column_stats: dict[str, dict[str, float]] = {}

            numeric_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(numeric_cols)} 'continuous' column(s)", "debug"
            )

            if is_polars(df):
                df_sel = df.select(numeric_cols) if numeric_cols else df
                for col in df_sel.columns:
                    series = df_sel[col].drop_nulls().to_numpy()
                    if series.size == 0:
                        self._log(f"    {col} skipped: empty after dropna()", "debug")
                        continue
                    mean = float(np.mean(series))
                    std = float(np.std(series))
                    skew_val = (
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
                    self._log(f"    {col}: skewness computed", "debug")

            else:
                numeric_df = (
                    df[numeric_cols]
                    if numeric_cols
                    else df.select_dtypes(include="number")
                )
                for col in numeric_df.columns:
                    series = numeric_df[col].dropna()
                    if series.empty:
                        self._log(f"    {col} skipped: empty after dropna()", "debug")
                        continue
                    if series.nunique() == 1:
                        skew_val = 0.0
                        self._log(f"    {col} skipped: constant values", "debug")
                    else:
                        skew_val = float(skew(series))
                    column_stats[col] = {
                        "skew": skew_val,
                        "mean": float(series.mean()),
                        "median": float(series.median()),
                        "std": float(series.std()),
                    }
                    self._log(f"    {col}: skewness computed", "debug")

            # Build TaskResult - data carries raw skew floats as before
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed skewness for {len(column_stats)} numeric column(s)."
                    )
                },
                data={col: stats["skew"] for col, stats in column_stats.items()},
                plots={},
                metadata={
                    "suggested_viz_type": "histogram",
                    "recommended_section": "Distributions",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        numeric_cols + list(excluded.keys())
                    ),
                },
            )

            # Generate per-column EDA + ML guidance blurbs
            for col, stats in column_stats.items():
                self._attach_guidance(col, stats)

            # ML impact scoring (kept for backward compat)
            if self.get_engine_param("enable_impact_scoring", True):
                for col, stats in column_stats.items():
                    abs_skew = abs(stats["skew"])
                    if abs_skew <= _SKEW_MOD:
                        continue
                    score = 0.6 if abs_skew <= _SKEW_HEAVY else 0.8
                    tip = get_recommendation_tip(self.name, {"skew": stats["skew"]})
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

    # ──────────────────────────────────────────────────────────────────────────
    # Guidance generation
    # ──────────────────────────────────────────────────────────────────────────

    def _attach_guidance(self, col: str, stats: dict[str, float]) -> None:
        """
        Generate and attach EDA + ML guidance blurbs for a single column.

        Both blurbs are self-contained: column name, metric value, direction,
        and implication are all explicit so each blurb is meaningful without
        surrounding context (for LLM/agent consumption downstream).

        EDA blurbs are purely descriptive - they characterise the distribution
        as observed. No modeling language, no action chips.

        ML blurbs are prescriptive - they name affected model families and
        provide structured actions (method, condition) a user or agent can
        act on directly.

        Columns with |skew| <= 0.5 are considered symmetric and receive no
        guidance (a clean result does not need a blurb).
        """
        skew_val = stats["skew"]
        mean = stats.get("mean")
        median = stats.get("median")
        abs_skew = abs(skew_val)

        if abs_skew <= _SKEW_MILD:
            return  # Symmetric - nothing to flag

        direction = "right (positive)" if skew_val > 0 else "left (negative)"
        tail_dir = "higher" if skew_val > 0 else "lower"
        bulk_dir = "lower" if skew_val > 0 else "higher"
        dir_word = "Right" if skew_val > 0 else "Left"

        metric: dict[str, float] = {"skewness": round(skew_val, 4)}
        if mean is not None:
            metric["mean"] = round(mean, 4)
        if median is not None:
            metric["median"] = round(median, 4)

        if abs_skew <= _SKEW_MOD:
            # ── Mild skew ─────────────────────────────────────────────────────
            level = "info"
            title = f"Mild {dir_word} Skew ({skew_val:+.2f})"

            eda_body = (
                f"{col} has a mild {direction} skew (skewness = {skew_val:.2f}). "
                f"The distribution is slightly asymmetric, with a minor lean toward "
                f"{tail_dir} values. The mean and median are close, so either is a "
                f"reasonable summary of the center. "
                f"Glance at the histogram to confirm no unusual clustering or gaps "
                f"are hiding behind the mild asymmetry."
            )

            ml_body = (
                f"{col} has mild skewness ({skew_val:.2f}). Most models will handle "
                f"this without transformation. If using linear or distance-based "
                f"models, monitor residuals after fitting - a transform may help "
                f"marginally. Tree-based models are unaffected."
            )
            ml_actions = [
                {
                    "action": "monitor",
                    "column": col,
                    "detail": "Check residual plots after fitting linear models",
                }
            ]

        elif abs_skew <= _SKEW_HEAVY:
            # ── Moderate skew ─────────────────────────────────────────────────
            level = "warn"
            title = f"Moderate {dir_word} Skew ({skew_val:+.2f})"

            eda_body = (
                f"{col} shows moderate {direction} skewness (skewness = {skew_val:.2f})"
                f". Values are concentrated toward the {bulk_dir} end of the range, "
                f"with a tail extending toward {tail_dir} values. "
                f"The median is likely more descriptive than the mean here. "
                f"Check for outliers or data quality issues to confirm the skew "
                f"is genuine rather than driven by a small number of anomalous values."
            )

            ml_body = (
                f"Skewness of {skew_val:.2f} in {col} will affect linear models "
                "(linear/logistic regression) and distance-based models (SVM, KNN) "
                "by distorting coefficient scale and distance calculations. "
                "Tree-based models (Random Forest, XGBoost) are largely robust. "
                "A transform is recommended before fitting linear or distance-based "
                "models."
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
            # ── Heavy skew ────────────────────────────────────────────────────
            level = "warn"
            title = f"Heavy {dir_word} Skew ({skew_val:+.2f})"

            eda_body = (
                f"{col} has heavy {direction} skewness (skewness = {skew_val:.2f}). "
                f"The bulk of values cluster near the {bulk_dir} end with a long "
                f"tail extending toward {tail_dir} values. "
                f"The mean is significantly distorted by the tail - the median is "
                f"a much more honest description of the typical value. "
                f"Inspect the tail values directly: check whether they represent "
                f"genuine data, outliers, or data entry errors before drawing "
                f"conclusions about this column."
            )

            ml_body = (
                f"Heavy skewness of {skew_val:.2f} in {col} will significantly distort "
                "linear model coefficients and KNN/SVM distance calculations. "
                "Tree-based models are robust but may still benefit from "
                "transformation for interpretability. Transformation is strongly "
                "recommended before using any non-tree model."
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

        # EDA blurb - descriptive only, no actions
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

        # ML blurb - prescriptive, structured actions
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
