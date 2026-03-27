# dsbf/eda/tasks/transformation_preview.py

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew
from sklearn.preprocessing import PowerTransformer

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

if TYPE_CHECKING:
    from pandas import Series

# ── Transform helpers ─────────────────────────────────────────────────────────


def _apply_yeo_johnson(series: pd.Series) -> pd.Series:
    """
    Apply Yeo-Johnson power transform to a Series.

    Fits the optimal lambda on the input data using sklearn's PowerTransformer.

    Args:
        series: Non-null numeric Series.

    Returns:
        Transformed Series with the same index.

    """
    values = series.dropna().to_numpy().reshape(-1, 1)
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    transformed = pt.fit_transform(values).flatten()
    result: Series = series.copy()
    result[series.notna()] = transformed
    return result


# ── Transform definitions ─────────────────────────────────────────────────────
#
# precondition_fn(series) → bool  - True if the transform is safe to apply
# apply_fn(series) → pd.Series    - applies the transform and returns result

_TRANSFORMS: list[dict] = [
    {
        "name": "log1p",
        "display": "log1p (log(1 + x))",
        "apply": lambda s: np.log1p(s),
        "precondition": lambda s: bool((s >= 0).all()),
        "precondition_note": "requires all values ≥ 0",
        "suitable_for": "right-skewed, non-negative values",
    },
    {
        "name": "sqrt",
        "display": "sqrt (√x)",
        "apply": lambda s: np.sqrt(s),
        "precondition": lambda s: bool((s >= 0).all()),
        "precondition_note": "requires all values ≥ 0",
        "suitable_for": "mild right skew, non-negative values",
    },
    {
        "name": "square",
        "display": "square (x²)",
        "apply": lambda s: np.square(s),
        "precondition": lambda s: True,
        "precondition_note": None,
        "suitable_for": "left-skewed distributions",
    },
    {
        "name": "yeo_johnson",
        "display": "Yeo-Johnson",
        "apply": _apply_yeo_johnson,
        "precondition": lambda s: len(s.dropna()) >= 3,
        "precondition_note": "requires ≥ 3 non-null values",
        "suitable_for": "any distribution including negative values",
    },
]


def _distribution_stats(series: pd.Series) -> dict[str, Any]:
    """
    Compute summary statistics describing a distribution's shape.

    Args:
        series: Non-null numeric Series.

    Returns:
        Dict with mean, median, std, skewness, min, max, p5, p95 keys.

    """
    clean: Series = series.dropna()
    if len(clean) < 3:
        return {}
    return {
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "std": round(float(clean.std()), 4),
        "skewness": round(float(scipy_skew(clean.values)), 4),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "p5": round(float(clean.quantile(0.05)), 4),
        "p95": round(float(clean.quantile(0.95)), 4),
    }


def _skew_reduction_pct(before: float, after: float) -> float:
    """
    Compute percentage reduction in absolute skewness.

    Args:
        before: Skewness before transform.
        after: Skewness after transform.

    Returns:
        Percentage reduction (positive = improvement), or 0.0 if before is 0.

    """
    if abs(before) == 0:
        return 0.0
    return round((abs(before) - abs(after)) / abs(before) * 100, 1)


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="transformation_preview",
    display_name="Transformation Preview",
    description=(
        "Computes before/after distribution statistics for recommended "
        "numeric transforms (log1p, sqrt, square, Yeo-Johnson). Lets the "
        "user evaluate transform effectiveness without applying them manually."
    ),
    depends_on=["infer_types", "detect_skewness"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["transformation", "distribution", "numeric", "preview"],
    expected_semantic_types=["continuous"],
)
class TransformationPreview(BaseTask):
    """
    Preview the effect of numeric transforms on skewed continuous columns.

    For each continuous column flagged as skewed by ``detect_skewness`` (or
    above ``skew_threshold`` when computed directly), applies a set of
    candidate transforms and computes before/after distribution statistics:

    - **log1p** - ``log(1 + x)``, for right-skewed non-negative values
    - **sqrt** - ``√x``, for mild right skew with non-negative values
    - **square** - ``x²``, for left-skewed distributions
    - **Yeo-Johnson** - power transform that handles any distribution
      including negative values; fits optimal lambda on the data

    Each transform result includes:
    - Before/after skewness, mean, median, std, min/max, 5th/95th percentiles
    - Skewness reduction percentage (how much the transform reduces |skewness|)
    - A ``recommended`` flag: True for the transform that achieves the greatest
      skewness reduction while meeting its preconditions

    Transforms that fail their precondition (e.g. log1p on a column with
    negative values) are skipped with an explanatory note rather than raising.

    EDA guidance is emitted for each column, ranking transforms by
    effectiveness and noting the preconditions for the recommended approach.

    This task is purely advisory - it never modifies the DataFrame.
    The actual transform should be applied by the user before modeling.

    Configurable parameters (via config["tasks"]["transformation_preview"]):
        skew_threshold (float): Absolute skewness above which a column is
            previewed. Default: 1.0
        min_n (int): Minimum non-null values required to attempt transforms.
            Default: 10
    """

    def run(self) -> None:
        """
        Execute transformation preview and populate self.output.

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

            skew_thresh_raw: Any | None = self.get_task_param("skew_threshold")
            skew_threshold: float = (
                float(skew_thresh_raw) if skew_thresh_raw is not None else 1.0
            )

            min_n_raw: Any | None = self.get_task_param("min_n")
            min_n: int = int(min_n_raw) if min_n_raw is not None else 10

            # --- Identify skewed columns ---
            # Prefer detect_skewness context result; fall back to direct computation
            skewed_cols: dict[str, float] = {}

            if self.context:
                skew_result: TaskResult | None = self.context.results.get(
                    "detect_skewness",
                )
                if skew_result and skew_result.status == "success":
                    for col, info in (skew_result.data or {}).items():
                        skew_val = (
                            info.get("skewness") if isinstance(info, dict) else info
                        )
                        if (
                            skew_val is not None
                            and abs(float(skew_val)) >= skew_threshold
                        ):
                            skewed_cols[col] = float(skew_val)

            if not skewed_cols:
                self._log(
                    "    No detect_skewness result in context - computing "
                    "skewness directly.",
                    "debug",
                )
                numeric_df = df.select_dtypes(include=np.number)
                for col in numeric_df.columns:
                    series = numeric_df[col].dropna()
                    if len(series) >= 3:
                        try:
                            sk = float(scipy_skew(series.values))
                            if abs(sk) >= skew_threshold:
                                skewed_cols[col] = sk
                        except Exception:
                            pass

            self._log(
                f"    {len(skewed_cols)} skewed column(s) to preview.",
                "debug",
            )

            previews: dict[str, dict] = {}

            for col, original_skew in skewed_cols.items():
                if col not in df.columns:
                    continue

                series = df[col].dropna()
                if len(series) < min_n:
                    self._log(
                        f"    '{col}' skipped: only {len(series)} non-null values.",
                        "debug",
                    )
                    continue

                before_stats: dict[str, Any] = _distribution_stats(series)
                transform_results: dict[str, dict] = {}

                for t in _TRANSFORMS:
                    t_name = t["name"]
                    try:
                        precondition_ok = t["precondition"](series)
                    except Exception:
                        precondition_ok = False

                    if not precondition_ok:
                        transform_results[t_name] = {
                            "skipped": True,
                            "skip_reason": t["precondition_note"],
                            "display": t["display"],
                            "suitable_for": t["suitable_for"],
                        }
                        continue

                    try:
                        transformed = t["apply"](series)
                        after_stats: dict[str, Any] = _distribution_stats(transformed)
                        skew_reduction: float = _skew_reduction_pct(
                            before_stats.get("skewness", 0.0),
                            after_stats.get("skewness", 0.0),
                        )
                        transform_results[t_name] = {
                            "skipped": False,
                            "display": t["display"],
                            "suitable_for": t["suitable_for"],
                            "precondition_note": t["precondition_note"],
                            "after_stats": after_stats,
                            "skew_reduction_pct": skew_reduction,
                            "recommended": False,  # set below
                        }
                    except Exception as e:
                        transform_results[t_name] = {
                            "skipped": True,
                            "skip_reason": f"computation failed: {type(e).__name__}",
                            "display": t["display"],
                            "suitable_for": t["suitable_for"],
                        }

                # Mark the best transform (highest skew reduction, not skipped)
                candidates: dict[str, dict] = {
                    name: v
                    for name, v in transform_results.items()
                    if not v.get("skipped", True)
                }
                if candidates:
                    best: tuple[str, dict] = max(
                        candidates.items(),
                        key=lambda x: x[1].get("skew_reduction_pct", -999),
                    )
                    transform_results[best[0]]["recommended"] = True

                previews[col] = {
                    "original_skewness": round(original_skew, 4),
                    "before_stats": before_stats,
                    "transforms": transform_results,
                    "n": len(series),
                }
                self._log(
                    f"    '{col}': skew={original_skew:.3f}, "
                    f"{len(candidates)} transform(s) computed.",
                    "debug",
                )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Transformation previews computed for "
                        f"{len(previews)} skewed column(s)."
                    ),
                    "previewed_count": len(previews),
                    "skew_threshold": skew_threshold,
                },
                data=previews,
                metadata={
                    "skew_threshold": skew_threshold,
                    "transforms_evaluated": [t["name"] for t in _TRANSFORMS],
                    "suggested_viz_type": "histogram",
                    "recommended_section": "Distributions",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, preview in previews.items():
                self._attach_guidance(col, preview)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, preview: dict) -> None:
        """
        Generate EDA guidance for a column's transformation preview.

        Args:
            col: Column name.
            preview: Preview dict with ``before_stats``, ``transforms``,
                ``original_skewness``, and ``n``.

        """
        original_skew = preview["original_skewness"]
        transforms = preview["transforms"]
        n = preview["n"]

        # Find the recommended transform
        recommended_name = next(
            (name for name, v in transforms.items() if v.get("recommended")),
            None,
        )

        if recommended_name is None:
            # All transforms were skipped - still emit an informational blurb
            self.add_guidance(
                result=self.output,
                column=col,
                phase="eda",
                level="info",
                title=(
                    f"Skewed Distribution - No Transform Applicable "
                    f"(skew={original_skew:.2f})"
                ),
                body=(
                    f"'{col}' is skewed (skewness={original_skew:.2f}, n={n:,}) "
                    f"but none of the candidate transforms could be applied "
                    f"(likely due to negative values preventing log1p and sqrt). "
                    f"Consider Yeo-Johnson transform, which handles negative values, "
                    f"or investigate whether the skew is driven by outliers that "
                    f"could be Winsorised first."
                ),
                actions=[],
                metric={"original_skewness": original_skew, "n": n},
            )
            return

        rec = transforms[recommended_name]
        after_skew = rec["after_stats"].get("skewness", 0.0)
        reduction = rec.get("skew_reduction_pct", 0.0)

        # Build a ranked summary of all applicable transforms
        ranked: list[tuple] = sorted(
            [(name, v) for name, v in transforms.items() if not v.get("skipped", True)],
            key=lambda x: -x[1].get("skew_reduction_pct", -999),
        )
        rank_lines: list[str] = []
        for rank, (name, v) in enumerate(ranked, 1):
            disp = v["display"]
            red = v.get("skew_reduction_pct", 0)
            after_sk = v["after_stats"].get("skewness", 0)
            star: str = " ★ recommended" if v.get("recommended") else ""
            rank_lines.append(
                f"  {rank}. {disp}: skewness {original_skew:.2f} → {after_sk:.2f} "
                f"({red:.0f}% reduction){star}",
            )
        ranking_str: str = "\n".join(rank_lines)

        eda_body: str = (
            f"'{col}' has a skewness of {original_skew:.2f} (n={n:,}). "
            f"The recommended transform is {rec['display']}, which reduces "
            f"absolute skewness by {reduction:.0f}% "
            f"({original_skew:.2f} → {after_skew:.2f}).\n\n"
            f"Transform comparison:\n{ranking_str}"
        )

        skipped_names: list[str] = [
            f"{v['display']} ({v.get('skip_reason', 'skipped')})"
            for name, v in transforms.items()
            if v.get("skipped")
        ]
        if skipped_names:
            eda_body += f"\n\nSkipped: {', '.join(skipped_names)}."

        precondition = rec.get("precondition_note")
        actions: list[dict[str, str]] = [
            {
                "action": "apply_transform",
                "transform": recommended_name,
                "method": rec["display"],
                "column": col,
                "detail": (
                    f"Reduces skewness by {reduction:.0f}%"
                    + (f"; {precondition}" if precondition else "")
                ),
            },
        ]

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="info",
            title=(
                f"Transform Preview: {rec['display']} reduces skew by "
                f"{reduction:.0f}% ({original_skew:.2f} → {after_skew:.2f})"
            ),
            body=eda_body.strip(),
            actions=actions,
            metric={
                "original_skewness": original_skew,
                "after_skewness": after_skew,
                "skew_reduction_pct": reduction,
                "recommended_transform": recommended_name,
                "n": n,
            },
        )
