# dsbf/eda/tasks/detect_outliers.py
#
# Consolidated outlier detection task. Supersedes outlier_detection_mad.py.
#
# Methods:
#   iqr              — Tukey IQR fences (Q1 - 1.5xIQR, Q3 + 1.5xIQR)
#   zscore           — Standard Z-score (|z| > threshold, default 3.0)
#   mad              — Modified Z-score / MAD (Iglewicz & Hoaglin 1993)
#   isolation_forest — Isolation Forest multivariate anomaly detection
#   all              — Run iqr + zscore + mad on every column; isolation_forest
#                      when enabled (default True when method="all")
#
# Output data structure:
#   Column-keyed entries (one per numeric column):
#       {
#           "iqr":    { outlier_count, outlier_pct, lower_fence, upper_fence,
#                       q1, q3, iqr_val, outlier_row_indices },
#           "zscore": { outlier_count, outlier_pct, threshold, max_zscore },
#           "mad":    { outlier_count, outlier_pct, threshold, median, mad,
#                       max_modified_z_score, top_outlier_values },
#           "methods_flagging": ["iqr", "mad"],  # methods that found outliers
#           "consensus": bool,   # True if >= 2 univariate methods agree
#           "n": int,
#       }
#   __dataset__ sentinel key (multivariate / row-level):
#       {
#           "isolation_forest": {
#               "contamination": float,
#               "n_flagged": int,
#               "flagged_row_indices": list[int],
#               "n_features_used": int,
#               "feature_columns": list[str],
#           }
#       }

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

with contextlib.suppress(ImportError):
    from sklearn.ensemble import IsolationForest

if TYPE_CHECKING:
    from pandas import Series

# ── Per-method scoring helpers ────────────────────────────────────────────────


def _iqr_outliers(
    series: pd.Series,
    n_rows: int,
) -> dict[str, Any]:
    """
    Detect outliers using Tukey IQR fences.

    Args:
        series: Non-null numeric Series.
        n_rows: Total row count (for outlier_pct denominator).

    Returns:
        Dict with outlier_count, outlier_pct, lower_fence, upper_fence,
        q1, q3, iqr_val, outlier_row_indices.

    """
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr_val: float = q3 - q1
    lower: float = q1 - 1.5 * iqr_val
    upper: float = q3 + 1.5 * iqr_val
    mask: Series[bool] = (series < lower) | (series > upper)
    count = int(mask.sum())
    return {
        "outlier_count": count,
        "outlier_pct": round(count / n_rows, 4),
        "lower_fence": round(lower, 6),
        "upper_fence": round(upper, 6),
        "q1": round(q1, 6),
        "q3": round(q3, 6),
        "iqr_val": round(iqr_val, 6),
        "outlier_row_indices": series[mask].index.tolist(),
    }


def _zscore_outliers(
    series: pd.Series,
    threshold: float,
    n_rows: int,
) -> dict[str, Any] | None:
    """
    Detect outliers using the standard Z-score.

    Returns None when std is zero (constant column).

    Args:
        series: Non-null numeric Series.
        threshold: |z| above which a value is flagged. Default: 3.0.
        n_rows: Total row count (for outlier_pct denominator).

    Returns:
        Dict with outlier_count, outlier_pct, threshold, max_zscore, or None.

    """
    std = float(series.std())
    if std == 0:
        return None
    mean = float(series.mean())
    zscores = ((series - mean) / std).abs()
    mask = zscores > threshold
    count = int(mask.sum())
    return {
        "outlier_count": count,
        "outlier_pct": round(count / n_rows, 4),
        "threshold": threshold,
        "max_zscore": round(float(zscores.max()), 4),
    }


def _mad_outliers(
    series: pd.Series,
    threshold: float,
    n_rows: int,
) -> dict[str, Any] | None:
    """
    Detect outliers using the Modified Z-score (MAD method).

    Returns None when MAD is zero (near-constant column).

    Args:
        series: Non-null numeric Series.
        threshold: |MZS| above which a value is flagged. Default: 3.5.
        n_rows: Total row count (for outlier_pct denominator).

    Returns:
        Dict with outlier_count, outlier_pct, threshold, median, mad,
        max_modified_z_score, top_outlier_values, or None.

    """
    median = float(series.median())
    mad = float((series - median).abs().median())
    if mad == 0:
        return None
    scores = (0.6745 * (series - median) / mad).abs()
    mask = scores > threshold
    count = int(mask.sum())
    top_vals = series[mask].reindex(scores[mask].nlargest(10).index).tolist()
    return {
        "outlier_count": count,
        "outlier_pct": round(count / n_rows, 4),
        "threshold": threshold,
        "median": round(median, 6),
        "mad": round(mad, 6),
        "max_modified_z_score": round(float(scores.max()), 4),
        "top_outlier_values": [round(float(v), 6) for v in top_vals],
    }


def _isolation_forest_outliers(
    df: pd.DataFrame,
    continuous_cols: list[str],
    contamination: float,
    random_state: int,
) -> dict[str, Any] | None:
    """
    Detect multivariate outliers using Isolation Forest.

    Operates on the joint distribution of all continuous columns. A row
    can be normal on every individual axis but anomalous in the joint
    distribution — this method catches such cases.

    Args:
        df: Full DataFrame (pandas).
        continuous_cols: Names of continuous columns to use as features.
        contamination: Expected proportion of outliers in the data.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with contamination, n_flagged, flagged_row_indices,
        n_features_used, feature_columns, or None if fewer than 2
        usable columns or fewer than 10 complete rows.

    """
    # Select columns with sufficient non-null values
    usable: list[str] = [col for col in continuous_cols if df[col].notna().sum() >= 10]
    if len(usable) < 2:
        return None

    X: Series = df[usable].copy()
    # Impute nulls with column medians for the IF pass
    for col in usable:
        median = X[col].median()
        X[col] = X[col].fillna(median if not np.isnan(median) else 0.0)

    complete_mask = X.notna().all(axis=1)
    X_complete = X[complete_mask]
    if len(X_complete) < 10:
        return None

    clf = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    labels = clf.fit_predict(X_complete.values)  # -1 = outlier, 1 = inlier
    flagged_local = np.where(labels == -1)[0]
    flagged_indices = X_complete.iloc[flagged_local].index.tolist()

    return {
        "contamination": contamination,
        "n_flagged": len(flagged_indices),
        "flagged_row_indices": flagged_indices,
        "n_features_used": len(usable),
        "feature_columns": usable,
    }


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="detect_outliers",
    display_name="Detect Outliers",
    description=(
        "Unified outlier detection: IQR fences, standard Z-score, Modified "
        "Z-score (MAD), and Isolation Forest (multivariate). Runs all univariate "
        "methods by default and synthesises a consensus flag per column. "
        "Supersedes outlier_detection_mad."
    ),
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    domain="core",
    runtime_estimate="moderate",
    phase="eda",
    tags=["outliers", "numeric", "robust", "mad", "isolation_forest"],
    expected_semantic_types=["continuous"],
)
class DetectOutliers(BaseTask):
    """
    Unified outlier detection across IQR, Z-score, MAD, and Isolation Forest.

    Runs all univariate methods on each continuous column in a single pass and
    synthesises a **consensus** flag: True when at least two methods independently
    identify a column as having outliers. The consensus view is more reliable
    than any single method because:

    - IQR is simple and distribution-free but sensitive to heavy tails.
    - Z-score is corrupted by the very outliers it detects (inflated std).
    - MAD is robust to both heavy tails and existing outliers, but undefined
      for near-constant columns.

    A column flagged by two or three methods is very likely genuinely problematic;
    a column flagged by only one method warrants investigation but not alarm.

    **Isolation Forest** (``method="all"`` or ``method="isolation_forest"``) runs
    on the joint distribution of all continuous columns. Row-level findings are
    stored under the ``__dataset__`` sentinel key since they cannot be attributed
    to a single column. This catches multivariate anomalies invisible to univariate
    methods — a row can be normal on every individual axis but anomalous in the
    joint distribution.

    **Output structure:**

    Column entries (keyed by column name)::

        {
            "iqr":    { outlier_count, outlier_pct, lower_fence, upper_fence,
                        q1, q3, iqr_val, outlier_row_indices },
            "zscore": { outlier_count, outlier_pct, threshold, max_zscore },
            "mad":    { outlier_count, outlier_pct, threshold, median, mad,
                        max_modified_z_score, top_outlier_values },
            "methods_flagging": list[str],
            "consensus": bool,
            "n": int,
        }

    Dataset-level entry::

        "__dataset__": {
            "isolation_forest": {
                "contamination": float,
                "n_flagged": int,
                "flagged_row_indices": list[int],
                "n_features_used": int,
                "feature_columns": list[str],
            }
        }

    Configurable parameters (via config["tasks"]["detect_outliers"]):
        method (str): ``"all"`` (default), ``"iqr"``, ``"zscore"``, ``"mad"``,
            or ``"isolation_forest"``.
        flag_threshold (float): Minimum outlier proportion for a column to be
            counted as having outliers in the summary. Default: 0.01
        zscore_threshold (float): |z| threshold for Z-score method. Default: 3.0
        mad_threshold (float): |MZS| threshold for MAD method. Default: 3.5
        min_n (int): Minimum non-null values to analyse a column. Default: 10
        run_isolation_forest (bool): Whether to run Isolation Forest when
            method="all". Default: True
        contamination (float): Expected outlier proportion for Isolation Forest.
            Default: 0.05
        random_state (int): Random seed for Isolation Forest. Default: 42
    """

    def run(self) -> None:
        """
        Execute outlier detection and populate self.output.

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

            method_raw: Any | None = self.get_task_param("method")
            method: str = str(method_raw) if method_raw is not None else "all"

            flag_thresh_raw: Any | None = self.get_task_param("flag_threshold")
            flag_threshold: float = (
                float(flag_thresh_raw) if flag_thresh_raw is not None else 0.01
            )

            zscore_thresh_raw: Any | None = self.get_task_param("zscore_threshold")
            zscore_threshold: float = (
                float(zscore_thresh_raw) if zscore_thresh_raw is not None else 3.0
            )

            mad_thresh_raw: Any | None = self.get_task_param("mad_threshold")
            mad_threshold: float = (
                float(mad_thresh_raw) if mad_thresh_raw is not None else 3.5
            )

            min_n_raw: Any | None = self.get_task_param("min_n")
            min_n: int = int(min_n_raw) if min_n_raw is not None else 10

            run_if_raw: Any | None = self.get_task_param("run_isolation_forest")
            run_isolation_forest: bool = (
                str(run_if_raw).lower() not in ("false", "0", "no")
                if run_if_raw is not None
                else True
            )

            contamination_raw: Any | None = self.get_task_param("contamination")
            contamination: float = (
                float(contamination_raw) if contamination_raw is not None else 0.05
            )

            random_state_raw: Any | None = self.get_task_param("random_state")
            random_state: int = (
                int(random_state_raw) if random_state_raw is not None else 42
            )

            run_iqr: bool = method in ("all", "iqr")
            run_zscore: bool = method in ("all", "zscore")
            run_mad: bool = method in ("all", "mad")
            run_if: bool = method == "isolation_forest" or (
                method == "all" and run_isolation_forest
            )

            n_rows: int = len(df)
            numeric_df = df.select_dtypes(include=np.number)
            results: dict[str, dict[str, Any]] = {}
            continuous_cols: list[str] = []

            for col in numeric_df.columns:
                series = numeric_df[col].dropna()
                if len(series) < min_n:
                    self._log(
                        f"    '{col}' skipped: only {len(series)} non-null values.",
                        "debug",
                    )
                    continue

                continuous_cols.append(col)
                entry: dict[str, Any] = {"n": len(series)}

                if run_iqr:
                    entry["iqr"] = _iqr_outliers(series, n_rows)

                if run_zscore:
                    zs: dict[str, Any] | None = _zscore_outliers(
                        series,
                        zscore_threshold,
                        n_rows,
                    )
                    if zs is not None:
                        entry["zscore"] = zs

                if run_mad:
                    mad: dict[str, Any] | None = _mad_outliers(
                        series,
                        mad_threshold,
                        n_rows,
                    )
                    if mad is not None:
                        entry["mad"] = mad

                # Consensus: methods that found at least one outlier
                methods_flagging: list[str] = [
                    m
                    for m in ("iqr", "zscore", "mad")
                    if m in entry and entry[m]["outlier_count"] > 0
                ]
                entry["methods_flagging"] = methods_flagging
                entry["consensus"] = len(methods_flagging) >= 2

                results[col] = entry
                self._log(
                    f"    '{col}': flagged by {methods_flagging} "
                    f"({'consensus' if entry['consensus'] else 'no consensus'})",
                    "debug",
                )

            # Isolation Forest — dataset-level, __dataset__ sentinel key
            dataset_entry: dict[str, Any] = {}
            if run_if and continuous_cols:
                self._log(
                    f"    Running Isolation Forest on {len(continuous_cols)} "
                    "continuous column(s).",
                    "debug",
                )
                if_result: dict[str, Any] | None = _isolation_forest_outliers(
                    df,
                    continuous_cols,
                    contamination,
                    random_state,
                )
                if if_result is not None:
                    dataset_entry["isolation_forest"] = if_result
                    self._log(
                        "    Isolation Forest: "
                        f"{if_result['n_flagged']} row(s) flagged.",
                        "debug",
                    )
                else:
                    self._log(
                        "    Isolation Forest skipped: "
                        "insufficient data or sklearn unavailable.",
                        "debug",
                    )

            cols_with_outliers: int = sum(
                1
                for v in results.values()
                if any(
                    v.get(m, {}).get("outlier_pct", 0) >= flag_threshold
                    for m in ("iqr", "zscore", "mad")
                )
            )
            consensus_cols: int = sum(1 for v in results.values() if v["consensus"])

            data: dict[str, Any] = {**results}
            if dataset_entry:
                data["__dataset__"] = dataset_entry

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Outlier detection on {len(results)} column(s); "
                        f"{cols_with_outliers} with outliers, "
                        f"{consensus_cols} with method consensus."
                    ),
                    "columns_analysed": len(results),
                    "columns_with_outliers": cols_with_outliers,
                    "consensus_columns": consensus_cols,
                    "method": method,
                },
                data=data,
                metadata={
                    "method": method,
                    "flag_threshold": flag_threshold,
                    "zscore_threshold": zscore_threshold,
                    "mad_threshold": mad_threshold,
                    "run_isolation_forest": run_if,
                    "contamination": contamination if run_if else None,
                    "suggested_viz_type": "boxplot",
                    "recommended_section": "Distributions",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            # Guidance: column-level
            for col, entry in results.items():
                any_outliers: bool = any(
                    entry.get(m, {}).get("outlier_count", 0) > 0
                    for m in ("iqr", "zscore", "mad")
                )
                if any_outliers:
                    self._attach_column_guidance(col, entry, n_rows)

            # Guidance: dataset-level Isolation Forest
            if "isolation_forest" in dataset_entry:
                self._attach_if_guidance(dataset_entry["isolation_forest"])

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_column_guidance(
        self,
        col: str,
        entry: dict[str, Any],
        n_rows: int,
    ) -> None:
        """
        Generate EDA and ML guidance for a column with outliers.

        Args:
            col: Column name.
            entry: Column result dict containing method sub-dicts.
            n_rows: Total row count.

        """
        methods_flagging = entry.get("methods_flagging", [])
        consensus = entry.get("consensus", False)
        n = entry["n"]

        # Use the highest outlier count across methods for primary display
        max_count = 0
        primary_method = None
        for m in ("mad", "iqr", "zscore"):
            c = entry.get(m, {}).get("outlier_count", 0)
            if c > max_count:
                max_count = c
                primary_method: str = m

        if max_count == 0 or primary_method is None:
            return

        pct = max_count / n_rows
        level: str = "warn" if pct >= 0.05 or consensus else "info"

        # Build method summary string
        method_parts: list[str] = []
        for m in ("iqr", "zscore", "mad"):
            d: Any | None = entry.get(m)
            if d and d["outlier_count"] > 0:
                label: str = {"iqr": "IQR", "zscore": "Z-score", "mad": "MAD"}[m]
                method_parts.append(
                    f"{label}: {d['outlier_count']} ({d['outlier_pct']:.1%})",
                )

        status = (
            "consensus — high confidence"
            if consensus
            else "single method only — verify"
        )
        consensus_note: str = (
            f" Flagged by {len(methods_flagging)} of 3 methods ({status})."
        )

        eda_body: str = (
            f"'{col}' has outliers detected across the following methods: "
            f"{'; '.join(method_parts)}.{consensus_note} "
            f"Investigate whether extreme values are genuine rare events, "
            f"data entry errors, or a distinct sub-population. Cross-method "
            f"agreement increases confidence that these are real anomalies "
            f"rather than artefacts of any single method's assumptions."
        )

        ml_body: str = (
            f"'{col}' has outliers confirmed by {len(methods_flagging)} detection "
            f"method(s). Distance-based models (KNN, SVM), linear models, and PCA "
            f"are sensitive to extreme values. Consider Winsorising at the "
            f"1st/99th percentile or applying a RobustScaler before training. "
            f"Tree-based models are largely unaffected by feature outliers but "
            f"remain sensitive when outliers appear in the target variable."
        )

        iqr_bounds = entry.get("iqr", {})
        lower = iqr_bounds.get("lower_fence")
        upper = iqr_bounds.get("upper_fence")

        actions: list[dict[str, str]] = [
            {
                "action": "winsorize",
                "column": col,
                "detail": (
                    f"Cap values at IQR fence [{lower:.4g}, {upper:.4g}]"
                    if lower is not None and upper is not None
                    else "Cap at IQR fence"
                ),
            },
            {
                "action": "robust_scale",
                "method": "RobustScaler",
                "column": col,
                "detail": "Scale using median and IQR instead of mean and std",
            },
        ]

        metric: dict = {
            "methods_flagging": methods_flagging,
            "consensus": consensus,
            "max_outlier_count": max_count,
            "outlier_pct": round(pct, 4),
            "n": n,
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=(
                f"Outliers: {len(methods_flagging)} method(s) agree "
                f"({'consensus' if consensus else 'verify'})"
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
            title=f"Outlier Sensitivity ({len(methods_flagging)} method(s))",
            body=ml_body.strip(),
            actions=actions,
            metric=metric,
        )

    def _attach_if_guidance(self, if_result: dict[str, Any]) -> None:
        """
        Generate EDA guidance for Isolation Forest multivariate findings.

        Args:
            if_result: Isolation Forest result dict.

        """
        n_flagged = if_result["n_flagged"]
        contamination = if_result["contamination"]
        features = if_result["feature_columns"]
        n_features = if_result["n_features_used"]

        body: str = (
            f"Isolation Forest flagged {n_flagged} row(s) as multivariate "
            f"anomalies across {n_features} continuous feature(s) "
            f"(contamination={contamination}). "
            f"Features used: {features[:5]}"
            f"{'...' if len(features) > 5 else ''}. "
            f"These rows are anomalous in the joint distribution of features — "
            f"they may appear normal on any individual column but fall in a "
            f"low-density region of the feature space. "
            f"Review the flagged row indices in the data output. "
            f"Common causes: sensor fusion errors, merged records from different "
            f"populations, or genuine rare multivariate combinations."
        )

        self.add_guidance(
            result=self.output,
            column="__dataset__",
            phase="eda",
            level="info",
            title=f"Multivariate Anomalies: {n_flagged} row(s) (Isolation Forest)",
            body=body.strip(),
            actions=[
                {
                    "action": "review_flagged_rows",
                    "detail": (
                        f"Inspect the {n_flagged} flagged row indices in "
                        "data['__dataset__']['isolation_forest']['flagged_row_indices']"
                    ),
                },
            ],
            metric={
                "n_flagged": n_flagged,
                "contamination": contamination,
                "n_features_used": n_features,
            },
        )
