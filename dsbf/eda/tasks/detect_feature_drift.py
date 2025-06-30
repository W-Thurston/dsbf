# dsbf/eda/tasks/detect_feature_drift.py

from typing import Optional

import numpy as np
import polars as pl
from scipy.stats import chi2_contingency, ks_2samp

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_text_polars


@register_task(
    name="detect_feature_drift",
    display_name="Detect Feature Drift",
    description=(
        "Detects distributional drift between current and "
        "reference datasets for shared columns."
    ),
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    tags=["drift", "comparison"],
)
class DetectFeatureDrift(BaseTask):
    def run(self) -> None:
        try:
            ctx = self.context
            df: pl.DataFrame = self.input_data
            reference: Optional[pl.DataFrame] = getattr(ctx, "reference_data", None)
            if reference is None:
                self.output = TaskResult(
                    name=self.name,
                    status="skipped",
                    summary={"message": "Reference dataset not provided in context."},
                )
                return

            shared_cols = [col for col in df.columns if col in reference.columns]
            if not shared_cols:
                self.output = TaskResult(
                    name=self.name,
                    status="skipped",
                    summary={
                        "message": (
                            "No shared columns between df and" " reference datasets."
                        )
                    },
                )
                return

            # Thresholds
            psi_threshold = float(self.get_task_param("psi") or 0.1)
            # ks_p_threshold = float(self.get_task_param("ks_pvalue") or 0.05)
            tvd_threshold = float(self.get_task_param("tvd") or 0.2)
            # chi2_p_threshold = float(self.get_task_param("chi2_pvalue") or 0.05)

            drift_results = {}
            numeric_cols, categorical_cols = [], []

            for col in shared_cols:
                try:
                    current_col = df.get_column(col)
                    reference_col = reference.get_column(col)

                    # Check numeric
                    if (
                        hasattr(current_col.dtype, "is_numeric")
                        and current_col.dtype.is_numeric()
                        and hasattr(reference_col.dtype, "is_numeric")
                        and reference_col.dtype.is_numeric()
                    ):

                        numeric_cols.append(col)

                        cur_np = current_col.drop_nulls().to_numpy()
                        ref_np = reference_col.drop_nulls().to_numpy()

                        if len(cur_np) == 0 or len(ref_np) == 0:
                            drift_results[col] = {
                                "type": "numerical",
                                "error": "Empty array after null removal",
                            }
                            continue

                        psi = compute_psi(ref_np, cur_np)
                        ks_stat, ks_p = ks_2samp(ref_np, cur_np)
                        severity = get_severity(psi, psi_threshold)

                        drift_results[col] = {
                            "type": "numerical",
                            "psi": round(psi, 4),
                            "ks_pvalue": round(float(ks_p), 4),  # type: ignore
                            "severity": severity,
                        }

                    # Otherwise treat as categorical
                    elif is_text_polars(current_col) and is_text_polars(reference_col):
                        categorical_cols.append(col)

                        cur_vals = current_col.drop_nulls().cast(str).value_counts()
                        ref_vals = reference_col.drop_nulls().cast(str).value_counts()

                        col_name, count_name = cur_vals.columns

                        cur_dict = {
                            row[col_name]: row[count_name]
                            for row in cur_vals.iter_rows(named=True)
                        }
                        ref_dict = {
                            row[col_name]: row[count_name]
                            for row in ref_vals.iter_rows(named=True)
                        }

                        all_keys = set(cur_dict.keys()) | set(ref_dict.keys())
                        total_cur = sum(cur_dict.values())
                        total_ref = sum(ref_dict.values())

                        tvd = 0.5 * sum(
                            abs(
                                (cur_dict.get(k, 0) / total_cur)
                                - (ref_dict.get(k, 0) / total_ref)
                            )
                            for k in all_keys
                        )

                        # Chi-squared
                        table = [
                            [cur_dict.get(k, 0) for k in all_keys],
                            [ref_dict.get(k, 0) for k in all_keys],
                        ]
                        _, chi2_p, _, _ = chi2_contingency(table)

                        severity = get_severity(tvd, tvd_threshold)

                        drift_results[col] = {
                            "type": "categorical",
                            "tvd": round(tvd, 4),
                            "chi2_pvalue": round(float(chi2_p), 4),  # type: ignore
                            "severity": severity,
                        }
                    else:
                        drift_results[col] = {
                            "type": "unsupported",
                            "error": f"Column '{col}' is neither numeric nor string",
                        }

                except Exception as e:
                    drift_results[col] = {
                        "type": "unknown",
                        "error": str(e),
                    }

            high_drift_cols = [
                col
                for col, res in drift_results.items()
                if res.get("severity") == "high"
            ]

            recommendations = []
            if high_drift_cols:
                recommendations.append(
                    f"High drift detected in columns: {high_drift_cols}."
                    " Consider reviewing data pipeline or retraining model."
                )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "total_columns_evaluated": len(shared_cols),
                    "numeric_columns_checked": len(numeric_cols),
                    "categorical_columns_checked": len(categorical_cols),
                    "high_drift_columns": high_drift_cols,
                },
                data=drift_results,
                recommendations=recommendations,
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)


def compute_psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    combined_min = min(ref.min(), cur.min())
    combined_max = max(ref.max(), cur.max())
    ref_percents, _ = np.histogram(
        ref, bins=bins, range=(combined_min, combined_max), density=True
    )
    cur_percents, _ = np.histogram(
        cur, bins=bins, range=(combined_min, combined_max), density=True
    )
    ref_percents = np.where(ref_percents == 0, 1e-6, ref_percents)
    cur_percents = np.where(cur_percents == 0, 1e-6, cur_percents)
    return float(
        np.sum((ref_percents - cur_percents) * np.log(ref_percents / cur_percents))
    )


def get_severity(value: float, threshold: float) -> str:
    if value < threshold:
        return "low"
    elif value < 2 * threshold:
        return "moderate"
    return "high"
