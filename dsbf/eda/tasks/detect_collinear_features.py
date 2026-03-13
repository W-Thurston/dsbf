# dsbf/eda/tasks/detect_collinear_features.py

import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import (
    TaskResult,
    add_reliability_warning,
    make_failure_result,
)
from dsbf.utils.backend import is_polars
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    display_name="Detect Collinear Features",
    description="Detects highly collinear features that may cause multicollinearity.",
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="modeling",
    domain="core",
    runtime_estimate="slow",
    tags=["multicollinearity", "numeric"],
    expected_semantic_types=["continuous"],
)
class DetectCollinearFeatures(BaseTask):
    def run(self) -> None:
        try:
            df = self.input_data
            flags = self.ensure_reliability_flags()

            vif_threshold = float(self.get_task_param("vif_threshold") or 10.0)

            if is_polars(df):
                self._log(
                    "    Falling back to Pandas: VIF calculation requires NumPy arrays",
                    "debug",
                )
                df = df.to_pandas()

            # Use semantic typing to select relevant columns
            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'continuous' column(s)",
                "debug",
            )
            numeric_df = df.select_dtypes(include=np.number).dropna()

            if numeric_df.shape[1] < 2:
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    summary={"message": "Not enough numeric features to compute VIF."},
                    data={"vif_scores": {}, "collinear_columns": []},
                    metadata={"vif_threshold": vif_threshold},
                )
                return

            vif_scores: dict[str, float] = {}
            for i in range(numeric_df.shape[1]):
                col = numeric_df.columns[i]
                vif_val = variance_inflation_factor(numeric_df.values, i)
                vif_scores[col] = float(vif_val)

            collinear_columns: list[str] = [
                col for col, vif in vif_scores.items() if vif > vif_threshold
            ]

            result = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Flagged {len(collinear_columns)} "
                        f"column(s) with VIF > {vif_threshold}."
                    ),
                },
                data={
                    "vif_scores": vif_scores,
                    "collinear_columns": collinear_columns,
                },
                metadata={
                    "vif_threshold": vif_threshold,
                    "suggested_viz_type": "heatmap",
                    "recommended_section": "Multicollinearity",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            # Generate guidance for every column with a meaningful VIF score
            vif_info = 5.0  # moderate correlation - worth knowing
            for col, vif in vif_scores.items():
                if vif >= vif_info:
                    self._attach_guidance(col, vif, vif_threshold)

            # Reliability warnings
            if flags["low_row_count"]:
                add_reliability_warning(
                    result,
                    level="heuristic_caution",
                    code="vif_low_n",
                    description=(
                        "VIF values may be unstable when sample size is small (N < 30)."
                    ),
                    recommendation=(
                        "Consider bootstrapping or collecting"
                        " more data before interpreting VIF."
                    ),
                )
            if flags["zero_variance_cols"]:
                add_reliability_warning(
                    result,
                    level="strong_warning",
                    code="vif_zero_variance",
                    description=(
                        "Some features have near-zero variance,"
                        " which can distort VIF calculations."
                    ),
                    recommendation=(
                        "Drop or transform zero-variance features before running VIF."
                    ),
                )

            # Apply ML scoring to self.output
            if (
                self.get_engine_param("enable_impact_scoring", True)
                and collinear_columns
            ):
                top_col: str = collinear_columns[0]
                top_vif: float = vif_scores[top_col]
                score: float = 0.75 if top_vif < 15 else 0.85
                tip: str | None = get_recommendation_tip(self.name, {"vif": top_vif})
                self.set_ml_signals(
                    result=result,
                    score=score,
                    tags=["drop", "transform"],
                    recommendation=tip
                    or (
                        f"Column '{top_col}' has high multicollinearity"
                        f" (VIF = {top_vif:.2f}). "
                        "Consider dropping this feature or applying regularization/PCA."
                    ),
                )
                result.summary["column"] = top_col

            self.output = result

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, vif: float, vif_threshold: float) -> None:
        """
        Generate EDA + ML guidance for a column with a notable VIF score.

        VIF tiers (standard statistical convention):
        - 5-10: moderate multicollinearity - info
        - 10-20: high - warn
        - >20:  severe - error
        """
        vif_str: str = f"{vif:.1f}"

        if vif >= 20:
            level = "error"
            tier = "severe"
            eda_interp: str = (
                f"A VIF of {vif_str} means {col} shares most of its variance with "
                f"other features in the dataset. There is very little independent "
                f"information in this column that is not already captured elsewhere."
            )
        elif vif >= 10:
            level = "warn"
            tier = "high"
            eda_interp = (
                f"A VIF of {vif_str} indicates strong linear overlap between {col} "
                f"and at least one other numeric feature. The column has limited "
                "independent variation beyond what the correlated features "
                "already carry."
            )
        else:
            level = "info"
            tier = "moderate"
            eda_interp = (
                f"A VIF of {vif_str} indicates moderate correlation between {col} "
                f"and other numeric features. Some redundancy is present, but the "
                f"column still carries meaningful independent signal."
            )

        eda_body: str = (
            f"{col} has a Variance Inflation Factor of {vif_str} - {tier} "
            "multicollinearity. "
            f"{eda_interp} "
            f"Check the correlation matrix (Relationships tab) to identify which "
            f"features are most strongly associated with {col}."
        )

        ml_body: str = (
            f"{col} has VIF = {vif_str}. "
            f"""{
                "Linear models (OLS regression, logistic regression, linear SVM) will"
                " be most affected: "
                "coefficient estimates become unstable and their standard errors"
                " inflate, "
                "making feature importance and hypothesis tests unreliable."
                if vif >= 10
                else "Linear models may show coefficient instability for this "
                "feature - "
                "interpret its coefficient cautiously in any regularised or "
                "unregularised regression."
            } """
            "Tree-based models (Random Forest, Gradient Boosting) are not directly "
            "harmed by multicollinearity, but feature importance scores will be split "
            "between the correlated columns rather than concentrated on one. "
            f"""{
                "Dropping the weaker of the collinear pair, applying ridge/elastic-net"
                "regularisation, or using PCA to orthogonalise"
                " the feature space are the standard remedies."
                if vif >= 10
                else "Consider whether this column adds meaningful "
                "signal beyond its correlated neighbours "
                "before including it in a linear model."
            }
            """
        )

        ml_actions: list[str] = []
        if vif >= 10:
            ml_actions = [
                {
                    "action": "drop",
                    "column": col,
                    "detail": "Drop the weaker of the collinear pair after "
                    "inspecting correlation matrix",
                },
                {
                    "action": "regularise",
                    "method": "ridge_or_elastic_net",
                    "detail": "Regularisation stabilises coefficients under "
                    "multicollinearity",
                },
                {
                    "action": "transform",
                    "method": "pca",
                    "detail": "PCA produces orthogonal components - eliminates "
                    "multicollinearity entirely",
                },
            ]
        else:
            ml_actions = [
                {
                    "action": "monitor",
                    "column": col,
                    "detail": "Check correlation matrix to confirm which feature is"
                    " the source of overlap",
                },
            ]

        metric: dict[str, float] = {
            "vif": round(vif, 4),
            "vif_threshold": vif_threshold,
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=f"{tier.title()} Multicollinearity (VIF = {vif_str})",
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )
        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=level,
            title=f"VIF = {vif_str} - Collinearity Impact on Models",
            body=ml_body.strip(),
            actions=ml_actions,
            metric=metric,
        )
