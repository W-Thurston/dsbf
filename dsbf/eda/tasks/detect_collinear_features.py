# dsbf/eda/tasks/detect_collinear_features.py

import numpy as np
from pandas import DataFrame
from statsmodels.stats.outliers_influence import variance_inflation_factor

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import (
    TaskResult,
    add_reliability_warning,
    make_failure_result,
)
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    display_name="Detect Collinear Features",
    description="Detects highly collinear features that may cause multicollinearity.",
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="modeling",
    domain="core",
    runtime_estimate="slow",
    phase="ml_readiness",
    tags=["multicollinearity", "numeric"],
    expected_semantic_types=["continuous"],
)
class DetectCollinearFeatures(BaseTask):
    """
    Detects multicollinearity among numeric features using Variance Inflation Factor.

    Computes VIF for every numeric column. VIF quantifies how much of a feature's
    variance is explained by linear combinations of all other features - a VIF of 10
    means 90% of the column's variance is shared with others, making its coefficient
    highly unstable in linear models.

    VIF tiers (standard statistical convention):

    - 1 - 5: low multicollinearity - acceptable
    - 5 - 10: moderate - guidance emitted at info level
    - 10 - 20: high - guidance emitted at warn level
    - > 20: severe - guidance emitted at error level

    Polars DataFrames are converted to pandas since ``variance_inflation_factor``
    requires numpy arrays. Rows with any null are dropped before fitting to avoid
    VIF calculation failures.

    Configurable parameters (via config["tasks"]["detect_collinear_features"]):
        vif_threshold (float): VIF above which a column is added to
            ``collinear_columns``. Default: 10.0
    """

    def run(self) -> None:
        """
        Execute VIF computation and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df: DataFrame = self.get_dataframe_pandas()
            flags: dict = self.ensure_reliability_flags()

            vif_threshold = float(self.get_task_param("vif_threshold") or 10.0)

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'continuous' column(s)",
                "debug",
            )

            if not matched_cols:
                self.output = self.make_empty_result(
                    (
                        "No continuous columns found — collinear feature detection"
                        " skipped."
                    ),
                    excluded,
                )
                return

            # Drop rows with any null before VIF to avoid statsmodels errors.
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
                vif_scores[col] = float(variance_inflation_factor(numeric_df.values, i))

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

            # Assign output before guidance so _attach_guidance can write to it.
            self.output = result

            # Generate guidance for every column with a meaningful VIF score.
            # Threshold of 5.0 covers moderate (5-10) and high/severe (>10) tiers.
            for col, vif in vif_scores.items():
                if vif >= 5.0:
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
                        "Consider bootstrapping or collecting more data before "
                        "interpreting VIF."
                    ),
                )
            if flags["zero_variance_cols"]:
                add_reliability_warning(
                    result,
                    level="strong_warning",
                    code="vif_zero_variance",
                    description=(
                        "Some features have near-zero variance, which can distort "
                        "VIF calculations."
                    ),
                    recommendation=(
                        "Drop or transform zero-variance features before running VIF."
                    ),
                )

            # ML impact scoring
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
                        f"Column '{top_col}' has high multicollinearity "
                        f"(VIF = {top_vif:.2f}). Consider dropping this feature "
                        "or applying regularization/PCA."
                    ),
                )
                result.summary["column"] = top_col

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
        Generate EDA and ML guidance blurbs for a column with a notable VIF score.

        Args:
            col: Column name.
            vif: Computed VIF value for this column.
            vif_threshold: Task-level threshold above which a column is flagged.

        """
        vif_str: str = f"{vif:.1f}"

        if vif >= 20:
            level = "error"
            tier = "severe"
            eda_interp: str = (
                f"A VIF of {vif_str} means '{col}' shares most of its variance with "
                "other features in the dataset. There is very little independent "
                "information in this column that is not already captured elsewhere."
            )
        elif vif >= 10:
            level = "warn"
            tier = "high"
            eda_interp = (
                f"A VIF of {vif_str} indicates strong linear overlap between '{col}' "
                "and at least one other numeric feature. The column has limited "
                "independent variation beyond what the correlated features already "
                "carry."
            )
        else:
            level = "info"
            tier = "moderate"
            eda_interp = (
                f"A VIF of {vif_str} indicates moderate correlation between '{col}' "
                "and other numeric features. Some redundancy is present, but the "
                "column still carries meaningful independent signal."
            )

        eda_body: str = (
            f"'{col}' has a Variance Inflation Factor of {vif_str} - {tier} "
            f"multicollinearity. {eda_interp} Check the correlation matrix "
            f"(Relationships tab) to identify which features are most strongly "
            f"associated with '{col}'."
        )

        if vif >= 10:
            ml_linear_impact = (
                "Linear models (OLS regression, logistic regression, linear SVM) will "
                "be most affected: coefficient estimates become unstable and their "
                "standard errors inflate, making feature importance and hypothesis "
                "tests unreliable."
            )
            ml_remedy = (
                "Dropping the weaker of the collinear pair, applying ridge/elastic-net "
                "regularisation, or using PCA to orthogonalise the feature space are "
                "the standard remedies."
            )
            ml_actions: list[dict] = [
                {
                    "action": "drop",
                    "column": col,
                    "detail": (
                        "Drop the weaker of the collinear pair after inspecting "
                        "the correlation matrix"
                    ),
                },
                {
                    "action": "regularise",
                    "method": "ridge_or_elastic_net",
                    "detail": (
                        "Regularisation stabilises coefficients under multicollinearity"
                    ),
                },
                {
                    "action": "transform",
                    "method": "pca",
                    "detail": (
                        "PCA produces orthogonal components - eliminates "
                        "multicollinearity entirely"
                    ),
                },
            ]
        else:
            ml_linear_impact = (
                "Linear models may show coefficient instability for this feature - "
                "interpret its coefficient cautiously in any regularised or "
                "unregularised regression."
            )
            ml_remedy = (
                "Consider whether this column adds meaningful signal beyond its "
                "correlated neighbours before including it in a linear model."
            )
            ml_actions = [
                {
                    "action": "monitor",
                    "column": col,
                    "detail": (
                        "Check the correlation matrix to confirm which feature is "
                        "the source of overlap"
                    ),
                },
            ]

        ml_body: str = (
            f"'{col}' has VIF = {vif_str}. {ml_linear_impact} "
            "Tree-based models (Random Forest, Gradient Boosting) are not directly "
            "harmed by multicollinearity, but feature importance scores will be split "
            "between the correlated columns rather than concentrated on one. "
            f"{ml_remedy}"
        )

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
