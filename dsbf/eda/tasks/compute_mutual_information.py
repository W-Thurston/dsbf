# dsbf/eda/tasks/compute_mutual_information.py

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

# ── Strength labelling ────────────────────────────────────────────────────────
#
# Mutual information is non-negative and has no fixed upper bound, but in
# practice feature-target MI rarely exceeds 1.0 for real datasets.
# Thresholds below are calibrated for normalised MI (divided by log(n_samples))
# which maps values into a roughly [0, 1] range.
#
# For unnormalised MI the absolute value depends on sample size, so we use
# a relative threshold approach: rank within the column set and flag the
# top-N as high signal. Both raw and normalised values are stored.


def _strength_label(mi_normalised: float) -> str:
    """
    Map a normalised MI score to a human-readable strength label.

    Args:
        mi_normalised: MI divided by log(n_samples), roughly in [0, 1].

    Returns:
        One of ``"high"``, ``"moderate"``, ``"low"``, or ``"negligible"``.

    """
    if mi_normalised >= 0.15:
        return "high"
    if mi_normalised >= 0.05:
        return "moderate"
    if mi_normalised >= 0.01:
        return "low"
    return "negligible"


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="compute_mutual_information",
    display_name="Compute Mutual Information",
    description=(
        "Computes mutual information between each feature and a target column "
        "using sklearn's mutual_info_regression (continuous target) or "
        "mutual_info_classif (categorical target). Captures non-linear "
        "relationships that Pearson correlation misses."
    ),
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="moderate",
    tags=["mutual_information", "relationships", "feature_selection"],
    expected_semantic_types=["any"],
)
class ComputeMutualInformation(BaseTask):
    """
    Compute mutual information between each feature and a designated target.

    Mutual information (MI) is a non-parametric dependence measure that
    captures any functional relationship — linear or non-linear — between
    a feature and a target. It complements ``compute_pairwise_associations``,
    which uses Pearson/Spearman (linear/monotonic only) for continuous pairs.

    **When to use MI over correlation:**
    A feature can have near-zero Pearson correlation with a target but high MI
    if the relationship is non-linear (e.g. quadratic, step-function). MI is
    also applicable across mixed type pairs without needing separate metrics.

    **Target column requirement:**
    A ``target_column`` must be configured. Without a target, MI cannot be
    computed — the task returns a success result with an empty suggestions
    dict and an explanatory message.

    **Target type routing:**
    - Continuous target → ``mutual_info_regression``
    - Categorical/binary target → ``mutual_info_classif``
    Target type is inferred from semantic types written by ``infer_types``.

    **Normalisation:**
    Raw MI scores are stored alongside normalised scores (divided by
    ``log(n_samples)``), which maps them into a roughly [0, 1] range for
    cross-dataset comparability. Strength labels are assigned on the normalised
    score.

    **Discrete feature handling:**
    Columns classified as ``categorical`` are passed with
    ``discrete_features=True`` to the sklearn estimator. Continuous columns
    use the default k-nearest-neighbour entropy estimator.

    **EDA guidance** is emitted for high-MI features (normalised score ≥ 0.05),
    explaining the non-linear signal. **ML guidance** is emitted as ranked
    feature importance chips, which the Relationships tab can surface for
    feature selection decisions.

    Configurable parameters (via config["tasks"]["compute_mutual_information"]):
        target_column (str): Name of the target column. Required.
        n_neighbors (int): k for the KNN entropy estimator. Default: 3.
        random_state (int): Seed for reproducibility. Default: 42.
        min_mi_for_guidance (float): Normalised MI threshold above which EDA
            guidance is emitted. Default: 0.05.
    """

    def run(self) -> None:
        """
        Execute MI computation and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_cols)} column(s)", "debug")

            target_col: str | None = self.get_task_param("target_column")
            n_neighbors_raw: Any | None = self.get_task_param("n_neighbors")
            n_neighbors: int = (
                int(n_neighbors_raw) if n_neighbors_raw is not None else 3
            )
            random_state_raw: Any | None = self.get_task_param("random_state")
            random_state: int = (
                int(random_state_raw) if random_state_raw is not None else 42
            )
            min_mi_raw: Any | None = self.get_task_param("min_mi_for_guidance")
            min_mi_for_guidance: float = (
                float(min_mi_raw) if min_mi_raw is not None else 0.05
            )

            if not target_col or target_col not in df.columns:
                self._log(
                    "    No target_column configured or column not found in "
                    "DataFrame — MI cannot be computed without a target.",
                    "debug",
                )
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    summary={
                        "message": (
                            "No target column configured. Set target_column in "
                            "task config to enable mutual information computation."
                        ),
                        "feature_count": 0,
                    },
                    data={"mi_scores": {}},
                    metadata={
                        "target_column": None,
                        "suggested_viz_type": "bar",
                        "recommended_section": "Relationships",
                        "display_priority": "medium",
                        "excluded_columns": excluded,
                        "column_types": self.get_column_type_info(
                            matched_cols + list(excluded.keys()),
                        ),
                    },
                )
                return

            # Read semantic types for feature/target type routing
            semantic_types: dict[str, str] = {}
            if self.context:
                semantic_types = self.context.get_metadata("semantic_types") or {}

            target_intent: str = semantic_types.get(target_col, "continuous")
            is_classification: bool = target_intent in ("categorical",)

            # --- Prepare target ---
            target_series = df[target_col].dropna()
            valid_idx = target_series.index

            if is_classification:
                # Encode categorical target as integer codes
                target_encoded = pd.Categorical(
                    df.loc[valid_idx, target_col],
                ).codes.astype(float)
            else:
                target_encoded = df.loc[valid_idx, target_col].astype(float).to_numpy()

            # --- Prepare features ---
            # Exclude the target itself, id/datetime columns, and all-null columns
            skip_intents: set[str] = {"id", "datetime", "unknown"}
            feature_cols: list = [
                col
                for col in df.columns
                if col != target_col
                and semantic_types.get(col, "continuous") not in skip_intents
                and df[col].notna().any()
            ]

            if not feature_cols:
                self._log("    No eligible feature columns found.", "debug")
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    summary={
                        "message": "No eligible feature columns found.",
                        "feature_count": 0,
                    },
                    data={"mi_scores": {}},
                    metadata={
                        "target_column": target_col,
                        "target_intent": target_intent,
                        "suggested_viz_type": "bar",
                        "recommended_section": "Relationships",
                        "display_priority": "medium",
                        "excluded_columns": excluded,
                        "column_types": self.get_column_type_info(
                            matched_cols + list(excluded.keys()),
                        ),
                    },
                )
                return

            # Build feature matrix aligned to valid target indices
            X_raw = df.loc[valid_idx, feature_cols].copy()

            # Encode categorical features as integer codes;
            # impute nulls with median/mode
            discrete_mask: list[bool] = []
            for col in feature_cols:
                col_intent: str = semantic_types.get(col, "continuous")
                if col_intent == "categorical" or X_raw[col].dtype == object:
                    X_raw[col] = pd.Categorical(
                        X_raw[col].fillna("__missing__"),
                    ).codes.astype(float)
                    discrete_mask.append(True)
                else:
                    # Impute numeric nulls with median
                    median_val = X_raw[col].median()
                    X_raw[col] = X_raw[col].fillna(
                        median_val if not np.isnan(median_val) else 0.0,
                    )
                    discrete_mask.append(False)

            X = X_raw.to_numpy().astype(float)
            n_samples = X.shape[0]

            self._log(
                f"    Computing MI for {len(feature_cols)} features against "
                f"'{target_col}' ({target_intent} target, n={n_samples}).",
                "debug",
            )

            # --- Compute MI ---
            try:
                if is_classification:
                    mi_raw = mutual_info_classif(
                        X,
                        target_encoded,
                        discrete_features=discrete_mask,
                        n_neighbors=n_neighbors,
                        random_state=random_state,
                    )
                else:
                    mi_raw = mutual_info_regression(
                        X,
                        target_encoded,
                        discrete_features=discrete_mask,
                        n_neighbors=n_neighbors,
                        random_state=random_state,
                    )
            except Exception as e:
                msg: str = f"sklearn MI computation failed: {type(e).__name__} - {e}"
                raise RuntimeError(
                    msg,
                ) from e

            # Normalise by log(n_samples) for cross-dataset comparability
            log_n: float | Any = np.log(n_samples) if n_samples > 1 else 1.0
            mi_normalised = mi_raw / log_n

            # --- Build results dict ---
            mi_scores: dict[str, dict[str, Any]] = {}
            for col, raw_val, norm_val in zip(
                feature_cols,
                mi_raw.tolist(),
                mi_normalised.tolist(),
                strict=False,
            ):
                strength: str = _strength_label(norm_val)
                mi_scores[col] = {
                    "mi_score": round(float(raw_val), 6),
                    "mi_normalised": round(float(norm_val), 6),
                    "strength": strength,
                    "is_discrete": discrete_mask[feature_cols.index(col)],
                }

            # Sort by MI score descending for display
            mi_scores = dict(sorted(mi_scores.items(), key=lambda x: -x[1]["mi_score"]))

            high_signal: list[str] = [
                col
                for col, v in mi_scores.items()
                if v["mi_normalised"] >= min_mi_for_guidance
            ]
            self._log(
                f"    {len(high_signal)} high-signal feature(s) "
                f"(normalised MI ≥ {min_mi_for_guidance}).",
                "debug",
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed MI for {len(mi_scores)} features against "
                        f"'{target_col}'; {len(high_signal)} with notable signal."
                    ),
                    "feature_count": len(mi_scores),
                    "high_signal_count": len(high_signal),
                    "target_column": target_col,
                    "target_intent": target_intent,
                },
                data={"mi_scores": mi_scores},
                metadata={
                    "target_column": target_col,
                    "target_intent": target_intent,
                    "n_neighbors": n_neighbors,
                    "random_state": random_state,
                    "n_samples": n_samples,
                    "min_mi_for_guidance": min_mi_for_guidance,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Relationships",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, scores in mi_scores.items():
                if scores["mi_normalised"] >= min_mi_for_guidance:
                    self._attach_guidance(
                        col,
                        scores,
                        target_col,
                        target_intent,
                        n_samples,
                    )

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(
        self,
        col: str,
        scores: dict[str, Any],
        target_col: str,
        target_intent: str,
        n_samples: int,
    ) -> None:
        """
        Generate EDA and ML guidance for a feature with notable MI signal.

        Args:
            col: Feature column name.
            scores: MI score dict for this column.
            target_col: Target column name.
            target_intent: Semantic intent of the target column.
            n_samples: Number of samples used in MI computation.

        """
        mi = scores["mi_score"]
        mi_norm = scores["mi_normalised"]
        strength = scores["strength"]

        eda_body: str = (
            f"'{col}' has {strength} mutual information with '{target_col}' "
            f"(MI={mi:.4f}, normalised={mi_norm:.4f}, n={n_samples:,}). "
            f"Unlike Pearson correlation, mutual information captures any "
            f"functional relationship — including non-linear, step-function, "
            f"and interaction patterns. A high MI score means '{col}' shares "
            f"meaningful information with the target, but does not indicate "
            f"the shape of that relationship. Inspect scatter plots or "
            f"conditional distributions to understand the pattern."
        )

        ml_body: str = (
            f"'{col}' has {strength} predictive signal for '{target_col}' "
            f"based on mutual information (MI={mi:.4f}, normalised={mi_norm:.4f}). "
            f"MI-based ranking is particularly useful for detecting non-linear "
            f"features that Pearson correlation would score near zero. "
            f"High-MI features are strong candidates for inclusion even if their "
            f"linear correlation with the target is low. "
            f"Note: MI scores can be inflated for high-cardinality categorical "
            f"features — verify that the signal is genuine and not an artefact "
            f"of cardinality."
        )

        metric: dict[str, int | str | Any] = {
            "mi_score": mi,
            "mi_normalised": mi_norm,
            "strength": strength,
            "target_column": target_col,
            "n_samples": n_samples,
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="info",
            title=f"Mutual Information with '{target_col}': {strength} (MI={mi:.4f})",
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level="info",
            title=f"Feature Signal: {strength} MI with '{target_col}'",
            body=ml_body.strip(),
            actions=[
                {
                    "action": "include_in_model",
                    "column": col,
                    "detail": (
                        f"MI={mi:.4f} — {strength} predictive signal for '{target_col}'"
                    ),
                },
            ],
            metric=metric,
        )
