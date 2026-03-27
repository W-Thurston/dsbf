# dsbf/eda/tasks/kendalls_tau.py

from typing import Any

import numpy as np
from scipy.stats import kendalltau

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

# ── Strength labelling ────────────────────────────────────────────────────────


def _strength(tau: float) -> str:
    """
    Map an absolute Kendall's tau to a strength label.

    Uses the same thresholds as Pearson/Spearman in compute_pairwise_associations
    for consistency across the Relationships tab.

    Args:
        tau: Absolute Kendall's tau in [0, 1].

    Returns:
        One of ``"strong"``, ``"moderate"``, ``"weak"``, ``"negligible"``.

    """
    v: float = abs(tau)
    if v >= 0.7:
        return "strong"
    if v >= 0.4:
        return "moderate"
    if v >= 0.2:
        return "weak"
    return "negligible"


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="kendalls_tau",
    display_name="Kendall's Tau",
    description=(
        "Computes Kendall's tau-b rank correlation for all continuous column "
        "pairs. More robust than Spearman on small samples and directly "
        "interpretable as a concordance probability."
    ),
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="moderate",
    tags=["correlation", "rank", "relationships", "numeric"],
    expected_semantic_types=["continuous"],
)
class KendallsTau(BaseTask):
    """
    Compute Kendall's tau-b rank correlation for all continuous column pairs.

    Kendall's tau is a rank-based correlation coefficient measuring the
    proportion of concordant minus discordant pairs relative to the total:

        τ = (concordant - discordant) / √((n₀ - n₁)(n₀ - n₂))

    where n₀ is the total pairs, n₁ and n₂ are ties in each variable
    (the -b variant handles ties).

    **When to use Kendall's tau over Spearman:**

    - **Small samples** (n < 30): Kendall's tau has better statistical
      properties than Spearman for small n - it is more robust to outliers
      and its p-value is more accurate.
    - **Interpretability**: tau has a direct probabilistic interpretation -
      a tau of 0.6 means 60% more concordant pairs than discordant ones.
    - **Tied data**: tau-b handles ties explicitly without approximation.

    Spearman (in ``compute_pairwise_associations``) is generally preferred for
    large samples due to computational efficiency; tau is preferred when n is
    small or a probability-of-concordance interpretation is needed.

    Each pair result includes the tau-b coefficient, p-value for independence,
    number of complete observations, and a strength label.

    EDA guidance is emitted for pairs where the association is at least
    ``weak`` (|tau| ≥ 0.2) and statistically significant at alpha.

    Configurable parameters (via config["tasks"]["kendalls_tau"]):
        alpha (float): Significance threshold for guidance emission. Default: 0.05
        min_n (int): Minimum complete pairs required. Default: 5
        cat_cardinality_limit (int): Upper limit on n_unique for a column
            to be treated as continuous. Default: none (uses semantic types)
    """

    def run(self) -> None:
        """
        Execute Kendall's tau computation and populate self.output.

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

            alpha_raw: Any | None = self.get_task_param("alpha")
            alpha: float = float(alpha_raw) if alpha_raw is not None else 0.05

            min_n_raw: Any | None = self.get_task_param("min_n")
            min_n: int = int(min_n_raw) if min_n_raw is not None else 5

            # Identify continuous columns from semantic types
            semantic_types: dict[str, str] = {}
            if self.context:
                semantic_types = self.context.get_metadata("semantic_types") or {}

            numeric_df = df.select_dtypes(include=np.number)
            continuous_cols: list[str] = [
                col
                for col in numeric_df.columns
                if semantic_types.get(col, "continuous") == "continuous"
            ]

            if not continuous_cols:
                continuous_cols = list(numeric_df.columns)

            self._log(
                f"    {len(continuous_cols)} continuous column(s) eligible.",
                "debug",
            )

            tau_results: dict[str, dict[str, Any]] = {}

            for i, col_a in enumerate(continuous_cols):
                for col_b in continuous_cols[i + 1 :]:
                    paired = df[[col_a, col_b]].dropna()
                    n: int = len(paired)

                    if n < min_n:
                        self._log(
                            f"    Skipping {col_a}|{col_b}: only {n} complete pairs.",
                            "debug",
                        )
                        continue

                    try:
                        tau_val, p_val = kendalltau(
                            paired[col_a].values,
                            paired[col_b].values,
                        )
                    except Exception as e:  # noqa: BLE001
                        self._log(f"    {col_a}|{col_b} failed: {e}", "debug")
                        continue

                    key: str = f"{col_a}|{col_b}"
                    tau_results[key] = {
                        "tau": round(float(tau_val), 6),
                        "p_value": round(float(p_val), 6),
                        "n": n,
                        "strength": _strength(tau_val),
                        "significant": bool(p_val < alpha),
                    }

            significant_count: int = sum(
                1 for v in tau_results.values() if v["significant"]
            )
            self._log(
                f"    {len(tau_results)} pair(s) computed, "
                f"{significant_count} significant at α={alpha}.",
                "debug",
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Kendall's tau computed for {len(tau_results)} pair(s); "
                        f"{significant_count} significant at α={alpha}."
                    ),
                    "pair_count": len(tau_results),
                    "significant_count": significant_count,
                    "alpha": alpha,
                },
                data=tau_results,
                metadata={
                    "alpha": alpha,
                    "min_n": min_n,
                    "suggested_viz_type": "heatmap",
                    "recommended_section": "Relationships",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for key, entry in tau_results.items():
                col_a, col_b = key.split("|", 1)
                if entry["significant"] and abs(entry["tau"]) >= 0.2:
                    self._attach_guidance(col_a, col_b, entry)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col_a: str, col_b: str, entry: dict[str, Any]) -> None:
        """
        Generate EDA guidance for a significant Kendall's tau pair.

        Args:
            col_a: First column name.
            col_b: Second column name.
            entry: tau result dict.

        """
        tau = entry["tau"]
        p = entry["p_value"]
        n = entry["n"]
        strength = entry["strength"]
        direction: str = "positive" if tau > 0 else "negative"

        body: str = (
            f"'{col_a}' and '{col_b}' have a {strength} {direction} rank "
            f"association (τ={tau:.3f}, p={p:.4f}, n={n}). "
            f"Kendall's tau has a direct probabilistic interpretation: "
            f"{'%.0f' % (abs(tau) * 100)}% more concordant pairs than "
            f"discordant ones. Unlike Pearson correlation, tau captures "
            f"any monotonic relationship and is robust to outliers. "
            f"For small samples (n < 30), tau's p-value is more accurate "
            f"than Spearman's. Inspect a scatter plot to understand the "
            f"shape of the relationship."
        )

        metric: dict[str, Any] = {
            "tau": tau,
            "p_value": p,
            "n": n,
            "strength": strength,
        }

        for col, other in ((col_a, col_b), (col_b, col_a)):
            self.add_guidance(
                result=self.output,
                column=col,
                phase="eda",
                level="info",
                title=(
                    f"Kendall's τ with '{other}': {strength} (τ={tau:.3f}, p={p:.4f})"
                ),
                body=body.strip(),
                actions=[],
                metric=metric,
            )
