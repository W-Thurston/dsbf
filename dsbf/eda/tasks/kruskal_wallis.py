# dsbf/eda/tasks/kruskal_wallis.py

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    name="kruskal_wallis",
    display_name="Kruskal-Wallis Test",
    description=(
        "Non-parametric alternative to one-way ANOVA. Tests whether distributions "
        "differ across categorical groups without assuming normality or equal "
        "variances. Appropriate when ANOVA assumptions are violated."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["kruskal_wallis", "statistics", "relationships", "non_parametric"],
    expected_semantic_types=["continuous", "categorical"],
)
class KruskalWallis(BaseTask):
    """
    Non-parametric test for distributional differences across categorical groups.

    The Kruskal-Wallis H-test is the non-parametric equivalent of one-way
    ANOVA. It tests whether samples from two or more groups come from the
    same distribution, without assuming normality or equal variances within
    groups. It operates on ranks rather than raw values.

    **When to use Kruskal-Wallis over ANOVA:**
    - Distribution within groups is clearly non-normal
    - Group variances are heterogeneous (Levene's test significant)
    - Ordinal data where arithmetic mean is not meaningful
    - Small group sizes where normality cannot be verified

    **Interpretation:**
    A significant result (p < alpha) means at least one group's distribution
    is stochastically different from the others — not necessarily that all
    group means differ. Post-hoc pairwise Mann-Whitney tests (available in
    ``mann_whitney_u``) localise which specific pairs drive the difference.

    Configurable parameters (via config["tasks"]["kruskal_wallis"]):
        alpha (float): Significance threshold. Default: 0.05
        min_group_n (int): Minimum observations per group. Default: 5
        cat_cardinality_limit (int): Skip categorical columns with more
            unique values than this. Default: 20
    """

    def run(self) -> None:
        """
        Execute Kruskal-Wallis tests and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_cols)} column(s)", "debug")

            alpha_raw: Any | None = self.get_task_param("alpha")
            alpha: float = float(alpha_raw) if alpha_raw is not None else 0.05

            min_group_raw: Any | None = self.get_task_param("min_group_n")
            min_group_n: int = int(min_group_raw) if min_group_raw is not None else 5

            card_raw: Any | None = self.get_task_param("cat_cardinality_limit")
            cat_cardinality_limit: int = int(card_raw) if card_raw is not None else 20

            semantic_types: dict[str, str] = {}
            if self.context:
                semantic_types = self.context.get_metadata("semantic_types") or {}

            continuous_cols: list[str] = [
                col
                for col in df.columns
                if semantic_types.get(col, "") == "continuous"
                and pd.api.types.is_numeric_dtype(df[col])
            ]
            categorical_cols: list[str] = [
                col
                for col in df.columns
                if semantic_types.get(col, "") == "categorical"
                and df[col].nunique() <= cat_cardinality_limit
            ]

            if not continuous_cols or not categorical_cols:
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    summary={
                        "message": "No eligible column pairs found.",
                        "pair_count": 0,
                    },
                    data={},
                    metadata={
                        "alpha": alpha,
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

            kw_results: dict[str, dict[str, Any]] = {}

            for cat_col in categorical_cols:
                for num_col in continuous_cols:
                    paired = df[[num_col, cat_col]].dropna()
                    groups: list = [
                        grp[num_col].to_numpy()
                        for _, grp in paired.groupby(cat_col)
                        if len(grp) >= min_group_n
                    ]

                    if len(groups) < 2:
                        continue

                    try:
                        h_stat, p_val = kruskal(*groups)
                    except Exception:
                        continue

                    if np.isnan(h_stat) or np.isnan(p_val):
                        continue

                    key: str = f"{num_col}|{cat_col}"
                    kw_results[key] = {
                        "h_statistic": round(float(h_stat), 4),
                        "p_value": round(float(p_val), 6),
                        "n_groups": len(groups),
                        "n_total": int(sum(len(g) for g in groups)),
                        "significant": bool(p_val < alpha),
                        "alpha": alpha,
                    }

            significant_count: int = sum(
                1 for v in kw_results.values() if v["significant"]
            )
            self._log(
                f"    {len(kw_results)} pair(s) tested, "
                f"{significant_count} significant at α={alpha}.",
                "debug",
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Kruskal-Wallis run on {len(kw_results)} pair(s); "
                        f"{significant_count} significant at α={alpha}."
                    ),
                    "pair_count": len(kw_results),
                    "significant_count": significant_count,
                    "alpha": alpha,
                },
                data=kw_results,
                metadata={
                    "alpha": alpha,
                    "min_group_n": min_group_n,
                    "cat_cardinality_limit": cat_cardinality_limit,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Relationships",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for key, entry in kw_results.items():
                if entry["significant"]:
                    num_col, cat_col = key.split("|", 1)
                    self._attach_guidance(num_col, cat_col, entry)

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
        num_col: str,
        cat_col: str,
        entry: dict[str, Any],
    ) -> None:
        """Generate EDA guidance for a significant Kruskal-Wallis result."""
        h = entry["h_statistic"]
        p = entry["p_value"]
        n = entry["n_total"]
        n_groups = entry["n_groups"]

        body: str = (
            f"The distribution of '{num_col}' differs significantly across the "
            f"{n_groups} group(s) of '{cat_col}' "
            f"(H={h:.2f}, p={p:.4f}, n={n:,}). "
            f"The Kruskal-Wallis test makes no normality or equal-variance "
            f"assumptions — this result is robust to skewed distributions. "
            f"At least one group's distribution is stochastically different "
            f"from the others. Use Mann-Whitney U pairwise tests to identify "
            f"which specific group pairs drive the difference."
        )

        metric: dict[str, Any] = {
            "h_statistic": h,
            "p_value": p,
            "n_groups": n_groups,
            "n_total": n,
        }

        for col in (num_col, cat_col):
            self.add_guidance(
                result=self.output,
                column=col,
                phase="eda",
                level="info",
                title=(
                    f"Kruskal-Wallis: '{num_col}' distribution differs "
                    f"by '{cat_col}' (H={h:.2f}, p={p:.4f})"
                ),
                body=body.strip(),
                actions=[],
                metric=metric,
            )
