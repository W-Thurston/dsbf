# dsbf/eda/tasks/one_way_anova.py

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    name="one_way_anova",
    display_name="One-Way ANOVA",
    description=(
        "Tests whether group means differ significantly across levels of a "
        "categorical column for each continuous column. Provides the p-value "
        "companion to the eta-squared effect sizes in compute_pairwise_associations."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["anova", "statistics", "relationships", "numeric", "categorical"],
    expected_semantic_types=["continuous", "categorical"],
)
class OneWayANOVA(BaseTask):
    """
    Test whether group means differ across categorical levels (one-way ANOVA).

    For each continuous x categorical column pair, runs a one-way ANOVA
    F-test: does the mean of the continuous variable differ significantly
    across the levels of the categorical variable?

    **Relationship to compute_pairwise_associations:**
    ``compute_pairwise_associations`` stores eta-squared (effect size) for
    continuous x categorical pairs. ANOVA provides the accompanying p-value
    to determine whether the observed difference is statistically significant
    given sample size.

    **Assumptions:**
    - Independence of observations
    - Approximate normality within groups (robust for large n by CLT)
    - Homogeneity of variance (Levene's test not run here — use
      ``kruskal_wallis`` as a non-parametric alternative when violated)

    Groups with fewer than ``min_group_n`` observations are dropped before
    testing. Pairs with fewer than 2 surviving groups are skipped.

    Configurable parameters (via config["tasks"]["one_way_anova"]):
        alpha (float): Significance threshold. Default: 0.05
        min_group_n (int): Minimum observations per group. Default: 5
        cat_cardinality_limit (int): Skip categorical columns with more
            unique values than this. Default: 20
    """

    def run(self) -> None:
        """
        Execute one-way ANOVA and populate self.output.

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
                self._log(
                    "    No eligible continuous/categorical column pairs.",
                    "debug",
                )
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

            anova_results: dict[str, dict[str, Any]] = {}

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
                        f_stat, p_val = f_oneway(*groups)
                    except Exception:
                        continue

                    if np.isnan(f_stat) or np.isnan(p_val):
                        continue

                    key: str = f"{num_col}|{cat_col}"
                    anova_results[key] = {
                        "f_statistic": round(float(f_stat), 4),
                        "p_value": round(float(p_val), 6),
                        "n_groups": len(groups),
                        "n_total": int(sum(len(g) for g in groups)),
                        "significant": bool(p_val < alpha),
                        "alpha": alpha,
                    }

            significant_count: int = sum(
                1 for v in anova_results.values() if v["significant"]
            )
            self._log(
                f"    {len(anova_results)} pair(s) tested, "
                f"{significant_count} significant at α={alpha}.",
                "debug",
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"One-way ANOVA run on {len(anova_results)} pair(s); "
                        f"{significant_count} significant at α={alpha}."
                    ),
                    "pair_count": len(anova_results),
                    "significant_count": significant_count,
                    "alpha": alpha,
                },
                data=anova_results,
                metadata={
                    "alpha": alpha,
                    "min_group_n": min_group_n,
                    "cat_cardinality_limit": cat_cardinality_limit,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Relationships",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys())
                    ),
                },
            )

            for key, entry in anova_results.items():
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
        """Generate EDA guidance for a significant ANOVA result."""
        f = entry["f_statistic"]
        p = entry["p_value"]
        n = entry["n_total"]
        n_groups = entry["n_groups"]

        body: str = (
            f"The mean of '{num_col}' differs significantly across the "
            f"{n_groups} level(s) of '{cat_col}' "
            f"(F={f:.2f}, p={p:.4f}, n={n:,}). "
            f"This confirms the eta-squared effect size — the categorical "
            f"grouping explains a statistically meaningful portion of the "
            f"variance in '{num_col}'. Inspect group means and distributions "
            f"to understand which specific levels drive the difference. "
            f"Note: ANOVA assumes approximate normality within groups and "
            f"equal variances; use Kruskal-Wallis if these assumptions are "
            f"in doubt."
        )

        metric: dict[str, Any] = {
            "f_statistic": f,
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
                    f"ANOVA: '{num_col}' means differ by "
                    f"'{cat_col}' (F={f:.2f}, p={p:.4f})"
                ),
                body=body.strip(),
                actions=[],
                metric=metric,
            )
