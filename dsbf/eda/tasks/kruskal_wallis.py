# dsbf/eda/tasks/kruskal_wallis.py

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.eda.tasks.one_way_anova import _apply_correction


@register_task(
    name="kruskal_wallis",
    display_name="Kruskal-Wallis Test",
    description=(
        "Non-parametric alternative to one-way ANOVA. Tests whether distributions "
        "differ across categorical groups without assuming normality or equal "
        "variances. Supports Bonferroni and Benjamini-Hochberg FDR correction."
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
    same distribution, without assuming normality or equal variances.

    **Multiple testing correction:**
    Same correction options as ``one_way_anova``: ``"fdr_bh"`` (default),
    ``"bonferroni"``, or ``"none"``. Both raw and corrected p-values are
    stored; the ``significant`` flag uses the corrected value. The number
    of tests run is reported in metadata and guidance blurbs.

    **When to use over ANOVA:**
    - Non-normal distributions within groups
    - Heterogeneous group variances
    - Ordinal data
    - Small group sizes where normality cannot be verified

    A significant result means at least one group differs; use
    ``mann_whitney_u`` for post-hoc pairwise localisation.

    Configurable parameters (via config["tasks"]["kruskal_wallis"]):
        alpha (float): Significance threshold. Default: 0.05
        correction (str): ``"fdr_bh"`` | ``"bonferroni"`` | ``"none"``.
            Default: ``"fdr_bh"``
        min_group_n (int): Minimum observations per group. Default: 5
        cat_cardinality_limit (int): Skip columns with more unique values.
            Default: 20
    """

    def run(self) -> None:
        """
        Execute Kruskal-Wallis tests with multiple testing correction.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run()

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No eligible columns found — Kruskal-Wallis test skipped.",
                    excluded,
                )
                return

            alpha_raw: Any | None = self.get_task_param("alpha")
            alpha: float = float(alpha_raw) if alpha_raw is not None else 0.05

            correction_raw: Any | None = self.get_task_param("correction")
            correction: str = (
                str(correction_raw) if correction_raw is not None else "fdr_bh"
            )
            if correction not in ("none", "bonferroni", "fdr_bh"):
                self._log(
                    f"    Unknown correction '{correction}'"
                    " - falling back to 'fdr_bh'.",
                    "warn",
                )
                correction = "fdr_bh"

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
                        "correction": correction,
                        "n_tests": 0,
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

            # Phase 1 - run all tests
            raw_results: dict[str, dict[str, Any]] = {}
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
                    raw_results[f"{num_col}|{cat_col}"] = {
                        "h_statistic": round(float(h_stat), 4),
                        "p_value": round(float(p_val), 6),
                        "n_groups": len(groups),
                        "n_total": int(sum(len(g) for g in groups)),
                        "alpha": alpha,
                    }

            # Phase 2 - apply correction
            keys: list[str] = list(raw_results.keys())
            corrected: list[float] = _apply_correction(
                [raw_results[k]["p_value"] for k in keys],
                correction,
            )

            kw_results: dict[str, dict[str, Any]] = {}
            for key, p_corr in zip(keys, corrected, strict=False):
                entry: dict[str, Any] = raw_results[key].copy()
                entry["p_value_corrected"] = round(float(p_corr), 6)
                entry["correction"] = correction
                entry["significant"] = bool(p_corr < alpha)
                kw_results[key] = entry

            n_tests: int = len(kw_results)
            significant_count: int = sum(
                1 for v in kw_results.values() if v["significant"]
            )
            self._log(
                f"    {n_tests} test(s), {significant_count} significant "
                f"at α={alpha} after {correction} correction.",
                "debug",
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Kruskal-Wallis: {n_tests} pair(s); "
                        f"{significant_count} significant at α={alpha} "
                        f"after {correction} correction."
                    ),
                    "pair_count": n_tests,
                    "significant_count": significant_count,
                    "alpha": alpha,
                    "correction": correction,
                },
                data=kw_results,
                metadata={
                    "alpha": alpha,
                    "correction": correction,
                    "n_tests": n_tests,
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
                    self._attach_guidance(num_col, cat_col, entry, n_tests)

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
        n_tests: int,
    ) -> None:
        """Generate EDA guidance for a significant Kruskal-Wallis result."""
        h = entry["h_statistic"]
        p_raw = entry["p_value"]
        p_corr = entry["p_value_corrected"]
        n = entry["n_total"]
        n_groups = entry["n_groups"]
        correction = entry["correction"]

        correction_note: str = (
            f"corrected p={p_corr:.4f} ({correction}, {n_tests} tests; "
            f"raw p={p_raw:.4f})"
            if correction != "none"
            else f"p={p_raw:.4f} (uncorrected; {n_tests} tests run)"
        )

        body: str = (
            f"The distribution of '{num_col}' differs significantly across the "
            f"{n_groups} group(s) of '{cat_col}' "
            f"(H={h:.2f}, {correction_note}, n={n:,}). "
            f"The Kruskal-Wallis test makes no normality or equal-variance "
            f"assumptions. At least one group's distribution is stochastically "
            f"different from the others. Use Mann-Whitney U pairwise tests to "
            f"identify which specific group pairs drive the difference."
        )

        metric: dict[str, int | Any] = {
            "h_statistic": h,
            "p_value": p_raw,
            "p_value_corrected": p_corr,
            "correction": correction,
            "n_tests": n_tests,
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
                    f"by '{cat_col}' (H={h:.2f}, p={p_corr:.4f})"
                ),
                body=body.strip(),
                actions=[],
                metric=metric,
            )
