# dsbf/eda/tasks/mann_whitney_u.py

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.eda.tasks.one_way_anova import _apply_correction
from dsbf.utils.backend import is_polars


@register_task(
    name="mann_whitney_u",
    display_name="Mann-Whitney U Test",
    description=(
        "Non-parametric pairwise comparison of a continuous column across "
        "exactly two category levels. Supports Bonferroni and Benjamini-Hochberg "
        "FDR multiple testing correction across all level pairs tested."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["mann_whitney", "statistics", "relationships", "non_parametric"],
    expected_semantic_types=["continuous", "categorical"],
)
class MannWhitneyU(BaseTask):
    """
    Pairwise non-parametric comparison for continuous vs categorical column pairs.

    The Mann-Whitney U test tests whether two independent samples come from
    the same distribution, without assuming normality.

    **Multiple testing correction:**
    Correction is applied across all pairwise level comparisons tested in a
    single run. This includes all (continuous x categorical x level_pair)
    combinations, which can be substantial for multi-level categoricals.
    Same options as ``one_way_anova``: ``"fdr_bh"`` (default),
    ``"bonferroni"``, ``"none"``. Both raw and corrected p-values are stored;
    the ``significant`` flag uses the corrected value.

    **Effect size:**
    Rank-biserial correlation r = 1 - 2U / (n₁ x n₂), in [-1, 1].

    Configurable parameters (via config["tasks"]["mann_whitney_u"]):
        alpha (float): Significance threshold. Default: 0.05
        correction (str): ``"fdr_bh"`` | ``"bonferroni"`` | ``"none"``.
            Default: ``"fdr_bh"``
        min_group_n (int): Minimum observations per group. Default: 5
        cat_cardinality_limit (int): Max unique values in categorical column.
            Default: 10
        max_pairs_per_column (int): Maximum pairwise level comparisons per
            categorical column. Default: 10
    """

    def run(self) -> None:
        """
        Execute Mann-Whitney U tests with multiple testing correction.

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

            correction_raw: Any | None = self.get_task_param("correction")
            correction: str = (
                str(correction_raw) if correction_raw is not None else "fdr_bh"
            )
            if correction not in ("none", "bonferroni", "fdr_bh"):
                self._log(
                    f"    Unknown correction '{correction}'"
                    " — falling back to 'fdr_bh'.",
                    "warn",
                )
                correction = "fdr_bh"

            min_group_raw: Any | None = self.get_task_param("min_group_n")
            min_group_n: int = int(min_group_raw) if min_group_raw is not None else 5

            card_raw: Any | None = self.get_task_param("cat_cardinality_limit")
            cat_cardinality_limit: int = int(card_raw) if card_raw is not None else 10

            max_pairs_raw: Any | None = self.get_task_param("max_pairs_per_column")
            max_pairs: int = int(max_pairs_raw) if max_pairs_raw is not None else 10

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
                        "test_count": 0,
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

            # Phase 1 — run all tests
            raw_results: dict[str, dict[str, Any]] = {}

            for cat_col in categorical_cols:
                levels = df[cat_col].dropna().unique().tolist()
                level_pairs: list[tuple] = [
                    (levels[i], levels[j])
                    for i in range(len(levels))
                    for j in range(i + 1, len(levels))
                ][:max_pairs]

                for num_col in continuous_cols:
                    for lev_a, lev_b in level_pairs:
                        group_a = (
                            df.loc[df[cat_col] == lev_a, num_col].dropna().to_numpy()
                        )
                        group_b = (
                            df.loc[df[cat_col] == lev_b, num_col].dropna().to_numpy()
                        )

                        if len(group_a) < min_group_n or len(group_b) < min_group_n:
                            continue
                        try:
                            u_stat, p_val = mannwhitneyu(
                                group_a,
                                group_b,
                                alternative="two-sided",
                            )
                        except Exception:
                            continue
                        if np.isnan(u_stat) or np.isnan(p_val):
                            continue

                        n1, n2 = len(group_a), len(group_b)
                        r_rb: float = 1.0 - (2.0 * float(u_stat)) / (n1 * n2)
                        key: str = f"{num_col}|{cat_col}|{lev_a}_vs_{lev_b}"
                        raw_results[key] = {
                            "u_statistic": round(float(u_stat), 4),
                            "p_value": round(float(p_val), 6),
                            "rank_biserial_r": round(float(r_rb), 4),
                            "n_group_a": int(n1),
                            "n_group_b": int(n2),
                            "level_a": str(lev_a),
                            "level_b": str(lev_b),
                            "num_col": num_col,
                            "cat_col": cat_col,
                            "alpha": alpha,
                        }

            # Phase 2 — apply correction
            keys: list[str] = list(raw_results.keys())
            corrected: list[float] = _apply_correction(
                [raw_results[k]["p_value"] for k in keys], correction
            )

            mw_results: dict[str, dict[str, Any]] = {}
            for key, p_corr in zip(keys, corrected, strict=False):
                entry: dict[str, Any] = raw_results[key].copy()
                entry["p_value_corrected"] = round(float(p_corr), 6)
                entry["correction"] = correction
                entry["significant"] = bool(p_corr < alpha)
                mw_results[key] = entry

            n_tests = len(mw_results)
            significant_count: int = sum(
                1 for v in mw_results.values() if v["significant"]
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
                        f"Mann-Whitney U: {n_tests} test(s); "
                        f"{significant_count} significant at α={alpha} "
                        f"after {correction} correction."
                    ),
                    "test_count": n_tests,
                    "significant_count": significant_count,
                    "alpha": alpha,
                    "correction": correction,
                },
                data=mw_results,
                metadata={
                    "alpha": alpha,
                    "correction": correction,
                    "n_tests": n_tests,
                    "min_group_n": min_group_n,
                    "cat_cardinality_limit": cat_cardinality_limit,
                    "max_pairs_per_column": max_pairs,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Relationships",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for key, entry in mw_results.items():
                if entry["significant"]:
                    self._attach_guidance(entry, n_tests)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, entry: dict[str, Any], n_tests: int) -> None:
        """Generate EDA guidance for a significant Mann-Whitney result."""
        num_col = entry["num_col"]
        cat_col = entry["cat_col"]
        lev_a = entry["level_a"]
        lev_b = entry["level_b"]
        u = entry["u_statistic"]
        p_raw = entry["p_value"]
        p_corr = entry["p_value_corrected"]
        r = entry["rank_biserial_r"]
        n1 = entry["n_group_a"]
        n2 = entry["n_group_b"]
        correction = entry["correction"]

        direction: str = "higher" if r > 0 else "lower"
        effect_desc: str = (
            "large" if abs(r) >= 0.5 else "moderate" if abs(r) >= 0.3 else "small"
        )
        correction_note: str = (
            f"corrected p={p_corr:.4f} ({correction}, {n_tests} tests; "
            f"raw p={p_raw:.4f})"
            if correction != "none"
            else f"p={p_raw:.4f} (uncorrected; {n_tests} tests run)"
        )

        body: str = (
            f"The distribution of '{num_col}' differs significantly between "
            f"'{lev_a}' (n={n1}) and '{lev_b}' (n={n2}) in '{cat_col}' "
            f"(U={u:.1f}, {correction_note}, r={r:.3f} — {effect_desc} effect). "
            f"Values in '{lev_a}' tend to be {direction} than in '{lev_b}'. "
            f"Rank-biserial r={r:.3f} means "
            f"{'%.0f' % ((abs(r) + 1) / 2 * 100)}% of observations in "
            f"'{lev_a}' exceed a randomly chosen observation in '{lev_b}'."
        )

        metric: dict[str, int | Any] = {
            "u_statistic": u,
            "p_value": p_raw,
            "p_value_corrected": p_corr,
            "correction": correction,
            "n_tests": n_tests,
            "rank_biserial_r": r,
            "level_a": lev_a,
            "level_b": lev_b,
            "n_group_a": n1,
            "n_group_b": n2,
        }

        for col in (num_col, cat_col):
            self.add_guidance(
                result=self.output,
                column=col,
                phase="eda",
                level="info",
                title=(
                    f"Mann-Whitney: '{num_col}' differs between "
                    f"'{lev_a}' vs '{lev_b}' (p={p_corr:.4f}, r={r:.3f})"
                ),
                body=body.strip(),
                actions=[],
                metric=metric,
            )
