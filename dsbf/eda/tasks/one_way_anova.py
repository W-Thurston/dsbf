# dsbf/eda/tasks/one_way_anova.py


from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result

if TYPE_CHECKING:
    from numpy import ndarray


def _apply_correction(p_values: list[float], method: str) -> list[float]:
    """
    Apply multiple testing correction to a list of p-values.

    Args:
        p_values: Raw p-values from individual tests.
        method: One of ``"none"``, ``"bonferroni"``, ``"fdr_bh"``.

    Returns:
        Corrected p-values in the same order as input, capped at 1.0.

    """
    if method == "none" or len(p_values) <= 1:
        return list(p_values)

    arr: ndarray = np.array(p_values, dtype=float)

    if method == "bonferroni":
        return list(np.minimum(arr * len(arr), 1.0))

    # Benjamini-Hochberg
    n = len(arr)
    order = np.argsort(arr)
    bh = arr[order] * n / np.arange(1, n + 1)
    # Enforce monotonicity right-to-left
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    corrected = np.empty(n)
    corrected[order] = bh
    return list(np.minimum(corrected, 1.0))


@register_task(
    name="one_way_anova",
    display_name="One-Way ANOVA",
    description=(
        "Tests whether group means differ significantly across levels of a "
        "categorical column for each continuous column. Supports Bonferroni "
        "and Benjamini-Hochberg FDR multiple testing correction."
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
    F-test. The ``significant`` flag is applied to the corrected p-value;
    both raw and corrected values are stored for transparency.

    **Multiple testing correction:**
    Running ANOVA across many column pairs inflates the family-wise error
    rate. The ``correction`` param controls the adjustment:

    - ``"fdr_bh"`` (default): Benjamini-Hochberg FDR. Best for exploratory
      EDA - less conservative than Bonferroni, retains more true signals.
    - ``"bonferroni"``: Multiply each p-value by n_tests. Controls family-wise
      error rate. Use when any single false positive is costly.
    - ``"none"``: Raw p-values only. Not recommended for wide datasets.

    The number of tests run and the correction method are stored in metadata
    and surfaced in guidance blurbs.

    Configurable parameters (via config["tasks"]["one_way_anova"]):
        alpha (float): Significance threshold applied to corrected p-values.
            Default: 0.05
        correction (str): ``"fdr_bh"`` | ``"bonferroni"`` | ``"none"``.
            Default: ``"fdr_bh"``
        min_group_n (int): Minimum observations per group. Default: 5
        cat_cardinality_limit (int): Skip categorical columns with more
            unique values than this. Default: 20
    """

    def run(self) -> None:
        """
        Execute one-way ANOVA with multiple testing correction.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run()

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No eligible columns found — one-way ANOVA skipped.",
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

            # Derive semantic type split from matched_cols + context metadata.
            # matched_cols is the authoritative list returned by
            # get_columns_by_intent();
            # we re-read semantic_types only to split those columns into the two
            # sub-lists the ANOVA loop needs.  We deliberately do NOT re-query
            # df.columns, which would re-introduce all columns regardless of the
            # type-inference result.
            semantic_types: dict[str, str] = {}
            if self.context:
                semantic_types = self.context.get_metadata("semantic_types") or {}

            continuous_cols: list[str] = [
                col
                for col in matched_cols
                if semantic_types.get(col, "") == "continuous"
                and pd.api.types.is_numeric_dtype(df[col])
            ]
            categorical_cols: list[str] = [
                col
                for col in matched_cols
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

            # Phase 1 - run all tests, collect raw results
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
                        f_stat, p_val = f_oneway(*groups)
                    except Exception:
                        continue
                    if np.isnan(f_stat) or np.isnan(p_val):
                        continue
                    raw_results[f"{num_col}|{cat_col}"] = {
                        "f_statistic": round(float(f_stat), 4),
                        "p_value": round(float(p_val), 6),
                        "n_groups": len(groups),
                        "n_total": int(sum(len(g) for g in groups)),
                        "alpha": alpha,
                    }

            # Phase 2 - apply correction across all collected p-values
            keys: list[str] = list(raw_results.keys())
            corrected: list[float] = _apply_correction(
                [raw_results[k]["p_value"] for k in keys],
                correction,
            )

            anova_results: dict[str, dict[str, Any]] = {}
            for key, p_corr in zip(keys, corrected, strict=False):
                entry: dict[str, Any] = raw_results[key].copy()
                entry["p_value_corrected"] = round(float(p_corr), 6)
                entry["correction"] = correction
                entry["significant"] = bool(p_corr < alpha)
                anova_results[key] = entry

            n_tests: int = len(anova_results)
            significant_count: int = sum(
                1 for v in anova_results.values() if v["significant"]
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
                        f"One-way ANOVA: {n_tests} pair(s); "
                        f"{significant_count} significant at α={alpha} "
                        f"after {correction} correction."
                    ),
                    "pair_count": n_tests,
                    "significant_count": significant_count,
                    "alpha": alpha,
                    "correction": correction,
                },
                data=anova_results,
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

            for key, entry in anova_results.items():
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
        """Generate EDA guidance for a significant ANOVA result."""
        f = entry["f_statistic"]
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
            f"The mean of '{num_col}' differs significantly across the "
            f"{n_groups} level(s) of '{cat_col}' "
            f"(F={f:.2f}, {correction_note}, n={n:,}). "
            f"Inspect group means and distributions to understand which "
            f"specific levels drive the difference. "
            f"Note: ANOVA assumes approximate normality within groups and "
            f"equal variances - use Kruskal-Wallis if these are in doubt."
        )

        metric: dict[str, int | Any] = {
            "f_statistic": f,
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
                    f"ANOVA: '{num_col}' means differ by '{cat_col}' "
                    f"(F={f:.2f}, p={p_corr:.4f})"
                ),
                body=body.strip(),
                actions=[],
                metric=metric,
            )
