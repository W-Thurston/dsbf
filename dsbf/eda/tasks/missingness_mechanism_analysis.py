# dsbf/eda/tasks/missingness_mechanism_analysis.py
#
# Analyses the likely mechanism behind missing values in each column.
#
# The three missing data mechanisms (Rubin 1976):
#   MCAR - Missing Completely At Random: missingness is independent of both
#           observed and unobserved data. Testable via Little's MCAR test.
#   MAR  - Missing At Random: missingness depends on observed data but not on
#           the missing values themselves. NOT directly testable - only
#           evidence consistent with MAR can be found.
#   MNAR - Missing Not At Random: missingness depends on the unobserved missing
#           values themselves. FUNDAMENTALLY UNVERIFIABLE from observed data.
#
# This task deliberately avoids claiming to confirm any mechanism. It reports:
#   1. What the data shows (correlations, group differences, MCAR test)
#   2. What each finding is consistent with
#   3. Explicit epistemic caveats for every column
#   4. A structured "consistent_with" assessment - never a definitive verdict
#
# Analysts should treat all outputs as hypothesis-generating, not confirmatory.
# Domain knowledge is the only reliable guide to mechanism.


from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, mannwhitneyu, pointbiserialr

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result

if TYPE_CHECKING:
    from numpy import ndarray
    from pandas import DataFrame, Series

# ── Statistical helpers ───────────────────────────────────────────────────────


def _little_mcar_test(
    df: pd.DataFrame,
) -> dict[str, Any] | None:
    """
    Perform a simplified Little's MCAR test on numeric columns.

    Little's test examines whether the means of observed variables differ
    across groups defined by missing-data patterns. A significant result
    (p < alpha) suggests the data is NOT consistent with MCAR. A non-
    significant result is consistent with MCAR but does not confirm it -
    the test has low power with small samples.

    This implementation uses the pattern-mean comparison approach:
    for each missing-data pattern, compare observed column means against
    the overall means. The test statistic follows a chi-squared distribution
    under the null hypothesis of MCAR.

    Args:
        df: DataFrame with numeric columns only.
        missing_cols: Columns that have at least one missing value.

    Returns:
        Dict with test_statistic, p_value, df (degrees of freedom),
        n_patterns, n, and caveats list. Returns None if fewer than
        2 numeric columns or fewer than 10 rows.

    """
    numeric_df: DataFrame = df.select_dtypes(include=np.number)
    cols: list[str] = [c for c in numeric_df.columns if numeric_df[c].notna().any()]
    if len(cols) < 2 or len(df) < 10:
        return None

    X: Series = numeric_df[cols].copy()
    n: int = len(X)
    overall_means: float = X.mean()

    # Identify unique missingness patterns
    pattern_key: Series = X.isna().astype(int).apply(lambda row: tuple(row), axis=1)
    patterns = pattern_key.unique()

    if len(patterns) < 2:
        return None

    d2 = 0.0
    dof = 0

    for pattern in patterns:
        mask = pattern_key == pattern
        group = X[mask]
        n_j: int = len(group)
        if n_j < 2:
            continue

        # Observed (non-missing) columns in this pattern
        observed_in_pattern: list[str] = [
            c for c, missing in zip(cols, pattern, strict=False) if missing == 0
        ]
        if len(observed_in_pattern) < 1:
            continue

        # Mean difference for observed columns in this pattern
        group_means = group[observed_in_pattern].mean()
        overall_means_sub = overall_means[observed_in_pattern]
        diff = (group_means - overall_means_sub).to_numpy()

        # Use diagonal covariance (full covariance inversion is numerically
        # unstable for small groups - diagonal is a conservative approximation)
        cov_diag = group[observed_in_pattern].var(ddof=1).to_numpy()
        cov_diag: ndarray = np.where(cov_diag > 1e-10, cov_diag, 1e-10)

        d2 += n_j * float(np.sum(diff**2 / cov_diag))
        dof += len(observed_in_pattern)

    if dof == 0:
        return None

    p_val = float(1.0 - chi2.cdf(d2, df=dof))

    caveats: list[str] = [
        "A non-significant result (p >= alpha) is consistent with MCAR but "
        "does not confirm it - the test has low statistical power on small "
        "samples and can miss subtle non-random patterns.",
        "This implementation uses a diagonal covariance approximation for "
        "numerical stability. Results may differ from software that uses the "
        "full covariance matrix (e.g., R's naniar::mcar_test).",
        "The test operates on the joint distribution of all numeric columns. "
        "If missingness is selective across column types it may not be detected.",
    ]

    if n < 50:
        caveats.append(
            f"Sample size is small (n={n}). Test power is low - the result "
            "should be treated with extra caution.",
        )

    return {
        "test_statistic": round(d2, 4),
        "p_value": round(p_val, 6),
        "degrees_of_freedom": dof,
        "n_patterns": len(patterns),
        "n": n,
        "caveats": caveats,
    }


def _missingness_correlations(
    df: pd.DataFrame,
    target_col: str,
    other_cols: list[str],
    min_n: int = 10,
) -> dict[str, dict[str, Any]]:
    """
    Compute point-biserial correlations between a missingness indicator
    and other observed columns.

    A high absolute correlation between is_missing(target_col) and another
    column suggests the missingness of target_col is related to that column's
    values. This is consistent with MAR but does not rule out MNAR.

    Args:
        df: Source DataFrame.
        target_col: Column whose missingness is being analysed.
        other_cols: Other columns to correlate against the indicator.
        min_n: Minimum paired non-null observations required.

    Returns:
        Dict keyed by other_col with correlation, p_value, n, strength.

    """  # noqa: D205
    is_missing: Series[bool] = df[target_col].isna().astype(float)
    results: dict[str, dict[str, Any]] = {}

    for col in other_cols:
        if col == target_col:
            continue
        paired: DataFrame = pd.concat([is_missing, df[col]], axis=1).dropna()
        if len(paired) < min_n:
            continue
        if paired.iloc[:, 1].nunique() < 2:
            continue
        try:
            r, p = pointbiserialr(
                paired.iloc[:, 0].values,
                paired.iloc[:, 1].values,
            )
            strength: str = (
                "strong" if abs(r) >= 0.4 else "moderate" if abs(r) >= 0.2 else "weak"
            )
            results[col] = {
                "correlation": round(float(r), 4),
                "p_value": round(float(p), 6),
                "n": len(paired),
                "strength": strength,
            }
        except Exception:
            pass

    return results


def _group_difference_tests(
    df: pd.DataFrame,
    target_col: str,
    other_cols: list[str],
    alpha: float,
    min_group_n: int = 5,
) -> dict[str, dict[str, Any]]:
    """
    Test whether observed columns differ between rows where target_col
    is missing vs present (Mann-Whitney U).

    A significant difference means the distribution of another column is
    different when target_col is missing vs observed. This is consistent
    with MAR (the other column may explain the missingness) but does NOT
    rule out MNAR (the missingness may still depend on the unobserved value).

    Args:
        df: Source DataFrame.
        target_col: Column whose missingness defines the groups.
        other_cols: Numeric columns to compare across groups.
        alpha: Significance threshold.
        min_group_n: Minimum group size to run the test.

    Returns:
        Dict keyed by other_col with u_statistic, p_value, significant,
        rank_biserial_r, n_missing_group, n_observed_group.

    """  # noqa: D205
    missing_mask: Series[bool] = df[target_col].isna()
    results: dict[str, dict[str, Any]] = {}

    numeric_cols: DataFrame = df.select_dtypes(include=np.number).columns
    cols_to_test: list[str] = [
        c for c in other_cols if c in numeric_cols and c != target_col
    ]

    for col in cols_to_test:
        group_missing = df.loc[missing_mask, col].dropna().to_numpy()
        group_observed = df.loc[~missing_mask, col].dropna().to_numpy()

        if len(group_missing) < min_group_n or len(group_observed) < min_group_n:
            continue

        try:
            u, p = mannwhitneyu(group_missing, group_observed, alternative="two-sided")
            n1, n2 = len(group_missing), len(group_observed)
            r_rb: float = 1.0 - (2.0 * float(u)) / (n1 * n2)
            results[col] = {
                "u_statistic": round(float(u), 4),
                "p_value": round(float(p), 6),
                "rank_biserial_r": round(float(r_rb), 4),
                "significant": bool(p < alpha),
                "n_missing_group": int(n1),
                "n_observed_group": int(n2),
            }
        except Exception:
            pass

    return results


def _assess_mechanism(
    mcar_result: dict[str, Any] | None,
    correlations: dict[str, dict],
    group_diffs: dict[str, dict],
    alpha: float,
) -> dict[str, Any]:
    """
    Derive a structured mechanism assessment from the evidence.

    Deliberately avoids claiming to confirm any mechanism. Returns a
    ``consistent_with`` field and explicit caveats for every conclusion.

    Args:
        mcar_result: Little's MCAR test result dict (or None).
        correlations: Missingness correlations with other columns.
        group_diffs: Group difference test results.
        alpha: Significance threshold.
        null_pct: Proportion of missing values in this column.

    Returns:
        Dict with consistent_with, confidence, evidence_summary, caveats.

    """
    evidence: list[str] = []
    caveats: list[str] = []
    consistent_with: list[str] = []

    # MCAR evidence
    if mcar_result is not None:
        if mcar_result["p_value"] < alpha:
            evidence.append(
                f"Little's MCAR test rejected (p={mcar_result['p_value']:.4f} "
                f"< {alpha}) - data is not consistent with MCAR.",
            )
        else:
            evidence.append(
                f"Little's MCAR test not rejected (p={mcar_result['p_value']:.4f} "
                f">= {alpha}) - data is consistent with MCAR.",
            )
            consistent_with.append("mcar")
            caveats.append(
                "MCAR not rejected does not mean MCAR is confirmed. "
                "The test may have low power, particularly with small samples.",
            )
    else:
        caveats.append(
            "Little's MCAR test could not be run (insufficient data or "
            "fewer than 2 numeric columns). MCAR cannot be assessed.",
        )

    # MAR evidence: notable correlations with other columns
    notable_corr: dict = {
        c: v
        for c, v in correlations.items()
        if abs(v["correlation"]) >= 0.2 and v["p_value"] < alpha
    }
    if notable_corr:
        top: list[tuple] = sorted(
            notable_corr.items(),
            key=lambda x: -abs(x[1]["correlation"]),
        )[:3]
        top_str: str = "; ".join(f"{c} (r={v['correlation']:.2f})" for c, v in top)
        evidence.append(
            f"Missingness indicator is notably correlated with: {top_str}. "
            f"This is consistent with MAR.",
        )
        consistent_with.append("mar")
        caveats.append(
            "Correlations between missingness and observed columns are "
            "consistent with MAR but cannot distinguish MAR from MNAR. "
            "A column can show these correlations and still be MNAR if the "
            "missingness also depends on the unobserved values.",
        )

    # MAR evidence: significant group differences
    sig_diffs: dict = {c: v for c, v in group_diffs.items() if v["significant"]}
    if sig_diffs:
        diff_cols: list = list(sig_diffs.keys())[:3]
        evidence.append(
            f"Distribution of {diff_cols} differs significantly between rows "
            f"where this column is missing vs observed. Consistent with MAR.",
        )
        if "mar" not in consistent_with:
            consistent_with.append("mar")
        if not any("cannot distinguish" in c for c in caveats):
            caveats.append(
                "Significant group differences are consistent with MAR but "
                "do not rule out MNAR.",
            )

    # MNAR: always note that it cannot be ruled out
    caveats.append(
        "MNAR (Missing Not At Random) cannot be confirmed or ruled out from "
        "observed data alone. If missingness is likely to depend on the "
        "unobserved values themselves (e.g. patients with worse outcomes "
        "less likely to attend follow-up), domain knowledge is the only "
        "reliable guide.",
    )

    if not consistent_with:
        consistent_with = ["indeterminate"]
        evidence.append(
            "Insufficient evidence to indicate a specific mechanism. "
            "Treat missingness as potentially non-random.",
        )

    # Confidence qualifier
    n_evidence: int = len([e for e in evidence if "consistent" in e.lower()])
    confidence: str = (
        "low"
        if n_evidence == 0
        else "moderate"  # never "high" - mechanism analysis is inherently uncertain
    )

    return {
        "consistent_with": consistent_with,
        "confidence": confidence,
        "evidence_summary": evidence,
        "caveats": caveats,
    }


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="missingness_mechanism_analysis",
    display_name="Missingness Mechanism Analysis",
    description=(
        "Analyses the likely mechanism behind missing values (MCAR/MAR/MNAR). "
        "Produces evidence-based assessments with explicit epistemic caveats. "
        "Never claims to confirm MNAR (unverifiable) or MAR (not directly testable)."
    ),
    depends_on=["infer_types", "summarize_nulls"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="moderate",
    tags=["missingness", "mcar", "mar", "mnar", "quality"],
    expected_semantic_types=["any"],
)
class MissingnessMechanismAnalysis(BaseTask):
    """
    Analyse the likely mechanism behind missing values in each column.

    Missing data mechanisms (Rubin 1976) determine which imputation strategies
    are valid and what biases may result from complete-case analysis:

    - **MCAR** (Missing Completely At Random): Missingness is unrelated to
      any data. Complete-case analysis is unbiased. *Testable* via Little's
      test - but only absence of evidence, not evidence of absence.
    - **MAR** (Missing At Random): Missingness depends on observed data only.
      Multiple imputation is valid. *Not directly testable* - only
      evidence consistent with MAR can be found from the data.
    - **MNAR** (Missing Not At Random): Missingness depends on the missing
      value itself. All standard imputation methods introduce bias.
      *Fundamentally unverifiable* from observed data - requires domain
      knowledge.

    **What this task does:**

    1. **Little's MCAR test** - chi-squared test on all numeric columns jointly.
       Reports whether data is inconsistent with MCAR, not whether it is MCAR.
    2. **Missingness correlations** - point-biserial correlation between each
       column's is_missing indicator and other observed columns.
    3. **Group difference tests** - Mann-Whitney U comparing distributions of
       observed columns between rows where a column is missing vs present.
    4. **Mechanism assessment** - structured ``consistent_with`` field
       (never a definitive verdict) with explicit caveats per column.

    **Epistemic design principles:**

    - Confidence is capped at ``"moderate"`` - mechanism analysis is
      inherently uncertain.
    - Every column result includes a ``caveats`` list explaining limitations.
    - MNAR is noted as unverifiable in every column result.
    - Guidance uses language like "consistent with" and "cannot rule out"
      rather than "is" or "confirmed".

    Configurable parameters (via config["tasks"]["missingness_mechanism_analysis"]):
        alpha (float): Significance threshold. Default: 0.05
        min_null_pct (float): Minimum null proportion to analyse a column.
            Default: 0.01
        min_n (int): Minimum non-null observations for group tests. Default: 10
        notable_correlation_threshold (float): |r| above which a correlation
            is considered notable. Default: 0.2
    """

    def run(self) -> None:
        """
        Execute missingness mechanism analysis and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run()

            if not matched_cols:
                self.output = self.make_empty_result(
                    (
                        "No eligible columns found — missingness mechanism analysis"
                        " skipped."
                    ),
                    excluded,
                )
                return

            alpha_raw: Any | None = self.get_task_param("alpha")
            alpha: float = float(alpha_raw) if alpha_raw is not None else 0.05

            min_null_raw: Any | None = self.get_task_param("min_null_pct")
            min_null_pct: float = (
                float(min_null_raw) if min_null_raw is not None else 0.01
            )

            min_n_raw: Any | None = self.get_task_param("min_n")
            min_n: int = int(min_n_raw) if min_n_raw is not None else 10

            corr_thresh_raw: Any | None = self.get_task_param(
                "notable_correlation_threshold",
            )
            corr_threshold: float = (
                float(corr_thresh_raw) if corr_thresh_raw is not None else 0.2
            )

            n_rows: int = len(df)

            # Read null percentages from context if available
            null_pcts: dict[str, float] = {}
            if self.context:
                null_result: TaskResult | None = self.context.results.get(
                    "summarize_nulls",
                )
                if null_result and null_result.status == "success":
                    null_pcts = null_result.data.get("null_percentages") or {}

            if not null_pcts and n_rows > 0:
                null_pcts = {col: df[col].isna().sum() / n_rows for col in df.columns}

            # Columns to analyse: only those with meaningful missingness
            missing_cols: list[str] = [
                col
                for col, pct in null_pcts.items()
                if pct >= min_null_pct and col in df.columns
            ]

            if not missing_cols:
                self._log("    No columns with sufficient missingness.", "debug")
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    summary={
                        "message": "No columns with sufficient missingness to analyse.",
                        "columns_analysed": 0,
                    },
                    data={},
                    metadata={
                        "alpha": alpha,
                        "min_null_pct": min_null_pct,
                        "suggested_viz_type": "table",
                        "recommended_section": "Quality",
                        "display_priority": "medium",
                        "excluded_columns": excluded,
                        "column_types": self.get_column_type_info(
                            matched_cols + list(excluded.keys()),
                        ),
                    },
                )
                return

            self._log(
                f"    {len(missing_cols)} column(s) with sufficient missingness.",
                "debug",
            )

            # Dataset-level: Little's MCAR test on all numeric columns
            little_result: dict[str, Any] | None = _little_mcar_test(df)
            if little_result:
                self._log(
                    f"    Little's MCAR test: "
                    f"statistic={little_result['test_statistic']}, "
                    f"p={little_result['p_value']:.4f}, "
                    f"{little_result['n_patterns']} pattern(s).",
                    "debug",
                )

            all_other_cols: list[str] = list(df.columns)

            results: dict[str, dict[str, Any]] = {}

            for col in missing_cols:
                null_pct: float = null_pcts.get(col, 0.0)

                # Missingness correlations
                correlations: dict[str, dict[str, Any]] = _missingness_correlations(
                    df,
                    col,
                    all_other_cols,
                    min_n=min_n,
                )

                # Group difference tests
                group_diffs: dict[str, dict[str, Any]] = _group_difference_tests(
                    df,
                    col,
                    all_other_cols,
                    alpha=alpha,
                    min_group_n=min_n // 2,
                )

                # Mechanism assessment
                assessment: dict[str, Any] = _assess_mechanism(
                    mcar_result=little_result,
                    correlations=correlations,
                    group_diffs=group_diffs,
                    alpha=alpha,
                )

                results[col] = {
                    "null_pct": round(null_pct, 4),
                    "n_missing": int(df[col].isna().sum()),
                    "n_total": n_rows,
                    "missingness_correlations": correlations,
                    "group_difference_tests": group_diffs,
                    "mechanism_assessment": assessment,
                }

                self._log(
                    f"    '{col}': consistent_with={assessment['consistent_with']}, "
                    f"confidence={assessment['confidence']}",
                    "debug",
                )

            mcar_rejected_overall = (
                little_result is not None and little_result["p_value"] < alpha
            )

            mcar: str = "rejected" if mcar_rejected_overall else "not rejected"

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Missingness mechanism analysis on {len(results)} column(s). "
                        f"MCAR {mcar} "
                        f"at α={alpha}. "
                        f"Note: mechanism diagnosis is inherently uncertain - "
                        f"see per-column caveats."
                    ),
                    "columns_analysed": len(results),
                    "mcar_rejected": mcar_rejected_overall,
                    "alpha": alpha,
                    "little_mcar_test": little_result,
                    "epistemic_note": (
                        "MCAR is the only mechanism with a statistical test. "
                        "MAR is not directly testable. MNAR is unverifiable "
                        "from observed data. All assessments are evidence-based "
                        "hypotheses, not confirmations."
                    ),
                },
                data=results,
                metadata={
                    "alpha": alpha,
                    "min_null_pct": min_null_pct,
                    "notable_correlation_threshold": corr_threshold,
                    "suggested_viz_type": "table",
                    "recommended_section": "Quality",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, col_data in results.items():
                self._attach_guidance(col, col_data)

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
        col_data: dict[str, Any],
    ) -> None:
        """
        Generate EDA guidance for a column's missingness mechanism analysis.

        Args:
            col: Column name.
            col_data: Column result dict.
            alpha: Significance threshold used.

        """
        assessment = col_data["mechanism_assessment"]
        consistent_with = assessment["consistent_with"]
        confidence = assessment["confidence"]
        evidence = assessment["evidence_summary"]
        caveats = assessment["caveats"]
        null_pct = col_data["null_pct"]
        n_missing = col_data["n_missing"]

        consistent_str: str = (
            " / ".join(
                c.upper().replace("_", " ")
                for c in consistent_with
                if c != "indeterminate"
            )
            or "Indeterminate"
        )

        evidence_str: str = "\n".join(f"• {e}" for e in evidence)
        caveats_str: str = "\n".join(f"⚠ {c}" for c in caveats)

        body: str = (
            f"'{col}' has {n_missing:,} missing values ({null_pct:.1%}). "
            f"Mechanism assessment: consistent with {consistent_str} "
            f"(confidence: {confidence}).\n\n"
            f"Evidence:\n{evidence_str}\n\n"
            f"Important caveats:\n{caveats_str}\n\n"
            f"These findings are hypotheses about the missingness mechanism, "
            f"not confirmations. Domain knowledge about how this column was "
            f"collected should take precedence over statistical findings."
        )

        # Imputation recommendations keyed to the mechanism evidence
        actions: list[dict[str, str]] = []
        if "mcar" in consistent_with:
            actions.append(
                {
                    "action": "consider_complete_case",
                    "detail": (
                        "If MCAR is plausible, complete-case analysis introduces "
                        "no systematic bias (but reduces sample size). "
                        "Standard imputation methods are all appropriate."
                    ),
                },
            )
        if "mar" in consistent_with:
            actions.append(
                {
                    "action": "use_multiple_imputation",
                    "detail": (
                        "If MAR is plausible, multiple imputation (e.g. MICE) "
                        "using the correlated columns as predictors produces "
                        "unbiased estimates. Single imputation underestimates "
                        "uncertainty."
                    ),
                },
            )
        actions.append(
            {
                "action": "consult_domain_knowledge",
                "detail": (
                    "Confirm with data collection context whether missingness "
                    "might depend on the unobserved value (MNAR). Statistical "
                    "tests cannot detect MNAR."
                ),
            },
        )

        level: str = "warn" if "indeterminate" in consistent_with else "info"

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=(
                f"Missingness Mechanism: consistent with {consistent_str} "
                f"({null_pct:.1%} missing, confidence: {confidence})"
            ),
            body=body.strip(),
            actions=actions,
            metric={
                "null_pct": null_pct,
                "n_missing": n_missing,
                "consistent_with": consistent_with,
                "confidence": confidence,
            },
        )
