# dsbf/eda/tasks/normality_tests.py

from typing import Any

import numpy as np
from scipy.stats import jarque_bera, kstest, shapiro

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result

# ── Constants ──────────────────────────────────────────────────────────────────

# Shapiro-Wilk is only reliable up to ~5000 observations.
# Above this threshold we switch to KS against a fitted normal distribution.
_SHAPIRO_MAX_N = 5_000

# Minimum observations required to run any normality test meaningfully.
_MIN_N = 8

# Default significance level for pass/fail verdict.
_DEFAULT_ALPHA = 0.05


# ── Per-test runners ───────────────────────────────────────────────────────────


def _run_shapiro(values: np.ndarray) -> dict:
    """
    Run Shapiro-Wilk normality test.

    Args:
        values: 1-D array of non-null numeric values (n ≤ 5000).

    Returns:
        Dict with ``statistic``, ``p_value``, and ``test`` keys.

    """
    stat, p = shapiro(values)
    return {"test": "shapiro_wilk", "statistic": float(stat), "p_value": float(p)}


def _run_ks(values: np.ndarray) -> dict:
    """
    Run one-sample Kolmogorov-Smirnov test against a fitted normal distribution.

    Parameters of the normal (mean, std) are estimated from the sample itself.
    This is a conservative test - fitting on the same data slightly inflates
    the p-value (Lilliefors correction is not applied here).

    Args:
        values: 1-D array of non-null numeric values.

    Returns:
        Dict with ``statistic``, ``p_value``, and ``test`` keys.

    """
    mu, sigma = float(np.mean(values)), float(np.std(values, ddof=1))
    if sigma == 0:
        return {"test": "ks_normal", "statistic": None, "p_value": None}
    stat, p = kstest(values, "norm", args=(mu, sigma))
    return {"test": "ks_normal", "statistic": float(stat), "p_value": float(p)}


def _run_jarque_bera(values: np.ndarray) -> dict:
    """
    Run Jarque-Bera test combining skewness and kurtosis.

    Jarque-Bera is fast and interpretable: it tests whether skewness and
    excess kurtosis jointly match a normal distribution. Asymptotically
    chi-squared with 2 degrees of freedom.

    Args:
        values: 1-D array of non-null numeric values.

    Returns:
        Dict with ``statistic``, ``p_value``, and ``test`` keys.

    """
    stat, p = jarque_bera(values)
    return {"test": "jarque_bera", "statistic": float(stat), "p_value": float(p)}


def _verdict(p_value: float | None, alpha: float) -> str:
    """
    Return a pass/fail verdict string based on p-value vs alpha.

    Args:
        p_value: Test p-value, or None if the test could not run.
        alpha: Significance threshold.

    Returns:
        ``"normal"`` if p ≥ alpha, ``"non_normal"`` if p < alpha,
        ``"inconclusive"`` if p_value is None.

    """
    if p_value is None:
        return "inconclusive"
    return "normal" if p_value >= alpha else "non_normal"


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="normality_tests",
    display_name="Normality Tests",
    description=(
        "Tests each continuous column for normality using Shapiro-Wilk "
        "(n ≤ 5000), Kolmogorov-Smirnov (n > 5000), and Jarque-Bera (all n). "
        "Emits guidance for columns that reject normality."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["distribution", "normality", "statistics", "numeric"],
    expected_semantic_types=["continuous"],
)
class NormalityTests(BaseTask):
    """
    Test each continuous column for normality using three complementary tests.

    Test selection strategy:

    - **Shapiro-Wilk** - optimal for small to medium samples (n ≤ 5000).
      Most powerful normality test for samples in this range. Not used for
      larger samples because it becomes over-sensitive and rejects normality
      for trivial deviations.
    - **Kolmogorov-Smirnov** (one-sample, against fitted normal) - used when
      n > 5000. Parameters (mean, std) are estimated from the data. Note:
      fitting on the same data inflates the p-value slightly; treat KS results
      as conservative.
    - **Jarque-Bera** - always run regardless of sample size. Tests whether
      skewness and excess kurtosis jointly match a normal distribution.
      Fast chi-squared test, useful for quick screening across many columns.

    Each column's output includes results from both the size-appropriate test
    (SW or KS) and JB, plus an overall verdict that is ``"non_normal"`` if
    either test rejects normality at the configured alpha.

    Columns with fewer than ``_MIN_N`` (8) non-null values are skipped.

    EDA guidance is emitted for columns that reject normality, explaining
    which test fired and what the deviation pattern implies. ML guidance
    advises on model families sensitive to the normality assumption.

    Configurable parameters (via config["tasks"]["normality_tests"]):
        alpha (float): Significance level for pass/fail verdict. Default: 0.05
    """

    def run(self) -> None:  # noqa: C901
        """
        Execute normality tests and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run("'continuous'")

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No continuous columns found — normality tests skipped.",
                    excluded,
                )
                return

            alpha_raw: Any | None = self.get_task_param("alpha")
            alpha: float = float(alpha_raw) if alpha_raw is not None else _DEFAULT_ALPHA

            numeric_df = df.select_dtypes(include=np.number)
            results: dict[str, dict] = {}

            for col in numeric_df.columns:
                values = numeric_df[col].dropna().to_numpy()

                if len(values) < _MIN_N:
                    self._log(
                        f"    '{col}' skipped: only {len(values)} non-null values "
                        f"(minimum {_MIN_N}).",
                        "debug",
                    )
                    continue

                n: int = len(values)

                # Size-appropriate primary test
                if n <= _SHAPIRO_MAX_N:
                    primary: dict = _run_shapiro(values)
                else:
                    primary = _run_ks(values)

                # Jarque-Bera always runs as secondary test
                jb: dict = _run_jarque_bera(values)

                primary_verdict: str = _verdict(primary["p_value"], alpha)
                jb_verdict: str = _verdict(jb["p_value"], alpha)

                # Overall verdict: non_normal if either test rejects
                if primary_verdict == "non_normal" or jb_verdict == "non_normal":
                    overall = "non_normal"
                elif primary_verdict == "inconclusive" or jb_verdict == "inconclusive":
                    overall = "inconclusive"
                else:
                    overall = "normal"

                results[col] = {
                    "n": n,
                    "primary_test": primary,
                    "jarque_bera": jb,
                    "primary_verdict": primary_verdict,
                    "jb_verdict": jb_verdict,
                    "overall_verdict": overall,
                    "alpha": alpha,
                }

                self._log(
                    f"    '{col}' (n={n}): {primary['test']} "
                    f"p={primary['p_value']:.4f}, "
                    f"JB p={jb['p_value']:.4f} → {overall}",
                    "debug",
                )

            non_normal_count: int = sum(
                1 for v in results.values() if v["overall_verdict"] == "non_normal"
            )
            self._log(
                f"    {non_normal_count} non-normal column(s) of "
                f"{len(results)} tested.",
                "debug",
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Tested {len(results)} columns for normality; "
                        f"{non_normal_count} rejected normality at α={alpha}."
                    ),
                    "tested_count": len(results),
                    "non_normal_count": non_normal_count,
                    "alpha": alpha,
                },
                data=results,
                metadata={
                    "alpha": alpha,
                    "shapiro_max_n": _SHAPIRO_MAX_N,
                    "suggested_viz_type": "table",
                    "recommended_section": "Distributions",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, info in results.items():
                if info["overall_verdict"] == "non_normal":
                    self._attach_guidance(col, info)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, info: dict) -> None:
        """
        Generate EDA and ML guidance for a column that rejects normality.

        Args:
            col: Column name.
            info: Result dict for this column from the main results dict.

        """
        primary = info["primary_test"]
        jb = info["jarque_bera"]
        alpha = info["alpha"]
        n = info["n"]

        # Describe which test(s) fired
        rejecting_tests: list[str] = []
        if info["primary_verdict"] == "non_normal":
            rejecting_tests.append(
                f"{primary['test'].replace('_', ' ').title()} "
                f"(p={primary['p_value']:.4f})",
            )
        if info["jb_verdict"] == "non_normal":
            rejecting_tests.append(f"Jarque-Bera (p={jb['p_value']:.4f})")
        tests_str: str = (
            " and ".join(rejecting_tests) if rejecting_tests else "normality test"
        )

        eda_body: str = (
            f"'{col}' (n={n}) rejects normality at α={alpha}: {tests_str}. "
            f"This means the distribution is statistically distinguishable from "
            f"a normal distribution - likely due to skewness, heavy or light tails, "
            f"multimodality, or a hard boundary at zero. Inspect the histogram, "
            f"skewness, and kurtosis findings to understand the specific pattern. "
            f"Many analysis methods assume normality in residuals rather than raw "
            f"features, so this finding does not automatically require transformation."
        )

        ml_body: str = (
            f"'{col}' is non-normal (rejected at α={alpha} by {tests_str}). "
            f"Models that explicitly assume normality - linear/logistic regression "
            f"coefficient tests, LDA, Gaussian Naive Bayes - will produce less "
            f"reliable inference. For inference validity, consider a log1p, Box-Cox, "
            f"or Yeo-Johnson transform. Tree-based models (Random Forest, XGBoost, "
            f"LightGBM) are distribution-free and unaffected. Normality of model "
            f"residuals matters more than normality of raw features for linear models."
        )

        metric: dict = {
            "n": n,
            "primary_test": primary["test"],
            "primary_p_value": primary["p_value"],
            "jb_p_value": jb["p_value"],
            "overall_verdict": info["overall_verdict"],
            "alpha": alpha,
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="info",
            title=f"Non-Normal Distribution ({tests_str})",
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level="info",
            title="Normality Assumption May Not Hold",
            body=ml_body.strip(),
            actions=[
                {
                    "action": "transform",
                    "method": "log1p",
                    "column": col,
                    "condition": "right-skewed, all values >= 0",
                },
                {
                    "action": "transform",
                    "method": "box_cox",
                    "column": col,
                    "condition": "all values > 0",
                },
                {
                    "action": "transform",
                    "method": "yeo_johnson",
                    "column": col,
                    "condition": "values may be negative",
                },
            ],
            metric=metric,
        )
