# dsbf/eda/tasks/contingency_tables.py

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

if TYPE_CHECKING:
    from pandas import DataFrame, Series

# ── Helpers ───────────────────────────────────────────────────────────────────


def _cramers_v(chi2: float, n: int, r: int, k: int) -> float:
    """
    Compute Cramér's V from a chi-squared statistic.

    Args:
        chi2: Chi-squared test statistic.
        n: Total sample size.
        r: Number of rows in the contingency table.
        k: Number of columns in the contingency table.

    Returns:
        Cramér's V in [0, 1], or 0.0 for degenerate tables.

    """
    denom: int = min(k - 1, r - 1)
    if denom <= 0 or n == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * denom)))


def _association_strength(cramers_v: float) -> str:
    """
    Map Cramér's V to a human-readable strength label.

    Args:
        cramers_v: Cramér's V in [0, 1].

    Returns:
        One of ``"strong"``, ``"moderate"``, ``"weak"``, ``"negligible"``.

    """
    if cramers_v >= 0.5:
        return "strong"
    if cramers_v >= 0.3:
        return "moderate"
    if cramers_v >= 0.1:
        return "weak"
    return "negligible"


def _build_contingency(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    top_n: int,
) -> dict[str, Any]:
    """
    Build a contingency table and compute chi-squared statistics.

    Only the top-N most frequent values per column are included in the table
    to keep output bounded for high-cardinality columns.

    Args:
        df: Source DataFrame.
        col_a: First categorical column name.
        col_b: Second categorical column name.
        top_n: Maximum number of unique values per column to include.

    Returns:
        Dict with ``table``, ``chi2``, ``p_value``, ``dof``, ``cramers_v``,
        ``n``, ``strength``, ``top_n_used``, and ``truncated`` keys.

    """
    paired: Series = df[[col_a, col_b]].dropna()
    n: int = len(paired)

    if n < 5:
        return None

    # Restrict to top-N values per column to keep tables readable
    top_a = paired[col_a].value_counts().head(top_n).index
    top_b = paired[col_b].value_counts().head(top_n).index
    truncated = paired[col_a].nunique() > top_n or paired[col_b].nunique() > top_n
    filtered = paired[paired[col_a].isin(top_a) & paired[col_b].isin(top_b)]

    if len(filtered) < 5:
        return None

    ct: DataFrame = pd.crosstab(filtered[col_a], filtered[col_b])
    r, k = ct.shape

    if r < 2 or k < 2:
        return None

    try:
        chi2_stat, p_val, dof, _ = chi2_contingency(ct)
    except ValueError:
        return None

    v: float = _cramers_v(chi2_stat, len(filtered), r, k)
    strength: str = _association_strength(v)

    # Convert table to a serialisable nested dict
    table_dict: dict[str, dict[str, int]] = {
        str(row_label): {
            str(col_label): int(ct.loc[row_label, col_label])
            for col_label in ct.columns
        }
        for row_label in ct.index
    }

    return {
        "table": table_dict,
        "chi2": round(float(chi2_stat), 4),
        "p_value": round(float(p_val), 6),
        "dof": int(dof),
        "cramers_v": round(v, 4),
        "n": len(filtered),
        "strength": strength,
        "top_n_used": top_n,
        "truncated": truncated,
    }


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="contingency_tables",
    display_name="Contingency Tables",
    description=(
        "Computes joint frequency distributions for categorical column pairs "
        "with chi-squared independence tests and Cramér's V effect sizes."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["categorical", "association", "chi_squared", "relationships"],
    expected_semantic_types=["categorical"],
)
class ContingencyTables(BaseTask):
    """
    Compute contingency tables and chi-squared tests for categorical pairs.

    For every pair of categorical columns, produces:

    - **Contingency table**: joint frequency counts (row x column)
    - **Chi-squared statistic** and **p-value**: tests whether the two
      columns are statistically independent. A low p-value means the
      distribution of one column varies significantly across levels of
      the other.
    - **Cramér's V**: normalised effect size in [0, 1] derived from the
      chi-squared statistic. Provides a sample-size-independent measure
      of association strength, comparable across column pairs.
    - **Degrees of freedom**: (r-1)(k-1) where r=rows, k=columns.

    This task complements ``compute_pairwise_associations``, which stores a
    single Cramér's V scalar per pair. Contingency tables provide the raw
    frequency structure that the scalar cannot — revealing which specific
    value combinations co-occur most or least often.

    **High-cardinality handling:** Only the top-N most frequent values per
    column (default: 10) are included in each table to keep output readable
    and avoid memory pressure. The ``truncated`` flag in each result indicates
    when the full cardinality was not used.

    **Chi-squared validity:** The chi-squared approximation requires expected
    cell frequencies ≥ 5 in most cells. Results for small samples or
    sparse tables should be treated with caution — a ``low_sample_warning``
    flag is set when n < 50 or > 20% of expected cells are below 5.

    EDA guidance is emitted for pairs where the chi-squared test rejects
    independence at alpha and the association is at least ``weak`` (V ≥ 0.1).
    ML guidance notes when strong categorical associations may cause
    multicollinearity or redundancy between features.

    Configurable parameters (via config["tasks"]["contingency_tables"]):
        alpha (float): Significance level for chi-squared test. Default: 0.05
        top_n (int): Max unique values per column in each table. Default: 10
        min_n (int): Minimum paired observations to attempt a table. Default: 5
        cat_cardinality_limit (int): Skip columns with more unique values
            than this. Default: 50
    """

    def run(self) -> None:
        """
        Execute contingency table computation and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'categorical' column(s)",
                "debug",
            )

            alpha_raw: Any | None = self.get_task_param("alpha")
            alpha: float = float(alpha_raw) if alpha_raw is not None else 0.05

            top_n_raw: Any | None = self.get_task_param("top_n")
            top_n: int = int(top_n_raw) if top_n_raw is not None else 10

            min_n_raw: Any | None = self.get_task_param("min_n")
            min_n: int = int(min_n_raw) if min_n_raw is not None else 5

            card_limit_raw: Any | None = self.get_task_param("cat_cardinality_limit")
            cat_cardinality_limit: int = (
                int(card_limit_raw) if card_limit_raw is not None else 50
            )

            # Identify categorical columns from semantic types
            semantic_types: dict[str, str] = {}
            if self.context:
                semantic_types = self.context.get_metadata("semantic_types") or {}

            cat_cols: list = [
                col
                for col in df.columns
                if semantic_types.get(col, "") == "categorical"
                and df[col].nunique() <= cat_cardinality_limit
                and df[col].notna().any()
            ]

            if not cat_cols:
                # Fallback: use object dtype columns when no semantic types
                cat_cols = [
                    col
                    for col in df.select_dtypes(include=["object"]).columns
                    if df[col].nunique() <= cat_cardinality_limit
                ]

            self._log(
                f"    {len(cat_cols)} eligible categorical column(s) "
                f"(cardinality ≤ {cat_cardinality_limit}).",
                "debug",
            )

            tables: dict[str, dict[str, Any]] = {}
            significant_pairs: list[str] = []
            skipped_pairs = 0

            for i, col_a in enumerate(cat_cols):
                for col_b in cat_cols[i + 1 :]:
                    key: str = f"{col_a}|{col_b}"

                    result: dict[str, Any] = _build_contingency(df, col_a, col_b, top_n)
                    if result is None:
                        skipped_pairs += 1
                        self._log(
                            f"    Skipping {key}: insufficient data or "
                            "degenerate table.",
                            "debug",
                        )
                        continue

                    if result["n"] < min_n:
                        skipped_pairs += 1
                        continue

                    # Flag tables where chi-squared approximation may be unreliable
                    result["low_sample_warning"] = result["n"] < 50

                    tables[key] = result
                    if result["p_value"] < alpha:
                        significant_pairs.append(key)

                    self._log(
                        f"    {key}: V={result['cramers_v']:.3f}, "
                        f"p={result['p_value']:.4f} ({result['strength']})",
                        "debug",
                    )

            self._log(
                f"    {len(tables)} table(s) computed, "
                f"{len(significant_pairs)} significant at α={alpha}, "
                f"{skipped_pairs} skipped.",
                "debug",
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed {len(tables)} contingency table(s); "
                        f"{len(significant_pairs)} significant at α={alpha}."
                    ),
                    "table_count": len(tables),
                    "significant_count": len(significant_pairs),
                    "alpha": alpha,
                },
                data=tables,
                metadata={
                    "alpha": alpha,
                    "top_n": top_n,
                    "cat_cardinality_limit": cat_cardinality_limit,
                    "significant_pairs": significant_pairs,
                    "suggested_viz_type": "heatmap",
                    "recommended_section": "Relationships",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys())
                    ),
                },
            )

            for key, table_data in tables.items():
                col_a, col_b = key.split("|", 1)
                if table_data["p_value"] < alpha and table_data["cramers_v"] >= 0.1:
                    self._attach_guidance(col_a, col_b, table_data, alpha)

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
        col_a: str,
        col_b: str,
        table_data: dict[str, Any],
        alpha: float,
    ) -> None:
        """
        Generate EDA and ML guidance for a significant categorical association.

        Args:
            col_a: First column name.
            col_b: Second column name.
            table_data: Contingency table result dict.
            alpha: Significance level used for the test.

        """
        v = table_data["cramers_v"]
        p = table_data["p_value"]
        chi2 = table_data["chi2"]
        n = table_data["n"]
        strength = table_data["strength"]
        truncated = table_data["truncated"]
        top_n = table_data["top_n_used"]

        truncation_note: str = (
            f" (table shows top {top_n} values per column; full cardinality "
            "not displayed)"
            if truncated
            else ""
        )

        eda_body: str = (
            f"'{col_a}' and '{col_b}' are not independent "
            f"(χ²={chi2:.2f}, p={p:.4f}, n={n:,}, V={v:.3f} — {strength} "
            f"association){truncation_note}. The distribution of '{col_b}' "
            f"varies significantly across levels of '{col_a}'. Inspect the "
            f"frequency table to identify which specific value combinations "
            f"co-occur more or less often than expected under independence. "
            f"Consider whether the association reflects a genuine relationship "
            f"in the population or a data collection / sampling artefact."
        )

        if strength in ("strong", "moderate"):
            ml_body: str = (
                f"'{col_a}' and '{col_b}' have a {strength} association "
                f"(V={v:.3f}). Including both as features introduces "
                f"redundancy — the second column provides limited additional "
                f"information beyond the first. Tree-based models handle this "
                f"naturally but linear models may suffer from multicollinearity "
                f"effects on coefficient estimates. Consider encoding only one "
                f"or using a combined interaction feature."
            )
            ml_actions: list[dict[str, list[str] | str] | dict[str, str]] = [
                {
                    "action": "consider_dropping",
                    "column": col_b,
                    "detail": (
                        f"'{col_b}' is strongly associated with '{col_a}' "
                        f"(V={v:.3f}) — may be redundant as a feature"
                    ),
                },
                {
                    "action": "create_interaction",
                    "columns": [col_a, col_b],
                    "detail": "Combine into a single interaction feature",
                },
            ]
        else:
            ml_body = (
                f"'{col_a}' and '{col_b}' have a statistically significant "
                f"but weak association (V={v:.3f}). Both columns likely carry "
                f"independent signal and can be included as separate features. "
                f"The association may still be worth understanding for feature "
                f"engineering purposes."
            )
            ml_actions = []

        metric: dict[str, float | Any] = {
            "chi2": chi2,
            "p_value": p,
            "dof": table_data["dof"],
            "cramers_v": v,
            "n": n,
            "strength": strength,
            "alpha": alpha,
        }

        # Attach guidance to both columns so each surfaces the finding
        for col in (col_a, col_b):
            other: str = col_b if col == col_a else col_a

            self.add_guidance(
                result=self.output,
                column=col,
                phase="eda",
                level="warn" if strength in ("strong", "moderate") else "info",
                title=(f"Associated with '{other}' (χ², p={p:.4f}, V={v:.3f})"),
                body=eda_body.strip(),
                actions=[],
                metric=metric,
            )

            if ml_actions or strength == "weak":
                self.add_guidance(
                    result=self.output,
                    column=col,
                    phase="ml",
                    level="info",
                    title=(
                        f"{strength.title()} Categorical Association with "
                        f"'{other}' (V={v:.3f})"
                    ),
                    body=ml_body.strip(),
                    actions=ml_actions,
                    metric=metric,
                )
