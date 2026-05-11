# dsbf/eda/tasks/detect_string_anomalies.py

import re
from re import Pattern
from typing import Any

import pandas as pd
from pandas import Series

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result

# ── Anomaly detectors ──────────────────────────────────────────────────────────


def _check_mixed_case(series: pd.Series) -> dict | None:
    """
    Detect values that differ only by capitalisation.

    'New York' and 'new york' are treated as two separate categories by pandas
    and any downstream model, silently inflating cardinality and corrupting
    value counts.

    Args:
        series: Non-null string Series.

    Returns:
        Finding dict or None if no mixed-case groups found.

    """
    lowered: Series[str] = series.str.lower()
    # Groups where the lowercased form has more than one distinct original casing
    case_groups: dict[str, list[str]] = {}
    for original, lower in zip(series, lowered, strict=False):
        case_groups.setdefault(lower, set()).add(original)

    mixed: dict[str, list[str]] = {
        k: sorted(v) for k, v in case_groups.items() if len(v) > 1
    }
    if not mixed:
        return None

    # Report the top 5 collision groups by frequency to keep output bounded
    top: list[tuple[str, list[str]]] = sorted(mixed.items(), key=lambda x: -len(x[1]))[
        :5
    ]
    return {
        "type": "mixed_case",
        "affected_groups": len(mixed),
        "examples": dict(top),
    }


def _check_whitespace(series: pd.Series) -> dict | None:
    """
    Detect values with leading or trailing whitespace.

    '  active' and 'active' are different strings - joins, groupbys, and
    value counts will treat them as distinct categories.

    Args:
        series: Non-null string Series.

    Returns:
        Finding dict or None if no whitespace-padded values found.

    """
    padded = series[series != series.str.strip()]
    if padded.empty:
        return None

    count: int = len(padded)
    pct: float = count / len(series)
    examples = padded.unique()[:5].tolist()
    return {
        "type": "leading_trailing_whitespace",
        "affected_count": count,
        "affected_pct": round(pct, 4),
        "examples": examples,
    }


def _check_invisible_chars(series: pd.Series) -> dict | None:
    """
    Detect values containing invisible Unicode characters.

    Zero-width spaces (U+200B), non-breaking spaces (U+00A0), soft hyphens
    (U+00AD), and other control characters are invisible in most displays but
    cause string comparisons to fail silently.

    Args:
        series: Non-null string Series.

    Returns:
        Finding dict or None if no invisible characters found.

    """
    # Matches zero-width joiner/non-joiner, zero-width space, soft hyphen,
    # non-breaking space, and general Unicode control characters (Cc category)
    _INVISIBLE_PATTERN: Pattern[str] = re.compile(
        r"[\u00ad\u00a0\u200b\u200c\u200d\u2060\ufeff]"
        r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
    )

    def _has_invisible(s: str) -> bool:
        return bool(_INVISIBLE_PATTERN.search(s))

    affected = series[series.apply(_has_invisible)]
    if affected.empty:
        return None

    count: int = len(affected)
    pct: float = count / len(series)
    examples: list = []
    for v in affected.unique()[:5]:
        # Show unicode escapes so invisible chars are visible in output
        examples.append(v.encode("unicode_escape").decode("ascii"))

    return {
        "type": "invisible_characters",
        "affected_count": count,
        "affected_pct": round(pct, 4),
        "examples": examples,
    }


def _check_length_outliers(series: pd.Series) -> dict | None:
    """
    Detect values whose string length is a statistical outlier.

    Uses a 3x IQR fence on string lengths. Values far outside the typical
    length range often indicate concatenated fields, error messages, or
    raw JSON/XML accidentally left in a categorical column.

    Args:
        series: Non-null string Series.

    Returns:
        Finding dict or None if no length outliers found.

    """
    lengths: Series[int] = series.str.len()
    q1: float = lengths.quantile(0.25)
    q3: float = lengths.quantile(0.75)
    iqr: float = q3 - q1

    if iqr == 0:
        return None

    upper_fence: float = q3 + 3.0 * iqr
    lower_fence: float = q1 - 3.0 * iqr

    outliers = series[(lengths > upper_fence) | (lengths < lower_fence)]
    if outliers.empty:
        return None

    count: int = len(outliers)
    pct: float = count / len(series)
    examples = outliers.unique()[:3].tolist()
    # Truncate very long examples for readability
    examples: list = [v[:120] + "..." if len(v) > 120 else v for v in examples]

    return {
        "type": "length_outliers",
        "affected_count": count,
        "affected_pct": round(pct, 4),
        "upper_fence": round(upper_fence, 1),
        "lower_fence": round(max(lower_fence, 0), 1),
        "examples": examples,
    }


# Ordered list of (check_fn, finding_type) - all checks run for every column
_CHECKS = [
    _check_mixed_case,
    _check_whitespace,
    _check_invisible_chars,
    _check_length_outliers,
]


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="detect_string_anomalies",
    display_name="Detect String Anomalies",
    description=(
        "Detects inconsistencies within categorical string columns: mixed case, "
        "leading/trailing whitespace, invisible Unicode characters, and "
        "length outliers."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["string", "anomaly", "categorical", "quality"],
    expected_semantic_types=["categorical", "text"],
)
class DetectStringAnomalies(BaseTask):
    """
    Detect structural inconsistencies within categorical string columns.

    Runs four checks on every string-typed column:

    - **Mixed case**: values that differ only by capitalisation
      (e.g. ``"New York"`` vs ``"new york"``). These profile as separate
      categories, silently inflating cardinality and corrupting value counts.
    - **Leading/trailing whitespace**: ``"  active"`` and ``"active"`` are
      different strings - groupbys and joins fail silently.
    - **Invisible Unicode characters**: zero-width spaces (U+200B),
      non-breaking spaces (U+00A0), soft hyphens, and control characters that
      are invisible in most displays but break string equality.
    - **Length outliers**: values whose string length is beyond 3x IQR from
      Q1/Q3, often indicating concatenated fields, error messages, or raw
      structured data (JSON, XML) left in a categorical column.

    Each finding type that fires produces an EDA guidance blurb explaining the
    anomaly and its downstream impact, plus a structured action chip for the
    recommended cleaning step.

    Only non-null values are checked. Columns with fewer than
    ``min_values`` (default 10) non-null values are skipped to avoid
    noise on near-empty columns.

    Configurable parameters (via config["tasks"]["detect_string_anomalies"]):
        min_values (int): Minimum non-null values required to analyse a column.
            Default: 10
    """

    def run(self) -> None:  # noqa: C901
        """
        Execute string anomaly detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run("'categorical'")

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No categorical columns found — string anomaly detection skipped.",
                    excluded,
                )
                return

            min_values_raw: Any | None = self.get_task_param("min_values")
            min_values: int = int(min_values_raw) if min_values_raw is not None else 10

            # Operate on object-dtype columns only - numeric/bool/datetime
            # columns have no string anomalies.
            string_cols = df.select_dtypes(include=["object"]).columns.tolist()

            findings: dict[str, list[dict]] = {}
            cols_with_anomalies = 0

            for col in string_cols:
                series = df[col].dropna().astype(str)

                if len(series) < min_values:
                    self._log(
                        f"    '{col}' skipped: only {len(series)} non-null values "
                        f"(min={min_values}).",
                        "debug",
                    )
                    continue

                col_findings: list[dict] = []
                for check_fn in _CHECKS:
                    try:
                        finding = check_fn(series)
                    except Exception as e:  # noqa: BLE001
                        self._log(
                            f"    '{col}' {check_fn.__name__} failed: "
                            f"{type(e).__name__} - {e}",
                            "debug",
                        )
                        continue
                    if finding is not None:
                        col_findings.append(finding)

                if col_findings:
                    findings[col] = col_findings
                    cols_with_anomalies += 1
                    self._log(
                        f"    '{col}': {len(col_findings)} anomaly type(s) found.",
                        "debug",
                    )

            total_findings: int = sum(len(v) for v in findings.values())
            self._log(
                f"    {cols_with_anomalies} column(s) with anomalies, "
                f"{total_findings} total finding(s).",
                "debug",
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"{cols_with_anomalies} column(s) have string anomalies "
                        f"({total_findings} total finding(s))."
                    ),
                    "columns_with_anomalies": cols_with_anomalies,
                    "total_findings": total_findings,
                },
                data=findings,
                metadata={
                    "min_values": min_values,
                    "suggested_viz_type": "table",
                    "recommended_section": "Quality",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, col_findings in findings.items():
                for finding in col_findings:
                    self._attach_guidance(col, finding)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, finding: dict) -> None:
        """
        Generate an EDA guidance blurb for a single string anomaly finding.

        Args:
            col: Column name.
            finding: Finding dict from one of the check functions, containing
                at minimum a ``type`` key.

        """
        ftype = finding["type"]

        if ftype == "mixed_case":
            n_groups = finding["affected_groups"]
            examples = finding["examples"]
            example_str: str = "; ".join(
                f"{k!r} → {v}" for k, v in list(examples.items())[:3]
            )
            title: str = f"Mixed Case - {n_groups} Collision Group(s)"
            body: str = (
                f"'{col}' contains values that differ only by capitalisation "
                f"({n_groups} collision group(s)). These are treated as separate "
                f"categories - value counts, groupbys, and joins will produce "
                f"incorrect results. Examples: {example_str}. "
                f"Standardise to a single case (e.g. lower) before analysis."
            )
            action: dict[str, str] = {
                "action": "normalise_case",
                "method": "str.lower()",
                "column": col,
                "detail": "Standardise all values to lowercase before analysis",
            }

        elif ftype == "leading_trailing_whitespace":
            count = finding["affected_count"]
            pct: str = f"{finding['affected_pct']:.1%}"
            examples = finding["examples"][:3]
            title = f"Whitespace Padding - {count} Value(s) ({pct})"
            body = (
                f"'{col}' has {count} value(s) ({pct}) with leading or trailing "
                f"whitespace. These are treated as distinct from their trimmed "
                f"counterparts, silently splitting categories. "
                f"Examples: {examples}. "
                f"Apply str.strip() before grouping or encoding."
            )
            action = {
                "action": "strip_whitespace",
                "method": "str.strip()",
                "column": col,
                "detail": "Remove leading/trailing whitespace from all values",
            }

        elif ftype == "invisible_characters":
            count = finding["affected_count"]
            pct = f"{finding['affected_pct']:.1%}"
            title = f"Invisible Characters - {count} Value(s) ({pct})"
            body = (
                f"'{col}' has {count} value(s) ({pct}) containing invisible Unicode "
                f"characters (zero-width spaces, non-breaking spaces, control "
                f"characters). These are invisible in display but break string "
                f"equality - two visually identical values may compare as unequal. "
                f"Strip or normalise Unicode before analysis."
            )
            action = {
                "action": "normalise_unicode",
                "method": "unicodedata.normalize + regex strip",
                "column": col,
                "detail": (
                    "Remove invisible characters: "
                    "re.sub(r'[\\u200b\\u00a0\\u00ad\\u200c\\u200d]', '', val)"
                ),
            }

        else:  # length_outliers
            count = finding["affected_count"]
            pct = f"{finding['affected_pct']:.1%}"
            upper = finding["upper_fence"]
            examples = finding["examples"][:2]
            title = f"Length Outliers - {count} Value(s) ({pct})"
            body = (
                f"'{col}' has {count} value(s) ({pct}) with string lengths far "
                f"outside the typical range (fence: {upper:.0f} chars). "
                f"These may be concatenated fields, error messages, or raw "
                f"structured data (JSON/XML) accidentally stored in a categorical "
                f"column. Examples: {examples}. "
                f"Investigate whether these represent valid data or ingestion errors."
            )
            action = {
                "action": "investigate",
                "column": col,
                "detail": (
                    f"Inspect values longer than {upper:.0f} characters - "
                    "they may need to be split, truncated, or excluded"
                ),
            }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="warn" if ftype in ("mixed_case", "invisible_characters") else "info",
            title=title,
            body=body.strip(),
            actions=[action],
            metric={k: v for k, v in finding.items() if k != "examples"},
        )
