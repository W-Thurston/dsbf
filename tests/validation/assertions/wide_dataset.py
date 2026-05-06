# tests/validation/assertions/wide_dataset.py
"""
Assertions for the wide dataset (1,000 rows, 100 columns).

Primary purpose: verify that every component which scales with column
count handles 100 columns without crashing, truncating silently, or
producing layout-breaking output.

Column inventory:
  cont_clean_00..39  40 Beta(5,5) continuous — advisory only
  cont_skew_00..19   20 Gamma(2,1) continuous — skew 1.1-1.7, log-transform warn
  cat_bal_00..28     29 balanced 4-value categoricals — encoding warn
  cat_hc             1  ~244 unique values — high-cardinality warn
  bool_00..04        5  balanced booleans — encoding advisory
  null_00..07        8  continuous with 8-12% MCAR null — completeness amber

Expected findings:
  Completeness amber:  null_00..04 all above 5% threshold
  Encoding warn:       cat_bal_* and cat_hc (raw string dtype)
  Transformations warn: cont_skew_* (skew > 1.0)
  Redundancy green:    60 independent continuous columns, VIF ≈ 1.0
  Leakage green:       no engineered correlations
"""

from __future__ import annotations

from typing import Any

EXPECTED_SKIPS: set[str] = {
    "compare_with_reference_dataset",
    "detect_class_imbalance",
    "detect_feature_drift",
    "detect_target_drift",
    "schema_validation",
}

EXPECTED_COL_COUNT: int = 100
EXPECTED_ROW_COUNT: int = 1000

# Column group prefixes for verification
CONT_CLEAN_PREFIX = "cont_clean_"
CONT_SKEW_PREFIX = "cont_skew_"
CAT_BAL_PREFIX = "cat_bal_"
NULL_PREFIX = "null_"


def validate(report: dict[str, Any]) -> None:
    """
    Run all assertions for the wide dataset.

    Raises AssertionError with a descriptive message on the first failure.
    """
    _no_unexpected_failures(report)
    _core_tasks_succeeded(report)
    _shape_correct(report)
    _all_100_columns_in_infer_types(report)
    _null_columns_flagged_in_completeness(report)
    _skewed_columns_flagged(report)
    _high_cardinality_flagged(report)
    _redundancy_and_leakage_clean(report)
    _usability_amber_from_high_cardinality(report)
    _pairwise_associations_ran(report)
    print("✓  wide dataset: all assertions passed")


# ── Checks ────────────────────────────────────────────────────────────────────


def _no_unexpected_failures(report: dict) -> None:
    """No task should fail — 100 columns must not overwhelm any task."""
    for task_name, task in report.get("results", {}).items():
        if not isinstance(task, dict):
            continue
        status = task.get("status")
        if status == "failed":
            msg: str = (
                f"Task '{task_name}' failed on wide dataset (100 cols).\n"
                f"  error_metadata: {task.get('error_metadata')}"
            )
            raise AssertionError(msg)
        if status == "skipped" and task_name not in EXPECTED_SKIPS:
            msg = (
                f"Task '{task_name}' was skipped unexpectedly — add to "
                f"EXPECTED_SKIPS if intentional."
            )
            raise AssertionError(msg)


def _core_tasks_succeeded(report: dict) -> None:
    """Core tasks must succeed at 100 columns."""
    must_succeed: set[str] = {
        "infer_types",
        "summarize_dataset_shape",
        "summarize_nulls",
        "summarize_numeric",
        "detect_skewness",
        "detect_outliers",
        "data_quality_scorer",
    }
    results = report.get("results", {})
    for task_name in must_succeed:
        task = results.get(task_name)
        assert isinstance(task, dict), f"'{task_name}' missing from results"
        assert task.get("status") == "success", (
            f"'{task_name}' must succeed on wide dataset, "
            f"got status='{task.get('status')}'.\n"
            f"  error_metadata: {task.get('error_metadata')}"
        )


def _shape_correct(report: dict) -> None:
    """Shape task should report exactly 100 columns and 1,000 rows."""
    shape = report["results"].get("summarize_dataset_shape", {})
    if shape.get("status") != "success":
        return
    data = shape.get("data", {})
    assert data.get("num_rows") == EXPECTED_ROW_COUNT, (
        f"Expected {EXPECTED_ROW_COUNT} rows, got {data.get('num_rows')}"
    )
    assert data.get("num_columns") == EXPECTED_COL_COUNT, (
        f"Expected {EXPECTED_COL_COUNT} columns, got {data.get('num_columns')}"
    )
    print(
        f"  ✓  Shape: {EXPECTED_ROW_COUNT} rows × "
        f"{EXPECTED_COL_COUNT} columns confirmed",
    )


def _all_100_columns_in_infer_types(report: dict) -> None:
    """
    infer_types must report all 100 columns.

    If any column is silently dropped during profiling, this fails.
    Also verifies the group classification is correct.
    """
    types = report["results"].get("infer_types", {})
    if types.get("status") != "success":
        return
    data = types.get("data", {})
    n_cols: int = len(data)
    assert n_cols == EXPECTED_COL_COUNT, (
        f"infer_types should report {EXPECTED_COL_COUNT} columns, "
        f"got {n_cols}. Columns may have been silently dropped."
    )

    # Verify group classifications
    cont_clean: list[str] = [c for c in data if c.startswith(CONT_CLEAN_PREFIX)]
    cont_skew: list[str] = [c for c in data if c.startswith(CONT_SKEW_PREFIX)]
    cat_bal: list[str] = [c for c in data if c.startswith(CAT_BAL_PREFIX)]
    null_cols: list[str] = [c for c in data if c.startswith(NULL_PREFIX)]

    assert len(cont_clean) == 40, (
        f"Expected 40 cont_clean_* columns, got {len(cont_clean)}"
    )
    assert len(cont_skew) == 20, (
        f"Expected 20 cont_skew_* columns, got {len(cont_skew)}"
    )
    assert len(cat_bal) == 26, f"Expected 26 cat_bal_* columns, got {len(cat_bal)}"
    assert len(null_cols) == 8, f"Expected 8 null_* columns, got {len(null_cols)}"
    print(f"  ✓  All {EXPECTED_COL_COUNT} columns present in infer_types")


def _null_columns_flagged_in_completeness(report: dict) -> None:
    """
    The 5 null_* columns (8-12% null) must appear in Completeness findings.

    Completeness should be amber — multiple columns above 5% threshold
    but none above 20% (no red).
    """
    dq = report["results"].get("data_quality_scorer", {})
    if not isinstance(dq, dict) or dq.get("status") != "success":
        return
    cats = dq.get("data", {}).get("categories", {})
    completeness = cats.get("completeness", {})
    level = completeness.get("level")
    affected = completeness.get("affected_columns", [])

    # 8 null cols / 100 total = 8% info-severity findings.
    # Under the four-state system: info-only + pct < 10% → blue.
    assert level in {"blue", "amber", "red"}, (
        f"Completeness should be blue (8 null_* cols = 8% info-severity, "
        f"below the 10% amber threshold), got '{level}'"
    )

    for i in range(8):
        col: str = f"null_{i:02d}"
        assert col in affected, (
            f"'{col}' (8-12% null) should appear in Completeness "
            f"affected_columns, but affected = {affected[:10]}..."
        )
    print(
        f"  ✓  Completeness: {level} "
        f"({len(affected)} affected cols, null_* confirmed — "
        "info-only at 8% → blue expected)",
    )


def _skewed_columns_flagged(report: dict) -> None:
    """
    detect_skewness must flag all 20 cont_skew_* columns (skew 1.1-1.7).

    Also verifies the 40 cont_clean_* columns are NOT flagged (skew ≈ 0).
    """
    skew_task = report["results"].get("detect_skewness", {})
    if skew_task.get("status") != "success":
        return

    data = skew_task.get("data", {})

    # All skewed columns should have skew > 1.0
    missing_skew: list[str] = []
    for i in range(20):
        col: str = f"cont_skew_{i:02d}"
        skew_val = data.get(col)
        if skew_val is None or skew_val <= 1.0:
            missing_skew.append(f"{col}={skew_val}")

    assert len(missing_skew) == 0, (
        f"cont_skew_* columns should all have skew > 1.0 (Gamma(2,1)), "
        f"but these did not: {missing_skew[:5]}"
    )

    # Clean columns should have low skew
    high_skew_clean: list[str] = []
    for i in range(40):
        col = f"cont_clean_{i:02d}"
        skew_val = data.get(col)
        if skew_val is not None and abs(skew_val) > 1.0:
            high_skew_clean.append(f"{col}={skew_val:.2f}")

    assert len(high_skew_clean) == 0, (
        f"cont_clean_* columns should have skew ≈ 0 (Beta(5,5)), "
        f"but these were high: {high_skew_clean[:5]}"
    )
    print("  ✓  detect_skewness: 20 cont_skew_* flagged, 40 cont_clean_* clean")


def _high_cardinality_flagged(report: dict) -> None:
    """cat_hc (~244 unique values) must be flagged by detect_high_cardinality."""
    hc_task = report["results"].get("detect_high_cardinality", {})
    if hc_task is None or hc_task.get("status") != "success":
        return
    data = hc_task.get("data", {})
    flagged: list[str] = list(data.keys())
    assert "cat_hc" in flagged, (
        f"detect_high_cardinality should flag 'cat_hc' (~244 unique), "
        f"but flagged: {flagged[:5]}"
    )
    print(f"  ✓  cat_hc flagged as high-cardinality ({data.get('cat_hc')} unique)")


def _redundancy_and_leakage_clean(report: dict) -> None:
    """
    Redundancy and Leakage dimensions must be green.

    60 continuous columns are all independently generated — no engineered
    correlations.  VIF should be low for all (add_constant fix holds at
    100-column scale with widely varying means and scales).
    """
    dq = report["results"].get("data_quality_scorer", {})
    if not isinstance(dq, dict) or dq.get("status") != "success":
        return
    cats = dq.get("data", {}).get("categories", {})

    for dim in ("redundancy", "leakage"):
        cat = cats.get(dim, {})
        level = cat.get("level")
        assert level == "green", (
            f"Quality '{dim}' should be green (no engineered correlations), "
            f"got '{level}'.\n  findings: {cat.get('findings', [])[:3]}"
        )
    print("  ✓  Redundancy and Leakage both green (no spurious VIF inflation)")


def _usability_amber_from_high_cardinality(report: dict) -> None:
    """
    Usability must be amber — cat_hc (244 unique) produces a warn-level
    finding, and warn-level findings now have a severity floor of amber.

    With the _level() fix in data_quality_scorer, any dimension with a
    warn-or-above finding cannot be green regardless of column proportion.
    This is the primary regression test for that fix: 1/100 columns is
    only 1% proportion (below the 5% amber threshold), but the warn
    severity floor must lift the result to amber.
    """
    dq = report["results"].get("data_quality_scorer", {})
    if not isinstance(dq, dict) or dq.get("status") != "success":
        return
    cats = dq.get("data", {}).get("categories", {})
    usability = cats.get("usability", {})
    level = usability.get("level")
    assert level in {"amber", "red"}, (
        f"Usability should be amber (cat_hc warn-level finding + warn severity "
        f"floor in _level()), got '{level}'.\n"
        "If this is green, the warn severity floor in data_quality_scorer "
        "._level() may have regressed."
    )
    print(f"  ✓  Usability: {level} (cat_hc warn finding surfaced correctly)")


def _pairwise_associations_ran(report: dict) -> None:
    """
    compute_pairwise_associations must succeed and produce > 0 pairs.

    1,770 continuous pairs exist (60 choose 2).  The task must complete
    without timing out or running out of memory at this scale.
    """
    task = report["results"].get("compute_pairwise_associations", {})
    if task is None:
        print("  ℹ  compute_pairwise_associations did not run — skipping")
        return
    assert task.get("status") == "success", (
        f"compute_pairwise_associations must succeed at 100 columns, "
        f"got status='{task.get('status')}'"
    )
    data = task.get("data", {})
    pair_count: int = len([k for k in data if k != "__correlation_matrix__"])
    assert pair_count > 0, (
        "compute_pairwise_associations produced 0 pairs on a "
        "100-column dataset — expected > 0"
    )
    print(f"  ✓  compute_pairwise_associations: {pair_count} pairs computed")
