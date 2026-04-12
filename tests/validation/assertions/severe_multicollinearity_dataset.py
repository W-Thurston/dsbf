# tests/validation/assertions/severe_multicollinearity_dataset.py
"""
Assertions for the severe-multicollinearity dataset (2,000 rows, 11 columns).

All columns are continuous.  The dataset targets four distinct VIF failure
modes — each block exercises a different way VIF computation can produce
wrong or misleading results in real-world data.

Block A — Classic severe collinearity (r=0.96-0.99)
  base, copy_a, copy_b, combo  ->  VIF 16-84, all should be flagged

Block B — Scale mismatch: independent columns at extreme different scales
  revenue    N(500k, 200k)   — genuinely independent, VIF should be ~1.0
  click_rate N(0.035, 0.012) — genuinely independent, VIF should be ~1.0
  Under the broken code (no add_constant), large nonzero means inflate
  VIF to 6-8, producing spurious warn-level findings for both columns.
  This is the primary regression test for the add_constant fix.

Block C — All-collinear trio (no independent column in subgroup)
  factor_x, factor_y, factor_z  ->  all derived from same latent factor

Block D — Sparse continuous column (65% NaN reduces dropna to ~35% of rows)
  signal, sparse_c  ->  correlated among observed rows, VIF ~36

Side effects (correct profiler behavior, not bugs):
  - base and copy_a (r≈0.99) also trigger detect_data_leakage → Leakage red
  - sparse_c 65% NaN triggers Completeness amber
  - gate = not_ready (error-level leakage findings from the near-perfect copies)
  - ML Readiness has no 'redundancy' dimension — collinearity is a Quality
    scorer concern only (data quality issue, not a pre-modeling action blocker)
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

# Blocks A + C: should all be flagged by detect_collinear_features (VIF >> 10)
EXPECTED_COLLINEAR: set[str] = {
    "base",
    "copy_a",
    "copy_b",
    "combo",
    "factor_x",
    "factor_y",
    "factor_z",
}

# Block B: genuinely independent despite extreme scales —
# the add_constant fix ensures these are NOT flagged
SCALE_MISMATCH_COLS: set[str] = {"revenue", "click_rate"}


def validate(report: dict[str, Any]) -> None:
    """
    Run all assertions for the severe-multicollinearity dataset.

    Raises AssertionError with a descriptive message on the first failure.
    """
    _no_unexpected_failures(report)
    _core_tasks_succeeded(report)
    _shape_correct(report)
    _all_columns_continuous(report)
    _block_a_c_collinear_columns_flagged(report)
    _block_b_scale_mismatch_not_flagged(report)
    _redundancy_dimension_red(report)
    _leakage_dimension_red(report)
    _completeness_from_sparse_c(report)
    _ml_readiness_gate_not_ready(report)
    print("✓  severe_multicollinearity dataset: all assertions passed")


# ── Checks ────────────────────────────────────────────────────────────────────


def _no_unexpected_failures(report: dict) -> None:
    """
    No task should fail — VIF must remain numerically stable across all
    four blocks including scale-mismatch and sparse-column cases.
    """
    for task_name, task in report.get("results", {}).items():
        if not isinstance(task, dict):
            continue
        status = task.get("status")
        if status == "failed":
            msg: str = (
                f"Task '{task_name}' failed on severe-multicollinearity data.\n"
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
    """Core tasks must succeed on this all-continuous dataset."""
    must_succeed: set[str] = {
        "infer_types",
        "summarize_dataset_shape",
        "summarize_nulls",
        "summarize_numeric",
        "detect_collinear_features",
        "data_quality_scorer",
    }
    results = report.get("results", {})
    for task_name in must_succeed:
        task = results.get(task_name)
        assert isinstance(task, dict), f"'{task_name}' missing from results"
        assert task.get("status") == "success", (
            f"'{task_name}' must succeed, got '{task.get('status')}'.\n"
            f"  error_metadata: {task.get('error_metadata')}"
        )


def _shape_correct(report: dict) -> None:
    """Shape task should report 2000 rows and 11 columns."""
    shape = report["results"].get("summarize_dataset_shape", {})
    if shape.get("status") != "success":
        return
    data = shape.get("data", {})
    assert (
        data.get("num_rows") == 2000
    ), f"Expected 2000 rows, got {data.get('num_rows')}"
    assert (
        data.get("num_columns") == 11
    ), f"Expected 11 columns, got {data.get('num_columns')}"


def _all_columns_continuous(report: dict) -> None:
    """
    infer_types must classify all columns as continuous.

    If any column is misclassified, the matched_cols filter would exclude
    it from VIF and the block assertions below would be vacuous.
    """
    types = report["results"].get("infer_types", {})
    if types.get("status") != "success":
        return
    non_continuous: list[str] = [
        col
        for col, info in types.get("data", {}).items()
        if info.get("analysis_intent_dtype") != "continuous"
    ]
    assert (
        len(non_continuous) == 0
    ), f"All 11 columns should be continuous, but {non_continuous} were not."


def _block_a_c_collinear_columns_flagged(report: dict) -> None:
    """
    Blocks A and C must all appear in detect_collinear_features output.

    Block A (base, copy_a, copy_b, combo): r=0.96-0.99, VIF 16-84.
    Block C (factor_x/y/z): all-collinear trio, VIF > 10.
    """
    task = report["results"].get("detect_collinear_features", {})
    if task.get("status") != "success":
        print(
            "  ℹ  detect_collinear_features unavailable — "
            "collinear column assertions skipped",
        )
        return

    data = task.get("data", {})
    flagged: list[str] = data.get("collinear_columns", [])
    vif_scores: dict = data.get("vif_scores", {})

    missing: list[str] = [col for col in EXPECTED_COLLINEAR if col not in flagged]
    all_vif: dict = {k: round(v, 1) for k, v in vif_scores.items()}
    assert len(missing) == 0, (
        f"Expected collinear columns not flagged: {missing}.\n"
        f"  flagged: {flagged}\n"
        f"  vif_scores: {all_vif}"
    )
    relevant: dict = {
        c: round(vif_scores.get(c, 0), 1) for c in sorted(EXPECTED_COLLINEAR)
    }
    print(f"  ✓  Blocks A+C collinear columns flagged: {relevant}")


def _block_b_scale_mismatch_not_flagged(report: dict) -> None:
    """
    Block B — revenue and click_rate must NOT be flagged as collinear.

    These two columns are genuinely independent — they share no real
    linear relationship.  Their only unusual property is extreme scale
    difference (revenue mean ~500k vs click_rate mean ~0.035).

    Under the broken VIF code (no add_constant), the intercept-less
    regression treats their large nonzero means as shared variance,
    inflating VIF to 6-8 and producing spurious warn-level findings.
    Under the fixed code (add_constant), both should score VIF < 3.0.

    This is the primary regression test for the add_constant fix: if
    revenue or click_rate appear in collinear_columns, the fix has
    regressed and the live bug is back.
    """
    task = report["results"].get("detect_collinear_features", {})
    if task.get("status") != "success":
        return

    data = task.get("data", {})
    flagged: list[str] = data.get("collinear_columns", [])
    vif_scores: dict = data.get("vif_scores", {})

    wrongly_flagged: list[str] = [col for col in SCALE_MISMATCH_COLS if col in flagged]
    all_vif_b: dict = {k: round(v, 1) for k, v in vif_scores.items()}
    assert len(wrongly_flagged) == 0, (
        f"Scale-mismatch columns incorrectly flagged as collinear: "
        f"{wrongly_flagged}.\n"
        f"  vif_scores: {all_vif_b}\n"
        f"\n"
        f"  These columns are genuinely independent. A high VIF here means\n"
        f"  the add_constant fix has regressed. Without add_constant, large\n"
        f"  nonzero means are mistaken for shared variance by the intercept-\n"
        f"  less regression."
    )

    # Assert VIF values are low (< 3) — not just absent from flagged list.
    # VIF of 5-9 would still indicate partial regression worth catching.
    for col in SCALE_MISMATCH_COLS:
        vif = vif_scores.get(col)
        if vif is None:
            continue
        assert vif < 3.0, (
            f"'{col}' VIF={vif:.2f} but expected < 3.0 for a genuinely "
            f"independent column. A value >= 3.0 suggests the add_constant "
            f"fix may have partially regressed."
        )

    vifs: dict = {
        c: round(vif_scores.get(c, 0), 3) for c in sorted(SCALE_MISMATCH_COLS)
    }
    print(f"  ✓  Block B scale-mismatch columns correctly not flagged: {vifs}")


def _redundancy_dimension_red(report: dict) -> None:
    """
    The Quality scorer's Redundancy dimension must be red.

    Note: detect_collinear_features routes to the Quality scorer's
    Redundancy dimension only — not to ML Readiness, which has no
    redundancy dimension (collinearity is a data quality concern,
    not a direct pre-modeling action blocker).
    """
    dq = report["results"].get("data_quality_scorer", {})
    if not isinstance(dq, dict) or dq.get("status") != "success":
        print("  ℹ  data_quality_scorer unavailable — redundancy check skipped")
        return
    cats = dq.get("data", {}).get("categories", {})
    redundancy = cats.get("redundancy", {})
    level = redundancy.get("level")
    assert level == "red", (
        f"Redundancy dimension should be red (7+ columns with VIF > 10), "
        f"got '{level}'.\n"
        f"  findings: {redundancy.get('findings', [])}"
    )
    affected = redundancy.get("affected_count", 0)
    print(f"  ✓  Quality Redundancy: red ({affected} column(s) flagged)")


def _leakage_dimension_red(report: dict) -> None:
    """
    The Quality scorer's Leakage dimension must be red.

    base and copy_a have r≈0.99 — near-perfect correlation that
    detect_data_leakage correctly identifies as a leakage risk.
    This is expected and correct behavior for this dataset.
    """
    dq = report["results"].get("data_quality_scorer", {})
    if not isinstance(dq, dict) or dq.get("status") != "success":
        return
    cats = dq.get("data", {}).get("categories", {})
    leakage = cats.get("leakage", {})
    level = leakage.get("level")
    assert level == "red", (
        f"Leakage dimension should be red (base/copy_a r≈0.99 triggers "
        f"leakage detection), got '{level}'"
    )
    affected = leakage.get("affected_columns", [])
    print(f"  ✓  Quality Leakage: red (near-perfect copies flagged: {affected})")


def _completeness_from_sparse_c(report: dict) -> None:
    """Completeness dimension should be amber from sparse_c (65% NaN)."""
    dq = report["results"].get("data_quality_scorer", {})
    if not isinstance(dq, dict) or dq.get("status") != "success":
        return
    cats = dq.get("data", {}).get("categories", {})
    completeness = cats.get("completeness", {})
    level = completeness.get("level")
    assert level in {
        "amber",
        "red",
    }, f"Completeness should be amber or red (sparse_c has 65% NaN), got '{level}'"
    print(f"  ✓  Quality Completeness: {level} (sparse_c 65% null flagged)")


def _ml_readiness_gate_not_ready(report: dict) -> None:
    """
    ML Readiness gate must be not_ready.

    base and copy_a (r≈0.99) trigger error-level leakage findings in the
    ML Readiness Leakage dimension, which drives the gate to not_ready.
    The Redundancy quality dimension is separate and does not affect the
    ML gate — detect_collinear_features is not routed to ML Readiness.
    """
    scorer = report["results"].get("ml_readiness_scorer", {})
    if not isinstance(scorer, dict) or scorer.get("status") != "success":
        print("  ℹ  ml_readiness_scorer unavailable — gate check skipped")
        return
    gate = scorer.get("data", {}).get("readiness_gate")
    cats = scorer.get("data", {}).get("categories", {})
    leakage_findings = cats.get("leakage", {}).get("findings", [])
    leakage_errors: list = [f for f in leakage_findings if f.get("level") == "error"]
    assert gate == "not_ready", (
        f"ML gate should be not_ready (error-level leakage findings from "
        f"near-perfect column copies), got '{gate}'"
    )
    print(
        f"  ✓  ML gate: not_ready "
        f"({len(leakage_errors)} error-level leakage finding(s) from "
        f"near-perfect column copies)",
    )
