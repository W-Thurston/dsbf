"""The eight archetype datasets, profiled in memory, against what each should show.

Each archetype is a generator in `tests/generators` with a known set of flaws (or
none). These tests state what the core checks must say about it: which findings, which
columns, which quality levels and readiness gate, and that nothing fails. A dataset
that should be all clear is held to no warning at all, so a check that starts crying
wolf on clean data fails here before it reaches a user.

The false-alarm and detection *rates* of each check are measured in
`tests/calibration`; these are single cases a reader can follow.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from dsbf.model import CheckStatus, Run, Severity
from tests.validation.support import (
    FAILED,
    column_types,
    findings,
    flagged,
    output,
    quality,
    readiness,
    statuses,
)

pytestmark = pytest.mark.integration

Archetype = Callable[[str], Run]
DIMENSIONS = ("completeness", "validity", "usability", "redundancy", "leakage")
NUMERIC_ONLY = (
    "bimodality",
    "collinearity",
    "kurtosis",
    "multivariate_outliers",
    "normality",
    "outliers",
    "skewness",
    "zeros",
)

ARCHETYPES = (
    "clean",
    "tiny",
    "near_clean",
    "all_categorical",
    "high_missingness",
    "severe_multicollinearity",
    "single_column",
    "wide",
)


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.slow) if n == "wide" else n for n in ARCHETYPES],
)
def test_no_check_fails_on_any_archetype(archetype: Archetype, name: str) -> None:
    run = archetype(name)
    broken = {
        k: [n.message for n in r.notes]
        for k, r in run.results.items()
        if r.status in FAILED
    }
    assert not broken, broken
    assert {i.status for i in run.health.issues} <= {CheckStatus.NOT_APPLICABLE}
    assert statuses(run, CheckStatus.UNAVAILABLE) == set()


def test_clean_dataset_is_all_clear(archetype: Archetype) -> None:
    run = archetype("clean")
    assert (run.profile.n_rows, len(run.profile.columns)) == (2000, 9)
    assert all(c.counts.nulls == 0 for c in run.profile.columns)
    assert "unknown" not in column_types(run).values()

    loud = {
        k: [
            (f.code, f.subject.columns)
            for f in r.findings
            if f.severity >= Severity.WARN
        ]
        for k, r in run.results.items()
    }
    assert {k: v for k, v in loud.items() if v} == {}
    for dimension in DIMENSIONS[:-1]:
        assert quality(run, dimension)["level"] == "green", dimension
    # Without a target, Leakage is not assessed rather than green.
    assert quality(run, "leakage")["assessed"] is False
    gate = readiness(run)
    assert gate["gate"] == "ready"
    assert gate["complete"] is False and gate["unassessed"] == ["leakage"]


def test_tiny_dataset_warns_about_its_size_and_nothing_breaks(
    archetype: Archetype,
) -> None:
    run = archetype("tiny")
    assert run.profile.n_rows == 25
    warned = findings(run, "sample_size", at_least=Severity.WARN)
    assert warned and warned[0].subject.kind == "dataset"
    assert quality(run, "usability")["level"] in ("amber", "red")


def test_near_clean_dataset_needs_work_but_is_not_blocked(archetype: Archetype) -> None:
    run = archetype("near_clean")
    assert (run.profile.n_rows, len(run.profile.columns)) == (3000, 9)

    # income has about 7% nulls: reported, but under the warning share.
    income = findings(run, "null_rate")
    assert [f.subject.columns for f in income] == [("income",)]
    assert income[0].severity == Severity.INFO
    assert 0.05 < output(run, "null_rate")["columns"]["income"]["null_share"] < 0.10

    # purchase_amount is right-skewed by about 2.
    skew = findings(run, "skewness", "skewed", Severity.WARN)
    assert [f.subject.columns for f in skew] == [("purchase_amount",)]
    advice = findings(run, "transformations", "transform_suggested")
    assert {f.subject.columns for f in advice} == {("purchase_amount",)}

    # plan_type is about 71% one value: worth a line, not a flag.
    dominant = output(run, "dominant_value")["columns"]["plan_type"]
    assert 0.65 < dominant["top_share"] < 0.80 and not dominant["flagged"]

    # One column in nine with a notice-level share of nulls is worth a line (blue),
    # not an action (amber), and nothing is red.
    assert quality(run, "completeness")["level"] == "blue"
    assert quality(run, "completeness")["affected_columns"] == ["income"]
    for dimension in ("validity", "usability", "redundancy"):
        assert quality(run, dimension)["level"] == "green", dimension
    gate = readiness(run)
    assert gate["gate"] == "needs_work"
    assert gate["dimensions"]["transformations"]["level"] == "amber"
    assert "red" not in {d["level"] for d in gate["dimensions"].values()}

    for check_id in ("id_columns", "constant_columns", "duplicate_columns"):
        assert findings(run, check_id) == [], check_id
    assert run.results["target_leakage"].status == CheckStatus.NOT_APPLICABLE


def test_all_categorical_dataset_skips_numeric_checks_cleanly(
    archetype: Archetype,
) -> None:
    run = archetype("all_categorical")
    assert (run.profile.n_rows, len(run.profile.columns)) == (2000, 9)
    assert "numeric" not in column_types(run).values()
    assert all(c.counts.nulls == 0 for c in run.profile.columns)
    for check_id in NUMERIC_ONLY:
        result = run.results[check_id]
        assert result.status == CheckStatus.NOT_APPLICABLE, check_id
        assert result.notes, f"{check_id} gave no reason"

    assert flagged(run, "dominant_value", at_least=Severity.WARN) == {"dominant"}
    assert flagged(run, "high_cardinality") == {"tag"}
    assert quality(run, "usability")["level"] == "amber"
    gate = readiness(run)
    assert gate["gate"] == "needs_work"
    assert gate["dimensions"]["transformations"]["level"] == "green"


def test_high_missingness_dataset_reports_every_pattern(archetype: Archetype) -> None:
    run = archetype("high_missingness")
    shares = {
        k: v["null_share"] for k, v in output(run, "null_rate")["columns"].items()
    }
    expected = {
        "age": (0.05, 0.12),
        "income": (0.20, 0.30),
        "device_type": (0.35, 0.47),
        "premium_score": (0.35, 0.47),
        "notes": (0.70, 0.80),
    }
    for column, (low, high) in expected.items():
        assert low < shares[column] < high, (column, shares[column])
    for column in ("region", "channel", "plan_type", "is_churned"):
        assert shares[column] == 0.0, column

    completeness = quality(run, "completeness")
    assert completeness["level"] == "red"
    assert {"income", "device_type", "premium_score", "notes"} <= set(
        completeness["affected_columns"]
    )
    assert flagged(run, "null_rate", at_least=Severity.ERROR) == {"notes"}

    # The two structured columns follow other columns; the random ones do not.
    depends = flagged(run, "missingness_mechanism", "missingness_depends_on_observed")
    assert {"device_type", "premium_score"} <= depends
    assert not depends & {"age", "income"}

    gate = readiness(run)
    assert gate["gate"] == "not_ready"
    assert gate["dimensions"]["missingness"]["level"] == "red"


def test_severe_multicollinearity_flags_the_collinear_blocks_only(
    archetype: Archetype,
) -> None:
    run = archetype("severe_multicollinearity")
    assert (run.profile.n_rows, len(run.profile.columns)) == (2000, 11)
    assert set(column_types(run).values()) == {"numeric"}

    collinear = flagged(run, "collinearity")
    assert {"base", "copy_a", "copy_b", "combo"} <= collinear  # block A
    assert {"factor_x", "factor_y", "factor_z"} <= collinear  # block C
    # Block B is independent at wildly different scales: it must not read as collinear.
    assert not collinear & {"revenue", "click_rate"}
    redundant = findings(run, "associations", "redundant_columns")
    assert any(set(f.subject.columns) == {"base", "copy_a"} for f in redundant)

    assert quality(run, "redundancy")["level"] == "red"
    # sparse_c is 65% null.
    assert quality(run, "completeness")["level"] == "red"
    assert readiness(run)["gate"] == "not_ready"


def test_single_column_dataset_degrades_cleanly(archetype: Archetype) -> None:
    run = archetype("single_column")
    assert (run.profile.n_rows, len(run.profile.columns)) == (500, 1)
    assert column_types(run) == {"value": "numeric"}
    for check_id in ("associations", "collinearity", "contingency"):
        assert run.results[check_id].status == CheckStatus.NOT_APPLICABLE, check_id
    assert flagged(run, "skewness", "skewed", Severity.WARN) == {"value"}
    assert flagged(run, "outliers") == {"value"}
    assert quality(run, "completeness")["level"] == "green"


@pytest.mark.slow
def test_wide_dataset_is_reported_in_full(archetype: Archetype) -> None:
    run = archetype("wide")
    assert (run.profile.n_rows, len(run.profile.columns)) == (1000, 100)
    assert len(output(run, "null_rate")["columns"]) == 100

    skewed = {f"cont_skew_{i:02d}" for i in range(20)}
    assert skewed <= flagged(run, "skewness", "skewed")
    holes = {f"null_{i:02d}" for i in range(8)}
    assert holes <= flagged(run, "null_rate")
    assert holes <= set(quality(run, "completeness")["affected_columns"])
    assert flagged(run, "high_cardinality") == {"cat_hc"}
    assert quality(run, "usability")["level"] == "amber"
    assert quality(run, "redundancy")["level"] == "green"
    assert findings(run, "collinearity", at_least=Severity.WARN) == []
