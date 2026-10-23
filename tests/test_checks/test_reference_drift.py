"""`reference_drift` on small frames, and the arithmetic under it."""

import math
from typing import Any

import numpy as np
import polars as pl
import pytest

from dsbf.checks import core_pack
from dsbf.checks.reference_drift import (
    bin_counts,
    coarsen,
    jensen_shannon,
    population_stability,
    total_variation,
)
from dsbf.data import load
from dsbf.engine import Reference, Registry, Selection, execute, make_plan
from dsbf.model import CheckResult, CheckStatus, Finding, Run, Severity
from dsbf.profiling import profile_dataset
from tests.test_checks.support import output_of

pytestmark = pytest.mark.unit

CHECK = "reference_drift"


def run_drift(
    frame: pl.DataFrame,
    reference: pl.DataFrame | Reference | None,
    *,
    sample_rows: int | None = None,
    **params,
) -> CheckResult:
    """The result of `reference_drift` on ``frame`` against ``reference``."""
    dataset = load(frame, name="current")
    if sample_rows is not None:
        dataset = dataset.sample(sample_rows, strategy="random", seed=1)
    profile = profile_dataset(dataset)
    if isinstance(reference, pl.DataFrame):
        loaded = load(reference, name="reference")
        reference = Reference.of_data(loaded, profile_dataset(loaded))
    plan = make_plan(
        Registry(core_pack().checks),
        Selection(include=CHECK),
        params={CHECK: params} if params else None,
        profile=profile,
        has_reference=reference is not None,
    )
    run = execute(
        plan, dataset, profile, dsbf_version="test", workers=1, reference=reference
    )
    return run.results[CHECK]


def normal(n: int = 2000, loc: float = 0.0, seed: int = 0) -> pl.Series:
    return pl.Series(np.random.default_rng(seed).normal(loc, 1.0, n))


def metric(finding: Finding, name: str) -> Any:
    """One metric of a finding's evidence, untyped for comparison."""
    return finding.evidence.metrics[name]


def codes(result: CheckResult) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for finding in result.findings:
        key = finding.subject.columns[0] if finding.subject.columns else ""
        found.setdefault(key, []).append(finding.code)
    return found


class TestArithmetic:
    def test_bins_merge_until_each_holds_the_minimum(self):
        edges, counts = coarsen([0, 1, 2, 3, 4, 5], [50, 1, 1, 40, 1], min_count=10)

        assert counts == [50, 42, 1] or counts == [50, 43]
        assert edges[0] == 0 and edges[-1] == 5
        assert sum(counts) == 93

    def test_a_thin_tail_joins_the_last_bin(self):
        edges, counts = coarsen([0, 1, 2, 3], [30, 30, 2], min_count=10)

        assert (edges, counts) == ([0, 1, 3], [30, 32])

    def test_the_bins_of_an_empty_reference_stay_one_bin(self):
        assert coarsen([0, 1, 2], [3, 4], min_count=100) == ([0, 2], [7])

    def test_values_fall_below_in_a_bin_or_above(self):
        counts = bin_counts([0.0, 1.0, 2.0], np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0]))

        # below, [0, 1), [1, 2] (closed at the top), above
        assert counts.tolist() == [1, 2, 2, 1]

    def test_a_point_mass_has_a_bin_for_the_value_itself(self):
        counts = bin_counts([5.0, 5.0], np.array([4.0, 5.0, 5.0, 6.0]))

        assert counts.tolist() == [1, 2, 1]

    def test_equal_distributions_have_no_distance(self):
        counts = np.array([0, 40, 60, 0])

        assert population_stability(counts, counts) == 0
        assert jensen_shannon(counts, counts) == 0
        assert total_variation([4, 6], [40, 60]) == 0

    def test_distributions_that_share_nothing_are_far_apart(self):
        a, b = np.array([100, 0]), np.array([0, 100])

        assert jensen_shannon(a, b) == pytest.approx(1.0)
        assert total_variation(a, b) == pytest.approx(1.0)
        assert math.isfinite(population_stability(a, b))
        assert population_stability(a, b) > 1


class TestNumericDrift:
    def test_identical_data_has_no_drift(self):
        frame = pl.DataFrame({"x": normal()})

        result = run_drift(frame, frame)

        assert result.status is CheckStatus.OK
        assert result.findings == ()
        column = output_of(result)["columns"]["x"]
        assert column["drift"] == "none"
        assert column["psi"] == pytest.approx(0, abs=1e-9)
        assert output_of(result)["drifted"] == []

    def test_a_shift_is_found_and_grows_with_the_shift(self):
        reference = pl.DataFrame({"x": normal()})
        psis = []
        for shift in (0.2, 0.4, 1.0):
            result = run_drift(
                pl.DataFrame({"x": normal(loc=shift, seed=1)}), reference
            )
            psis.append(output_of(result)["columns"]["x"]["psi"])

        assert psis == sorted(psis)
        assert psis[0] < 0.1 <= psis[1] < 0.25 < psis[2]

    def test_moderate_drift_is_a_notice_and_high_drift_a_warning(self):
        reference = pl.DataFrame({"x": normal()})

        moderate = run_drift(pl.DataFrame({"x": normal(loc=0.4, seed=1)}), reference)
        high = run_drift(pl.DataFrame({"x": normal(loc=1.0, seed=1)}), reference)

        (m,), (h,) = moderate.findings, high.findings
        assert (m.code, m.severity) == ("distribution_shift", Severity.NOTICE)
        assert (h.code, h.severity) == ("distribution_shift", Severity.WARN)
        assert h.evidence.method == "psi"
        assert metric(h, "reference") == "reference"
        assert metric(h, "psi") > metric(h, "psi_noise")
        assert h.evidence.n_used == 2000

    def test_thresholds_are_parameters(self):
        reference = pl.DataFrame({"x": normal()})
        frame = pl.DataFrame({"x": normal(loc=0.5, seed=1)})

        result = run_drift(frame, reference, psi_warn=0.05, psi_high=0.1)

        assert result.findings[0].severity is Severity.WARN

    def test_values_beyond_the_reference_range_are_counted(self):
        reference = pl.DataFrame({"x": normal()})
        frame = pl.DataFrame({"x": normal(loc=8.0, seed=1)})

        (finding,) = run_drift(frame, reference).findings

        assert metric(finding, "share_outside_reference_range") > 0.99
        assert math.isfinite(metric(finding, "psi"))

    def test_a_constant_reference_is_a_point_mass(self):
        reference = pl.DataFrame({"x": [3.0] * 500})

        same = run_drift(pl.DataFrame({"x": [3.0] * 500}), reference)
        moved = run_drift(pl.DataFrame({"x": normal(500)}), reference)

        assert same.findings == ()
        assert [f.code for f in moved.findings] == ["distribution_shift"]

    def test_a_current_column_the_profile_types_categorical_is_binned_on_its_values(
        self,
    ):
        # Two values are categorical to the profile, numeric to a reference of many.
        reference = pl.DataFrame({"x": normal()})
        frame = pl.DataFrame({"x": [0.0, 1.0] * 1000})

        result = run_drift(frame, reference)

        column = output_of(result)["columns"]["x"]
        assert column["compared"] == "numeric"
        assert column["drift"] != "none"

    def test_non_finite_values_are_left_out(self):
        values = [*normal(500).to_list(), float("inf"), float("-inf")]
        frame = pl.DataFrame({"x": values})

        result = run_drift(frame, frame)

        assert result.findings == ()
        assert output_of(result)["columns"]["x"]["psi"] == pytest.approx(0, abs=1e-9)

    def test_a_column_of_nulls_has_no_distribution_to_compare(self):
        reference = pl.DataFrame({"x": normal(100)})
        frame = pl.DataFrame({"x": pl.Series([None] * 100, dtype=pl.Float64)})

        result = run_drift(frame, reference)

        column = output_of(result)["columns"]["x"]
        assert column["compared"] == "none"
        assert "no finite values" in column["reason"]
        assert codes(result) == {"x": ["missingness_shift"]}

    def test_a_small_sample_has_less_confidence(self):
        reference = pl.DataFrame({"x": normal(2000)})

        (small,) = run_drift(
            pl.DataFrame({"x": normal(20, 2.0, 1)}), reference
        ).findings
        (large,) = run_drift(
            pl.DataFrame({"x": normal(2000, 2.0, 1)}), reference
        ).findings

        assert small.confidence < large.confidence


class TestCategoricalDrift:
    LEVELS = ["a", "b", "c"]

    def column(
        self, weights: list[float], n: int = 3000, seed: int = 0
    ) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        return pl.DataFrame({"k": rng.choice(self.LEVELS, n, p=weights).tolist()})

    def test_identical_shares_have_no_drift(self):
        reference = self.column([0.5, 0.3, 0.2])

        result = run_drift(self.column([0.5, 0.3, 0.2], seed=1), reference)

        assert result.findings == ()
        assert output_of(result)["columns"]["k"]["tvd"] < 0.05

    def test_reweighted_levels_are_found_with_the_largest_shifts(self):
        reference = self.column([0.5, 0.3, 0.2])

        (finding,) = run_drift(self.column([0.2, 0.3, 0.5], seed=1), reference).findings

        assert finding.code == "distribution_shift"
        assert finding.evidence.method == "tvd"
        assert metric(finding, "tvd") == pytest.approx(0.3, abs=0.04)
        shifts = metric(finding, "largest_shifts")
        assert shifts["a"] < 0 < shifts["c"]
        assert metric(finding, "listed_values_complete") is True

    def test_a_new_level_is_other_mass_the_reference_never_had(self):
        reference = self.column([0.5, 0.3, 0.2])
        current = pl.DataFrame({"k": ["a"] * 1000 + ["z"] * 1000})

        (finding,) = run_drift(current, reference).findings

        assert metric(finding, "new_value_share") == pytest.approx(0.5)
        assert finding.severity is Severity.WARN

    def test_a_reference_with_many_levels_pools_the_tail(self):
        labels = [f"v{i}" for i in range(40)]
        reference = pl.DataFrame({"k": labels * 50})

        result = run_drift(pl.DataFrame({"k": labels * 50}), reference)

        assert result.findings == ()
        assert output_of(result)["columns"]["k"]["tvd"] == 0

    def test_an_incomplete_list_says_so_in_the_evidence(self):
        reference = pl.DataFrame({"k": [f"v{i}" for i in range(40)] * 50})
        current = pl.DataFrame({"k": ["v0"] * 2000})

        (finding,) = run_drift(current, reference).findings

        assert metric(finding, "listed_values_complete") is False
        assert metric(finding, "new_value_share") is None

    def test_booleans_compare_their_true_and_false_counts(self):
        reference = pl.DataFrame({"flag": [True] * 900 + [False] * 100})

        same = run_drift(reference, reference)
        flipped = run_drift(
            pl.DataFrame({"flag": [True] * 100 + [False] * 900}), reference
        )

        assert same.findings == ()
        (finding,) = flipped.findings
        assert metric(finding, "tvd") == pytest.approx(0.8)


class TestSchemaAndMissingness:
    def test_added_and_dropped_columns_are_dataset_findings(self):
        reference = pl.DataFrame({"x": normal(200), "old": normal(200, seed=2)})
        frame = pl.DataFrame({"x": normal(200), "new": normal(200, seed=3)})

        result = run_drift(frame, reference)

        by_code = {f.code: f for f in result.findings}
        assert set(by_code) == {"columns_added", "columns_dropped"}
        assert metric(by_code["columns_added"], "columns") == ("new",)
        assert metric(by_code["columns_dropped"], "columns") == ("old",)
        assert by_code["columns_added"].subject.kind == "dataset"
        output = output_of(result)
        assert (output["columns_added"], output["columns_dropped"]) == (
            ["new"],
            ["old"],
        )

    def test_a_changed_type_is_reported_and_the_distribution_is_not_compared(self):
        reference = pl.DataFrame({"x": normal(300)})
        frame = pl.DataFrame({"x": [f"v{i % 4}" for i in range(300)]})

        result = run_drift(frame, reference)

        assert codes(result) == {"x": ["type_changed"]}
        column = output_of(result)["columns"]["x"]
        assert (column["compared"], column["reason"]) == ("none", "the type changed")

    def test_a_shift_in_nulls_is_reported(self):
        reference = pl.DataFrame({"x": normal(1000)})
        values = normal(1000, seed=1).to_list()
        values[:400] = [None] * 400
        frame = pl.DataFrame({"x": pl.Series(values, dtype=pl.Float64)})

        result = run_drift(frame, reference)

        finding = next(f for f in result.findings if f.code == "missingness_shift")
        assert finding.severity is Severity.WARN  # 0.4 is over twice the 0.1 threshold
        assert metric(finding, "null_share") == pytest.approx(0.4)
        assert metric(finding, "reference_null_share") == 0

    def test_a_small_change_in_nulls_is_not(self):
        reference = pl.DataFrame({"x": normal(1000).to_list()[:500] + [None] * 500})
        frame = pl.DataFrame({"x": normal(1000, seed=1).to_list()[:550] + [None] * 450})

        assert "missingness_shift" not in codes(run_drift(frame, reference)).get(
            "x", []
        )

    def test_columns_with_nothing_in_common_skip_the_check(self):
        result = run_drift(pl.DataFrame({"a": [1.0, 2.0]}), pl.DataFrame({"b": [1.0]}))

        assert result.status is CheckStatus.SKIPPED
        assert result.notes[0].code == "no_shared_columns"

    def test_datetime_and_text_distributions_are_not_compared(self):
        words = [f"word number {i}" for i in range(200)]
        frame = pl.DataFrame({"text": words})

        column = output_of(run_drift(frame, frame))["columns"]["text"]

        assert column["compared"] == "none"


class TestTarget:
    def frames(self, current_weights: list[float]):
        rng = np.random.default_rng(0)

        def make(weights: list[float], seed: int) -> pl.DataFrame:
            r = np.random.default_rng(seed)
            return pl.DataFrame(
                {
                    "y": r.choice(["no", "yes"], 3000, p=weights).tolist(),
                    "x": normal(3000),
                }
            )

        del rng
        return make(current_weights, 1), make([0.9, 0.1], 0)

    def test_drift_in_the_target_is_reported_one_level_higher(self):
        frame, reference = self.frames([0.6, 0.4])

        plain = run_drift(frame, reference)
        targeted = run_drift(frame, reference, target="y")

        assert plain.findings[0].code == "distribution_shift"
        (finding,) = targeted.findings
        assert finding.code == "target_drift"
        assert finding.severity > plain.findings[0].severity

    def test_a_target_the_reference_lacks_is_a_finding(self):
        frame = pl.DataFrame({"x": normal(300), "y": ["a", "b", "a"] * 100})
        reference = pl.DataFrame({"x": normal(300)})

        result = run_drift(frame, reference, target="y")

        assert "target_missing_in_reference" in codes(result)["y"]

    def test_a_target_that_is_not_a_column_is_noted(self):
        frame = pl.DataFrame({"x": normal(300)})

        result = run_drift(frame, frame, target="nope")

        assert result.status is CheckStatus.OK
        assert [n.code for n in result.notes] == ["target_not_found"]


class TestReference:
    def test_without_a_reference_the_check_is_not_applicable(self):
        frame = pl.DataFrame({"x": normal(100)})

        result = run_drift(frame, None)

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert result.notes[0].code == "no_reference"

    def test_a_run_makes_the_same_comparison_as_the_data_it_came_from(self):
        reference = pl.DataFrame({"x": normal(), "k": ["a", "b"] * 1000})
        frame = pl.DataFrame(
            {"x": normal(loc=1.0, seed=1), "k": ["a", "a", "b"] * 666 + ["a", "b"]}
        )
        dataset = load(reference, name="reference")
        stored = execute(
            make_plan(Registry([]), profile=profile_dataset(dataset)),
            dataset,
            profile_dataset(dataset),
            dsbf_version="test",
        )
        assert isinstance(stored, Run)

        from_data = run_drift(frame, reference)
        from_run = run_drift(frame, Reference.of_run(stored))

        assert output_of(from_run)["reference_kind"] == "run"
        assert output_of(from_data)["reference_kind"] == "data"
        assert [
            metric(f, "psi") for f in from_run.findings if "psi" in f.evidence.metrics
        ] == [
            metric(f, "psi") for f in from_data.findings if "psi" in f.evidence.metrics
        ]

    def test_samples_on_either_side_are_noted(self):
        reference = pl.DataFrame({"x": normal(1000)})

        result = run_drift(
            pl.DataFrame({"x": normal(1000)}), reference, sample_rows=400
        )

        (note,) = [n for n in result.notes if n.code == "sampled_comparison"]
        assert "400 of 1,000 rows" in note.message
        assert "all 1,000 rows" in note.message
        assert all(f.evidence.sampled is not None for f in result.findings)

    def test_unsampled_data_makes_no_note(self):
        frame = pl.DataFrame({"x": normal(100)})

        assert run_drift(frame, frame).notes == ()

    def test_an_unknown_parameter_is_a_planning_error(self):
        from dsbf.engine import PlanError

        with pytest.raises(PlanError):
            run_drift(pl.DataFrame({"x": [1.0]}), pl.DataFrame({"x": [1.0]}), psi="0.1")

    def test_the_thresholds_must_be_ordered(self):
        from dsbf.engine import PlanError

        with pytest.raises(PlanError):
            run_drift(
                pl.DataFrame({"x": [1.0]}),
                pl.DataFrame({"x": [1.0]}),
                psi_warn=0.3,
                psi_high=0.2,
            )


class TestAdvice:
    def test_every_finding_code_carries_a_tip(self):
        n = 1000
        reference = pl.DataFrame(
            {
                "num": normal(n),
                "cat": ["a", "b"] * (n // 2),
                "kind": normal(n, seed=4),
                "gone": normal(n, seed=5),
                "y": ["no", "yes"] * (n // 2),
                "holes": normal(n, seed=6),
            }
        )
        holes = normal(n, seed=7).to_list()
        holes[:500] = [None] * 500
        frame = pl.DataFrame(
            {
                "num": normal(n, loc=1.0, seed=1),
                "cat": ["a"] * 900 + ["b"] * 100,
                "kind": [f"v{i % 5}" for i in range(n)],
                "new": normal(n, seed=8),
                "y": ["no"] * 900 + ["yes"] * 100,
                "holes": pl.Series(holes, dtype=pl.Float64),
            }
        )

        result = run_drift(frame, reference, target="y")

        found = {(f.subject.columns, f.code): f for f in result.findings}
        assert {code for _, code in found} == {
            "distribution_shift",
            "target_drift",
            "type_changed",
            "missingness_shift",
            "columns_added",
            "columns_dropped",
        }
        for key, finding in found.items():
            assert finding.guidance is not None, key
            assert finding.guidance.actions, key
