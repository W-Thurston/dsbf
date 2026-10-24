"""The synthesis checks: quality dimensions and ML readiness over findings."""

import numpy as np
import polars as pl
import pytest

from dsbf.checks import core_pack
from dsbf.checks.ml_readiness import MlReadinessOutput
from dsbf.checks.quality_dimensions import QualityDimensionsOutput
from dsbf.checks.synthesis import (
    ADVICE_CHECKS,
    QUALITY_DIMENSIONS,
    QUALITY_INPUTS,
    READINESS_INPUTS,
    InputReport,
    InputState,
    coverage,
    input_state,
    inputs_of,
    is_assessed,
    traffic_light,
)
from dsbf.data import load
from dsbf.engine import Registry, Selection, execute, make_plan
from dsbf.model import (
    CheckResult,
    CheckStatus,
    Dimension,
    Note,
    Run,
    Severity,
    Timing,
)
from dsbf.profiling import profile_dataset
from tests.test_model.builders import strict_loads

pytestmark = pytest.mark.unit

CHECKS = {c.id: c for c in core_pack().checks}
SYNTHESIS = ("quality_dimensions", "ml_readiness")
# Past the 1 000 rows `sample_size` wants for a train, validation and test split.
N = 1200
RNG = np.random.default_rng(11)


def run_with(frame: pl.DataFrame, only: set[str] | None = None, **params) -> Run:
    """Run the synthesis checks beside ``only`` (default: every other core check).

    ``params`` is a mapping of check id to parameters, as keyword arguments.
    """
    ids = set(CHECKS) if only is None else only | set(SYNTHESIS)
    dataset = load(frame, name="frame")
    profile = profile_dataset(dataset)
    plan = make_plan(
        Registry(CHECKS[i] for i in sorted(ids)),
        Selection(depth="deep", entry_points=False),
        params=params or None,
        profile=profile,
    )
    return execute(plan, dataset, profile, dsbf_version="test", workers=1)


def quality(run: Run) -> dict:
    result = run.results["quality_dimensions"]
    assert result.status is CheckStatus.OK, result.notes
    assert result.output is not None
    return result.output


def readiness(run: Run) -> dict:
    result = run.results["ml_readiness"]
    assert result.status is CheckStatus.OK, result.notes
    assert result.output is not None
    return result.output


def clean_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": RNG.uniform(size=N),
            "b": RNG.uniform(size=N),
            "group": [f"g{i % 4}" for i in range(N)],
        }
    )


def result_of(status: CheckStatus, code: str | None = None) -> CheckResult:
    return CheckResult(
        check_id="x",
        check_version="1.0.0",
        status=status,
        notes=()
        if status is CheckStatus.OK
        else (Note(code=code or "c", message="m"),),
        timing=Timing(wall_seconds=0.0),
    )


class TestTheInputsMatchTheChecks:
    def test_every_listed_check_is_a_core_check(self):
        listed = set(inputs_of(QUALITY_INPUTS.values())) | set(
            inputs_of(READINESS_INPUTS.values())
        )

        assert listed <= set(CHECKS)
        assert not listed & set(SYNTHESIS)

    def test_a_quality_input_declares_the_dimension_it_is_listed_under(self):
        for dimension, ids in QUALITY_INPUTS.items():
            for check_id in ids:
                assert dimension in CHECKS[check_id].spec.dimensions, (
                    check_id,
                    dimension,
                )

    def test_every_check_that_declares_a_quality_dimension_is_listed_or_advice(self):
        for check_id, found in CHECKS.items():
            if found.spec.kind == "synthesis" or check_id in ADVICE_CHECKS:
                continue
            for dimension in found.spec.dimensions:
                if dimension in QUALITY_DIMENSIONS:
                    assert check_id in QUALITY_INPUTS[dimension], (
                        f"{check_id} declares {dimension.value} but "
                        "quality_dimensions does not read it"
                    )

    def test_advice_checks_are_core_checks_and_in_no_quality_dimension(self):
        assert ADVICE_CHECKS <= set(CHECKS)
        assert not ADVICE_CHECKS & set(inputs_of(QUALITY_INPUTS.values()))

    def test_a_check_is_in_one_readiness_dimension(self):
        listed = [i for ids in READINESS_INPUTS.values() for i in ids]

        assert len(listed) == len(set(listed))

    def test_a_synthesis_check_runs_after_every_input_it_reads(self):
        assert CHECKS["quality_dimensions"].spec.after == inputs_of(
            QUALITY_INPUTS.values()
        )
        assert CHECKS["ml_readiness"].spec.after == inputs_of(READINESS_INPUTS.values())

    def test_the_five_quality_dimensions_are_all_listed(self):
        assert tuple(QUALITY_INPUTS) == QUALITY_DIMENSIONS


class TestTheStateOfAnInput:
    @pytest.mark.parametrize(
        ("result", "state"),
        [
            (None, "not_run"),
            (result_of(CheckStatus.OK), "assessed"),
            (result_of(CheckStatus.NOT_APPLICABLE, "no_target"), "needs_setting"),
            (result_of(CheckStatus.NOT_APPLICABLE, "no_bounds"), "needs_setting"),
            (
                result_of(CheckStatus.NOT_APPLICABLE, "no_eligible_columns"),
                "nothing_to_check",
            ),
            (result_of(CheckStatus.NOT_APPLICABLE, "no_strings"), "nothing_to_check"),
            (result_of(CheckStatus.ERROR), "failed"),
            (result_of(CheckStatus.TIMEOUT), "failed"),
            (result_of(CheckStatus.SKIPPED), "failed"),
            (result_of(CheckStatus.UNAVAILABLE), "failed"),
        ],
    )
    def test_each_result_reads_as_one_state(self, result, state):
        assert input_state(result)[0] == state

    def test_only_a_completed_check_has_no_reason(self):
        assert input_state(result_of(CheckStatus.OK))[1] is None
        assert input_state(None)[1]
        assert input_state(result_of(CheckStatus.ERROR))[1] == "m"

    def reports(self, *states: InputState) -> list[InputReport]:
        return [
            InputReport(check_id=f"c{i}", state=s, findings=0, reason=None)
            for i, s in enumerate(states)
        ]

    def test_coverage_counts_what_was_read_or_had_nothing_to_read(self):
        inputs = self.reports(
            "assessed", "nothing_to_check", "needs_setting", "not_run", "failed"
        )

        assert coverage(inputs) == pytest.approx(0.4)
        assert coverage([]) == 0.0

    def test_a_dimension_is_assessed_when_one_input_was_covered(self):
        assert is_assessed(self.reports("not_run", "nothing_to_check"))
        assert not is_assessed(self.reports("not_run", "needs_setting", "failed"))
        assert not is_assessed([])


class TestTheTrafficLight:
    @pytest.mark.parametrize(
        ("pct", "any_affected", "severity", "level"),
        [
            (0.0, False, None, "green"),
            (0.5, False, Severity.ERROR, "green"),
            (0.02, True, Severity.ERROR, "red"),
            (0.02, True, Severity.WARN, "amber"),
            (0.5, True, Severity.WARN, "amber"),
            (0.51, True, Severity.WARN, "red"),
            (0.0, True, Severity.WARN, "amber"),
            (0.0, True, Severity.INFO, "blue"),
            (0.09, True, Severity.NOTICE, "blue"),
            (0.30, True, Severity.INFO, "blue"),
            (0.31, True, Severity.INFO, "amber"),
            (1.0, True, Severity.INFO, "amber"),
        ],
    )
    def test_the_level_follows_the_severity_and_the_share_of_columns(
        self, pct, any_affected, severity, level
    ):
        assert traffic_light(pct, any_affected, severity) == level


class TestQualityDimensions:
    def test_a_clean_frame_is_green_where_it_was_read(self):
        output = quality(run_with(clean_frame()))

        dims = output["dimensions"]
        assert list(dims) == [d.value for d in QUALITY_DIMENSIONS]
        for name in ("completeness", "validity", "usability", "redundancy"):
            assert dims[name]["level"] == "green", name
            assert dims[name]["affected_columns"] == []
        assert output["total_columns"] == 3
        assert output["clean_columns"] == ["a", "b", "group"]

    def test_leakage_is_not_assessed_without_a_target(self):
        leakage = quality(run_with(clean_frame()))["dimensions"]["leakage"]

        assert leakage["assessed"] is False
        assert leakage["level"] is None
        assert leakage["coverage"] == 0.0
        assert [(i["check_id"], i["state"]) for i in leakage["inputs"]] == [
            ("target_leakage", "needs_setting")
        ]

    def test_a_copy_of_the_target_makes_leakage_red(self):
        frame = clean_frame().with_columns(
            target=pl.col("a") > 0.5, copy=(pl.col("a") > 0.5)
        )

        run = run_with(frame, target_leakage={"target": "target"})

        leakage = quality(run)["dimensions"]["leakage"]
        assert leakage["assessed"] is True
        assert leakage["level"] == "red"
        assert leakage["affected_columns"] == ["copy"]
        assert leakage["findings"][0]["check_id"] == "target_leakage"
        assert leakage["findings"][0]["severity"] == "error"

    def test_missing_values_lower_completeness(self):
        values = [None if i % 2 == 0 else float(i) for i in range(N)]
        frame = clean_frame().with_columns(holes=pl.Series(values, dtype=pl.Float64))

        completeness = quality(run_with(frame))["dimensions"]["completeness"]

        assert completeness["affected_columns"] == ["holes"]
        assert completeness["pct_affected"] == pytest.approx(0.25)
        assert completeness["max_severity"] == "error"
        assert completeness["level"] == "red"
        finding = completeness["findings"][0]
        assert (finding["check_id"], finding["code"]) == ("null_rate", "high_null_rate")

    def test_a_constant_column_is_a_usability_problem(self):
        frame = clean_frame().with_columns(const=pl.lit(1.0))

        usability = quality(run_with(frame))["dimensions"]["usability"]

        assert "const" in usability["affected_columns"]
        assert "constant_columns" in {f["check_id"] for f in usability["findings"]}

    def test_a_column_both_redundancy_checks_flag_counts_once(self):
        a = RNG.normal(size=N)
        frame = pl.DataFrame({"a": a, "same": a, "other": RNG.normal(size=N)})

        redundancy = quality(run_with(frame))["dimensions"]["redundancy"]

        assert sorted(redundancy["affected_columns"]) == ["a", "same"]
        assert redundancy["affected_count"] == 2
        assert len(redundancy["findings"]) >= 2

    def test_a_finding_about_the_dataset_counts_without_a_column(self):
        frame = pl.DataFrame({"a": RNG.normal(size=8)})

        usability = quality(run_with(frame))["dimensions"]["usability"]

        assert usability["affected_columns"] == []
        assert usability["level"] != "green"
        assert {f["check_id"] for f in usability["findings"]} == {"sample_size"}

    def test_the_findings_are_listed_most_severe_first(self):
        values = [None if i % 2 == 0 else float(i) for i in range(N)]
        frame = clean_frame().with_columns(
            holes=pl.Series(values, dtype=pl.Float64),
            some=pl.Series([None if i % 10 == 0 else 1.0 * i for i in range(N)]),
        )

        findings = quality(run_with(frame))["dimensions"]["completeness"]["findings"]

        ranks = [list(Severity).index(Severity(f["severity"])) for f in findings]
        assert ranks == sorted(ranks, reverse=True)

    def test_a_check_missing_from_the_run_lowers_coverage(self):
        run = run_with(clean_frame(), only=set(CHECKS) - {"null_rate", "imputation"})

        completeness = quality(run)["dimensions"]["completeness"]

        assert completeness["assessed"] is False
        assert completeness["level"] is None
        assert completeness["coverage"] == 0.0
        assert completeness["inputs"][0]["state"] == "not_run"
        assert quality(run)["dimensions"]["validity"]["assessed"] is True

    def test_a_partly_missing_dimension_is_still_assessed_with_less_coverage(self):
        run = run_with(clean_frame(), only=set(CHECKS) - {"duplicate_rows"})

        validity = quality(run)["dimensions"]["validity"]

        assert validity["assessed"] is True
        states = {i["check_id"]: i["state"] for i in validity["inputs"]}
        assert states["duplicate_rows"] == "not_run"
        assert states["schema"] == "needs_setting"
        assert validity["coverage"] < 1

    def test_the_inputs_the_caller_can_give_raise_coverage(self):
        frame = clean_frame()
        bare = quality(run_with(frame))["dimensions"]["validity"]["coverage"]

        given = quality(
            run_with(
                frame,
                out_of_bounds={"bounds": {"a": {"min": -100, "max": 100}}},
            )
        )["dimensions"]["validity"]["coverage"]

        assert given > bare

    def test_it_reports_and_raises_no_finding(self):
        run = run_with(
            clean_frame().with_columns(const=pl.lit(1.0)),
        )

        assert run.results["quality_dimensions"].findings == ()
        assert run.results["ml_readiness"].findings == ()

    def test_the_output_validates_against_its_model_and_is_strict_json(self):
        run = run_with(clean_frame())

        QualityDimensionsOutput.model_validate(quality(run))
        strict_loads(run.model_dump_json())

    def test_a_dataset_with_no_checks_planned_is_not_assessed_anywhere(self):
        run = run_with(clean_frame(), only=set())

        dims = quality(run)["dimensions"]

        assert not any(d["assessed"] for d in dims.values())
        assert quality(run)["clean_columns"] == ["a", "b", "group"]


class TestMlReadiness:
    def test_a_clean_frame_is_ready_for_the_dimensions_that_were_assessed(self):
        output = readiness(run_with(clean_frame()))

        assert output["gate"] == "ready"
        assert output["complete"] is False
        assert output["unassessed"] == ["leakage"]
        assert output["dimensions"]["leakage"]["level"] is None
        for name in ("transformations", "encoding", "missingness", "unusable"):
            assert output["dimensions"][name]["level"] == "green", name

    def test_with_a_target_every_dimension_is_assessed(self):
        frame = clean_frame().with_columns(target=pl.col("a") + RNG.normal(size=N))

        output = readiness(run_with(frame, target_leakage={"target": "target"}))

        assert output["complete"] is True
        assert output["unassessed"] == []
        assert output["dimensions"]["leakage"]["level"] == "green"

    def test_a_leaking_column_makes_the_gate_not_ready(self):
        frame = clean_frame().with_columns(
            target=pl.col("a") > 0.5, copy=(pl.col("a") > 0.5)
        )

        output = readiness(run_with(frame, target_leakage={"target": "target"}))

        assert output["gate"] == "not_ready"
        leakage = output["dimensions"]["leakage"]
        assert leakage["level"] == "red"
        assert leakage["action_columns"] == ["copy"]
        assert "copy" in output["action_columns"]

    def test_a_warning_makes_the_gate_needs_work(self):
        skewed = np.exp(RNG.normal(0, 1.5, size=N))
        frame = clean_frame().with_columns(skewed=pl.Series(skewed))

        output = readiness(run_with(frame))

        assert output["dimensions"]["transformations"]["level"] == "amber"
        assert output["gate"] == "needs_work"
        assert "skewed" in output["dimensions"]["transformations"]["action_columns"]

    def test_advice_alone_does_not_move_a_dimension_off_green(self):
        output = readiness(run_with(clean_frame()))

        encoding = output["dimensions"]["encoding"]
        assert encoding["level"] == "green"
        assert encoding["action_columns"] == []
        assert encoding["advisory_columns"] == ["group"]

    def test_a_finding_carries_the_ml_actions_it_was_given(self):
        output = readiness(run_with(clean_frame()))

        findings = output["dimensions"]["encoding"]["findings"]
        assert any(f["actions"] for f in findings)
        action = next(a for f in findings for a in f["actions"])
        assert action["action"] == "encode"

    def test_the_column_buckets_partition_the_columns(self):
        values = [None if i % 2 == 0 else float(i) for i in range(N)]
        frame = clean_frame().with_columns(holes=pl.Series(values, dtype=pl.Float64))

        output = readiness(run_with(frame))

        action = set(output["action_columns"])
        advisory = set(output["advisory_columns"])
        clean = set(output["clean_columns"])
        assert "holes" in action
        assert not (action & advisory or action & clean or advisory & clean)
        assert action | advisory | clean == set(frame.columns)

    def test_a_column_flagged_in_one_dimension_and_advised_in_another_is_action(self):
        values = [None if i % 2 == 0 else float(i) for i in range(N)]
        frame = pl.DataFrame({"holes": pl.Series(values, dtype=pl.Float64)})

        output = readiness(run_with(frame))

        assert output["action_columns"] == ["holes"]
        assert "holes" not in output["advisory_columns"]

    def test_no_planned_input_is_not_assessed(self):
        output = readiness(run_with(clean_frame(), only=set()))

        assert output["gate"] == "not_assessed"
        assert output["complete"] is False
        assert len(output["unassessed"]) == 5

    def test_the_output_validates_against_its_model(self):
        run = run_with(clean_frame())

        MlReadinessOutput.model_validate(readiness(run))

    def test_a_dataset_with_no_column_of_a_kind_is_assessed_green_for_it(self):
        frame = pl.DataFrame({"c": [f"v{i % 3}" for i in range(N)]})

        output = readiness(run_with(frame))

        assert output["dimensions"]["transformations"]["assessed"] is True
        assert output["dimensions"]["transformations"]["level"] == "green"


def test_both_checks_are_in_the_core_pack_and_declared_as_reports():
    for check_id in SYNTHESIS:
        spec = CHECKS[check_id].spec
        assert spec.kind == "synthesis"
        assert spec.applies_to is None
        assert spec.requires.checks == ()
        assert spec.packs == ("core",)
        assert (
            Dimension.USABILITY in spec.dimensions or check_id == "quality_dimensions"
        )
