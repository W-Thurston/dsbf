"""A core check replaces a legacy task without starving the tasks that still read it."""

from pathlib import Path

import pytest

from dsbf.checks import core_pack
from dsbf.engine import Registry, Selection, execute
from dsbf.legacy import (
    SUPERSEDED_BY,
    SUPERSEDED_PACK,
    legacy_checks,
    make_legacy_plan,
    superseded_tasks,
)
from dsbf.model import CheckStatus
from tests.test_legacy.support import make_session

CORE = {"core": ("zeros",), "legacy": ("detect_zeros",)}


@pytest.mark.unit
class TestWhichTasksAreReplaced:
    def test_a_task_is_replaced_when_its_core_check_is_in_a_selected_pack(self):
        assert superseded_tasks(("core", "legacy"), CORE) == {"detect_zeros": "zeros"}

    def test_without_the_core_pack_nothing_is_replaced(self):
        assert superseded_tasks(("legacy",), CORE) == {}

    def test_a_task_named_in_include_is_not_replaced(self):
        assert superseded_tasks(("core",), CORE, include=("detect_zeros",)) == {}

    def test_the_mapping_can_be_given(self):
        mapping = {"detect_out_of_bounds": "zeros"}

        assert superseded_tasks(("core",), CORE, mapping=mapping) == mapping

    def test_the_first_migration_is_the_pilot_check(self):
        assert SUPERSEDED_BY == {"detect_zeros": "zeros"}


def plan_for(tmp_path: Path, superseded, *, include=(), exclude=(), packs=("core",)):
    session = make_session(tmp_path)
    registry = Registry(legacy_checks(session, superseded=superseded))
    registry.register_pack(core_pack())
    selection = Selection(
        packs=packs, include=include, exclude=exclude, entry_points=False
    )
    return make_legacy_plan(
        registry, selection, superseded, profile=session.profile
    ), registry


@pytest.mark.unit
class TestThePlan:
    def test_the_default_selection_runs_the_core_check_and_not_the_task(self, tmp_path):
        plan, _ = plan_for(
            tmp_path, {"detect_zeros": "zeros"}, packs=("core", "legacy")
        )

        assert "zeros" in plan
        assert plan.get("zeros").selected
        assert not plan.get("detect_zeros").selected, "present only as a requirement"
        assert "ml_readiness_scorer" in plan.get("detect_zeros").required_by

    def test_a_replaced_task_nothing_requires_is_left_out(self, tmp_path):
        plan, _ = plan_for(tmp_path, {"detect_zeros": "zeros"})

        assert plan.ids == ("zeros",)

    def test_a_replaced_task_sits_in_its_own_pack(self, tmp_path):
        _, registry = plan_for(tmp_path, {"detect_zeros": "zeros"})

        assert registry.get("detect_zeros").spec.packs == (SUPERSEDED_PACK,)
        assert registry.get("detect_zeros").spec.id not in registry.packs["legacy"]

    def test_a_task_read_without_being_required_is_pulled_in(self, tmp_path):
        replaced = {"detect_out_of_bounds": "zeros"}

        plan, _ = plan_for(tmp_path, replaced, include=("data_quality_scorer",))

        assert "detect_out_of_bounds" in plan
        assert "detect_out_of_bounds" in plan.get("data_quality_scorer").after

    def test_the_read_is_not_pulled_in_when_no_reader_is_planned(self, tmp_path):
        plan, _ = plan_for(tmp_path, {"detect_out_of_bounds": "zeros"})

        assert "detect_out_of_bounds" not in plan

    def test_a_read_the_caller_excluded_stays_out(self, tmp_path):
        plan, _ = plan_for(
            tmp_path,
            {"detect_out_of_bounds": "zeros"},
            include=("data_quality_scorer",),
            exclude=("detect_out_of_bounds",),
        )

        assert "detect_out_of_bounds" not in plan

    def test_the_replacement_is_part_of_the_cache_key(self, tmp_path):
        session = make_session(tmp_path)
        plain = {c.id: c for c in legacy_checks(session)}["detect_zeros"]
        replaced = {
            c.id: c
            for c in legacy_checks(session, superseded={"detect_zeros": "zeros"})
        }["detect_zeros"]

        assert plain.resolve_params().model_dump() != (
            replaced.resolve_params().model_dump()
        )
        assert replaced.resolve_params().model_dump()["superseded_by"] == "zeros"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    outcome = {}
    for label, superseded in (
        ("replaced", {"detect_out_of_bounds": "zeros"}),
        ("plain", {}),
    ):
        tmp_path = tmp_path_factory.mktemp(label)
        plan, _ = plan_for(
            tmp_path,
            superseded,
            include=("data_quality_scorer",),
            packs=("core", "legacy"),
        )
        session = make_session(tmp_path)
        outcome[label] = execute(
            plan, session.dataset, session.profile, dsbf_version="test", workers=1
        )
    return outcome


@pytest.mark.integration
class TestARunWithAReplacedTask:
    def test_the_scorer_still_receives_the_replaced_result(self, runs):
        replaced = runs["replaced"].results
        plain = runs["plain"].results

        assert replaced["detect_out_of_bounds"].status is CheckStatus.OK
        assert replaced["data_quality_scorer"].status is CheckStatus.OK
        assert (
            replaced["data_quality_scorer"].output
            == plain["data_quality_scorer"].output
        )

    def test_the_replaced_task_reports_its_output_and_no_findings(self, runs):
        replaced = runs["replaced"].results["detect_out_of_bounds"]
        plain = runs["plain"].results["detect_out_of_bounds"]

        assert replaced.output == plain.output
        assert replaced.findings == ()
        assert [n.code for n in replaced.notes] == ["findings_superseded"]
        assert "zeros" in replaced.notes[0].message
