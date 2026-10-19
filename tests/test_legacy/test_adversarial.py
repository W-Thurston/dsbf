"""The adapted tasks survive the hostile frame and still yield a complete run."""

import warnings

import polars as pl
import pytest

from dsbf.config import load_default_config
from dsbf.data import load
from dsbf.engine import Registry, Selection, execute, make_plan
from dsbf.legacy import LegacySession, legacy_checks
from dsbf.model import CheckStatus
from dsbf.profile import profile_dataset
from tests.fixtures.adversarial import build_adversarial_frame
from tests.test_model.builders import strict_loads

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def run_and_plan(tmp_path_factory):
    warnings.simplefilter("ignore")
    dataset = load(pl.from_pandas(build_adversarial_frame()), name="adversarial")
    profile = profile_dataset(dataset)
    session = LegacySession(
        dataset,
        profile,
        config=load_default_config(),
        output_dir=tmp_path_factory.mktemp("figures"),
    )
    plan = make_plan(
        Registry(legacy_checks(session)), Selection(depth="deep"), profile=profile
    )
    return execute(plan, dataset, profile, dsbf_version="test", workers=4), plan


def test_every_planned_check_has_a_result(run_and_plan):
    run, plan = run_and_plan

    assert set(run.results) == {entry.id for entry in plan.entries}
    assert len(run.results) > 60


def test_no_check_raised(run_and_plan):
    run, _ = run_and_plan

    raised = {
        name: [n.message for n in r.notes if n.code == "check_failed"]
        for name, r in run.results.items()
        if any(n.code == "check_failed" for n in r.notes)
    }
    assert raised == {}


def test_no_task_failed_or_timed_out(run_and_plan):
    run, _ = run_and_plan

    bad = {
        name: (r.status.value, [n.message for n in r.notes])
        for name, r in run.results.items()
        if r.status in (CheckStatus.ERROR, CheckStatus.TIMEOUT)
    }
    assert bad == {}


def test_the_run_is_strict_json_and_has_findings(run_and_plan):
    run, _ = run_and_plan

    strict_loads(run.model_dump_json())
    assert sum(len(r.findings) for r in run.results.values()) > 0
