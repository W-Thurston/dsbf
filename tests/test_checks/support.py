"""Run one core check on a small frame."""

import polars as pl

from dsbf.checks import core_pack
from dsbf.data import load
from dsbf.engine import Registry, Selection, execute, make_plan
from dsbf.model import Action, CheckResult, Finding
from dsbf.profiling import profile_dataset


def run_core_check(frame: pl.DataFrame, check_id: str, **params) -> CheckResult:
    """The result of ``check_id`` on ``frame``, with ``params`` set."""
    dataset = load(frame, name="frame")
    profile = profile_dataset(dataset)
    plan = make_plan(
        Registry(core_pack().checks),
        Selection(include=check_id),
        params={check_id: params} if params else None,
        profile=profile,
    )
    run = execute(plan, dataset, profile, dsbf_version="test", workers=1)
    return run.results[check_id]


def by_column(result: CheckResult) -> dict[str, list]:
    """The findings of a result, by the column each is about."""
    found: dict[str, list] = {}
    for finding in result.findings:
        found.setdefault(finding.subject.columns[0], []).append(finding)
    return found


def output_of(result: CheckResult) -> dict:
    """The output of a result that has one."""
    assert result.output is not None
    return result.output


def actions_of(finding: Finding) -> tuple[Action, ...]:
    """The actions tips attached to a finding."""
    assert finding.guidance is not None
    return finding.guidance.actions
