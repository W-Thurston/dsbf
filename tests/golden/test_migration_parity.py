"""Each core check against the legacy task it replaces, on the golden datasets.

Run with ``uv run pytest -m golden``. A difference must be listed in
``migration/accepted.py`` with its reason, and a listed difference that no longer
occurs fails too. See ``tests/golden/README.md``.
"""

from __future__ import annotations

import pytest

from dsbf.legacy import RETIRED_TASKS, SUPERSEDED_BY
from tests.golden.datasets import GOLDEN_DATASETS, KITCHEN5K, GoldenDataset
from tests.golden.migration.distribution import merge_core
from tests.golden.migration.findings import unmapped_titles
from tests.golden.migration.harness import compare, judge
from tests.golden.migration.registry import ACCEPTED, COMPARATORS

pytestmark = pytest.mark.golden


def _param(dataset: GoldenDataset):
    marks = [pytest.mark.slow] if dataset is KITCHEN5K else []
    return pytest.param(dataset, id=dataset.name, marks=marks)


@pytest.mark.parametrize("dataset", [_param(d) for d in GOLDEN_DATASETS])
def test_a_core_check_differs_from_its_task_only_where_accepted(
    dataset: GoldenDataset, engine_run
) -> None:
    run = engine_run(dataset)
    assert run.core is not None
    problems: list[str] = []
    for task, checks in sorted(SUPERSEDED_BY.items()):
        found = compare(
            task,
            COMPARATORS[task],
            run.report["checks"].get(task),
            merge_core(run.core["results"], checks),
        )
        legacy = run.report["checks"].get(task)
        if legacy and legacy["status"] == "ok":
            problems += [
                f"? {task}: the finding titled {title!r} maps to no core finding code"
                for title in unmapped_titles(
                    task, legacy["result"].get("guidance") or {}
                )
            ]
        verdict = judge(task, dataset.name, found, ACCEPTED)
        problems += [
            f"+ {task}: {d.describe()}  (add to accepted.py: "
            f"Accepted({task!r}, {dataset.name!r}, {d.subject!r}, {d.kind!r}, "
            f"{d.detail!r}, <reason>))"
            for d in verdict.unexpected
        ]
        problems += [
            f"- {task}: {a.difference.describe()}  (accepted, no longer occurs)"
            for a in verdict.stale
        ]
    if problems:
        pytest.fail(
            f"{dataset.name}: the core checks differ from their legacy tasks "
            "other than as accepted.\n" + "\n".join(problems),
            pytrace=False,
        )


@pytest.mark.parametrize("dataset", [_param(d) for d in GOLDEN_DATASETS])
def test_a_retired_task_raises_no_finding_for_a_core_check_to_carry(
    dataset: GoldenDataset, engine_run
) -> None:
    """The tasks that restate the profile are dropped, so each said nothing."""
    run = engine_run(dataset)
    spoken = {
        task: unmapped_titles(task, entry["result"].get("guidance") or {})
        for task in sorted(RETIRED_TASKS)
        if (entry := run.report["checks"].get(task))
        and entry["status"] == "ok"
        and entry["result"].get("guidance")
    }

    assert spoken == {}, f"{dataset.name}: a retired task raised findings: {spoken}"


@pytest.mark.parametrize("dataset", [_param(d) for d in GOLDEN_DATASETS])
def test_the_default_bounds_of_the_legacy_task_flagged_nothing(
    dataset: GoldenDataset, engine_run
) -> None:
    """`out_of_bounds` takes no bounds by default, so none of this is lost.

    The task checked columns named ``age``, ``temperature``, ``percent`` and ``score``
    against fixed ranges; if it had flagged one on a golden dataset, dropping the
    defaults would have dropped a finding.
    """
    entry = engine_run(dataset).report["checks"]["detect_out_of_bounds"]

    result = entry["result"] or {}  # none when the task had no column to read
    assert entry["status"] in {"ok", "not_applicable"}
    assert not result.get("data")
    assert not result.get("guidance")
