"""Each core check against the legacy task it replaces, on the golden datasets.

Run with ``uv run pytest -m golden``. A difference must be listed in
``migration/accepted.py`` with its reason, and a listed difference that no longer
occurs fails too. See ``tests/golden/README.md``.
"""

from __future__ import annotations

import pytest

from dsbf.legacy import SUPERSEDED_BY
from tests.golden.datasets import GOLDEN_DATASETS, KITCHEN5K, GoldenDataset
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
    for task, check in sorted(SUPERSEDED_BY.items()):
        found = compare(
            task,
            COMPARATORS[task],
            run.report["checks"].get(task),
            run.core["results"][check],
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
