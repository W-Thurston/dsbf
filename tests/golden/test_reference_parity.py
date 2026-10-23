"""`reference_drift` against the drift tasks, on each golden dataset and a copy of it.

Run with ``uv run pytest -m golden``. The reference is made by ``reference_one.py``: the
data with some numeric columns moved, a categorical column reweighted, nulls added to a
numeric column and the last column removed. A difference must be listed in
``migration/accepted_reference_drift.py`` under ``<dataset>+reference``, with its
reason; one that no longer occurs fails too. See ``tests/golden/README.md``.
"""

from __future__ import annotations

import pytest

from tests.golden.datasets import GOLDEN_DATASETS, KITCHEN5K, GoldenDataset
from tests.golden.migration.findings import unmapped_titles
from tests.golden.migration.harness import compare, judge
from tests.golden.migration.registry import ACCEPTED, COMPARATORS

pytestmark = pytest.mark.golden

TASKS = (
    "compare_with_reference_dataset",
    "detect_feature_drift",
    "detect_target_drift",
)


def _param(dataset: GoldenDataset):
    marks = [pytest.mark.slow] if dataset is KITCHEN5K else []
    return pytest.param(dataset, id=dataset.name, marks=marks)


@pytest.mark.parametrize("dataset", [_param(d) for d in GOLDEN_DATASETS])
def test_reference_drift_differs_from_its_tasks_only_where_accepted(
    dataset: GoldenDataset, reference_run
) -> None:
    run = reference_run(dataset)
    core = run["core"]["results"]["reference_drift"]
    key = f"{dataset.name}+reference"
    problems: list[str] = []
    for task in TASKS:
        legacy = run["legacy"][task]
        found = compare(task, COMPARATORS[task], legacy, core)
        if legacy["status"] == "ok":
            problems += [
                f"? {task}: the finding titled {title!r} maps to no core finding code"
                for title in unmapped_titles(
                    task, legacy["result"].get("guidance") or {}
                )
            ]
        verdict = judge(task, key, found, ACCEPTED)
        problems += [
            f"+ {task}: {d.describe()}  (add to accepted_reference_drift.py: "
            f"Accepted({task!r}, {key!r}, {d.subject!r}, {d.kind!r}, "
            f"{d.detail!r}, <reason>))"
            for d in verdict.unexpected
        ]
        problems += [
            f"- {task}: {a.difference.describe()}  (accepted, no longer occurs)"
            for a in verdict.stale
        ]
    if problems:
        pytest.fail(
            f"{dataset.name}: reference_drift differs from the drift tasks other than "
            "as accepted.\n" + "\n".join(problems),
            pytrace=False,
        )


@pytest.mark.parametrize("dataset", [_param(d) for d in GOLDEN_DATASETS])
def test_the_perturbed_copy_is_detected(dataset: GoldenDataset, reference_run) -> None:
    """The perturbed copy runs clean: one added column, none dropped, no errors."""
    run = reference_run(dataset)
    core = run["core"]["results"]["reference_drift"]

    assert core["status"] == "ok"
    output = core["output"]
    assert output["columns_dropped"] == []
    if len(output["columns"]) + len(output["columns_added"]) > 1:
        assert len(output["columns_added"]) == 1
    assert run["core"]["health"]["counts"]["error"] == 0
