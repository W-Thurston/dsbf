"""The fields a stored run is read by are filled on the golden runs.

Run with ``uv run pytest -m golden``.
"""

from __future__ import annotations

import pytest

from dsbf.checks import core_pack
from dsbf.engine import Registry
from tests.golden.datasets import GOLDEN_DATASETS, KITCHEN5K, GoldenDataset

pytestmark = pytest.mark.golden


def _param(dataset: GoldenDataset):
    marks = [pytest.mark.slow] if dataset is KITCHEN5K else []
    return pytest.param(dataset, id=dataset.name, marks=marks)


@pytest.fixture(scope="module")
def specs():
    registry = Registry()
    registry.register_pack(core_pack())
    return {check.id: check.spec for check in registry.checks}


@pytest.mark.parametrize("dataset", [_param(d) for d in GOLDEN_DATASETS])
def test_every_emitted_code_is_declared(dataset: GoldenDataset, core_run, specs):
    undeclared = {
        (finding["check_id"], finding["code"])
        for result in core_run(dataset)["results"].values()
        for finding in result["findings"]
        if finding["code"] not in specs[finding["check_id"]].codes
    }

    assert not undeclared


@pytest.mark.parametrize("dataset", [_param(d) for d in GOLDEN_DATASETS])
def test_a_check_that_reads_columns_lists_those_it_examined(
    dataset: GoldenDataset, core_run, specs
):
    empty = [
        check_id
        for check_id, result in core_run(dataset)["results"].items()
        if result["status"] == "ok"
        and (
            specs[check_id].applies_to is not None
            or specs[check_id].kind == "synthesis"
        )
        and not result["columns"]
    ]

    assert not empty
