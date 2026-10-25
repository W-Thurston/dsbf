"""Share one child run per golden dataset between the tests that read it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.golden.datasets import GoldenDataset
from tests.golden.runner import profile


@pytest.fixture(scope="session")
def core_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[GoldenDataset], dict[str, Any]]:
    """The core pack's ``Run`` of a dataset, as a dict.

    A run takes minutes on the larger datasets, so each is made once per session.
    """
    runs: dict[str, dict[str, Any]] = {}

    def get(dataset: GoldenDataset) -> dict[str, Any]:
        if dataset.name not in runs:
            workdir: Path = tmp_path_factory.mktemp(f"core-{dataset.name}")
            runs[dataset.name] = profile(dataset, workdir)
        return runs[dataset.name]

    return get
