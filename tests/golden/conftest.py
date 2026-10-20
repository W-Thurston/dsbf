"""Share one engine child run per golden dataset between the suites that read it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.golden.datasets import GoldenDataset
from tests.golden.runner import GoldenRun, profile_with_engine


@pytest.fixture(scope="session")
def engine_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[GoldenDataset], GoldenRun]:
    """The new engine's run of a dataset: the legacy tasks and the core pack.

    A run takes minutes on the larger datasets, so each is made once per session.
    """
    runs: dict[str, GoldenRun] = {}

    def get(dataset: GoldenDataset) -> GoldenRun:
        if dataset.name not in runs:
            workdir: Path = tmp_path_factory.mktemp(f"engine-{dataset.name}")
            runs[dataset.name] = profile_with_engine(dataset, workdir)
        return runs[dataset.name]

    return get
