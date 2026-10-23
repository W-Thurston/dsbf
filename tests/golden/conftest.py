"""Share one engine child run per golden dataset between the suites that read it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.golden.datasets import GoldenDataset
from tests.golden.runner import (
    GoldenRun,
    profile_against_reference,
    profile_with_engine,
)


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


@pytest.fixture(scope="session")
def reference_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[GoldenDataset], dict[str, Any]]:
    """The drift tasks and ``reference_drift`` against a perturbed copy of a dataset."""
    runs: dict[str, dict[str, Any]] = {}

    def get(dataset: GoldenDataset) -> dict[str, Any]:
        if dataset.name not in runs:
            workdir: Path = tmp_path_factory.mktemp(f"reference-{dataset.name}")
            runs[dataset.name] = profile_against_reference(dataset, workdir)
        return runs[dataset.name]

    return get
