from __future__ import annotations

from collections.abc import Callable

import pytest

import dsbf
from dsbf.model import Run
from tests.generators.dsbf_test_dataset_generator import GENERATORS


@pytest.fixture(scope="session")
def archetype() -> Callable[[str], Run]:
    """The `Run` of an archetype dataset, profiled once per session.

    The archetypes are the generators of `tests/generators` at their default size and
    seed, profiled with `dsbf.profile()` as a user would, in memory.
    """
    runs: dict[str, Run] = {}

    def profile(name: str) -> Run:
        if name not in runs:
            runs[name] = dsbf.profile(GENERATORS[name](), name=name)
        return runs[name]

    return profile
