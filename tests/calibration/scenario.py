"""What a scenario is, and the registry the scenario modules fill."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tests.calibration.scoring import Planted

DEFAULT_SEEDS = 20


@dataclass(frozen=True)
class Scenario:
    """A generator of frames with one known truth, run once per seed."""

    name: str
    build: Callable[[int], Planted]
    seeds: int
    summary: str


SCENARIOS: dict[str, Scenario] = {}


def scenario(
    name: str, *, seeds: int = DEFAULT_SEEDS
) -> Callable[[Callable[[int], Planted]], Callable[[int], Planted]]:
    """Register ``build(seed) -> Planted`` as the scenario ``name``.

    The docstring of ``build`` is the scenario's summary. A frame must come from
    ``numpy.random.default_rng(seed)`` and nothing else random, so that a seed gives
    the same frame on every machine.
    """

    def register(build: Callable[[int], Planted]) -> Callable[[int], Planted]:
        if name in SCENARIOS:
            raise ValueError(f"scenario {name!r} is registered twice")
        SCENARIOS[name] = Scenario(name, build, seeds, (build.__doc__ or "").strip())
        return build

    return register
