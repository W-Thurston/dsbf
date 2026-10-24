"""The planted-truth scenarios; importing this package registers all of them."""

from tests.calibration.scenario import SCENARIOS, Scenario
from tests.calibration.scenarios import (
    advice,
    contracts,
    distributions,
    drift,
    duplicates,
    leakage,
    missingness,
    noise,
    relationships,
    rules,
    series,
    strings,
    synthesis,
)

__all__ = [
    "SCENARIOS",
    "Scenario",
    "advice",
    "contracts",
    "distributions",
    "drift",
    "duplicates",
    "leakage",
    "missingness",
    "noise",
    "relationships",
    "rules",
    "series",
    "strings",
    "synthesis",
]
