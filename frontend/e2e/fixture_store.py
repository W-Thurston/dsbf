"""Build the run store the Playwright tests browse (``fixture_store.py <DSBF_HOME>``).

Two runs of titanic: a full scan, and a 300-row sample that also runs a check that
raises and a plugin that fails to import, under a policy that fails. A third dataset,
``cities``, has near-duplicate names so a finding carries a fuzzy-pair artifact.
"""

import sys
import warnings
from pathlib import Path

import polars as pl

import dsbf
from dsbf.store import open_repository

HERE = Path(__file__).parent
TITANIC = HERE.parents[1] / "tests" / "golden" / "data" / "titanic.csv"
POLICY = {
    "version": 1,
    "rules": [
        {
            "name": "no-broken-checks",
            "when": {"health.status": "error"},
            "then": "fail",
        }
    ],
}

# Forty distinct names, each repeated with one misspelt twin, so the pair list is long
# enough to page.
_BASE = [
    f"{stem}{suffix}"
    for stem in ("Spring", "Oak", "River", "Cedar")
    for suffix in (
        "field",
        "ton",
        "ville",
        "burgh",
        "port",
        "wood",
        "dale",
        "mouth",
        "ford",
        "bridge",
    )
]
CITIES = pl.DataFrame({"name": _BASE + [n.replace("e", "ee", 1) for n in _BASE]})


def main(home: Path) -> None:
    warnings.simplefilter("ignore")
    repository = open_repository(home)
    repository.save(dsbf.profile(TITANIC, name="titanic", dataset="titanic"))
    repository.save(
        dsbf.profile(
            TITANIC,
            name="titanic",
            dataset="titanic",
            sample=300,
            checks=["always_fails"],
            plugins=HERE / "plugins",
            policy=POLICY,
        )
    )
    repository.save(
        dsbf.profile(
            CITIES, name="cities", dataset="cities", checks=["fuzzy_duplicates"]
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
