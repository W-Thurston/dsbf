"""Build the run store the Playwright tests browse (``fixture_store.py <DSBF_HOME>``).

Two runs of titanic: a full scan, and a 300-row sample that also runs a check that
raises and a plugin that fails to import, under a policy that fails. A third dataset,
``cities``, has near-duplicate names so a finding carries a fuzzy-pair artifact. A
fourth, ``bare``, is titanic (named ``bare``) stored without a data snapshot. The
``clean`` and ``high_missingness`` archetypes follow, for the quality tab, and last
``kitchen5k`` with its time index, for the time series tab. A last dataset,
``relocated``, is titanic profiled from two paths, for the lineage. ``settings`` is
titanic (named ``settings``) that the settings tests save and rerun. ``relocated``
and ``kitchen5k`` read from a temporary directory that is gone once the store is built,
so a rerun of either finds no source.
"""

import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import polars as pl

import dsbf
from dsbf.store import open_repository

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))  # the archetype generators live in tests/

from tests.generators.dsbf_test_dataset_generator import GENERATORS  # noqa: E402
from tests.golden.datasets import KITCHEN5K  # noqa: E402

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
    repository.save(
        dsbf.profile(TITANIC, name="bare", dataset="bare", depth="quick"),
        snapshot_rows=0,
    )
    repository.save(dsbf.profile(TITANIC, name="settings", dataset="settings"))
    for archetype in ("clean", "high_missingness"):
        repository.save(
            dsbf.profile(GENERATORS[archetype](), name=archetype, dataset=archetype)
        )
    with tempfile.TemporaryDirectory() as directory:
        moved = Path(shutil.copy(TITANIC, Path(directory) / "titanic_moved.csv"))
        for source in (TITANIC, moved):
            repository.save(
                dsbf.profile(
                    source, name="relocated", dataset="relocated", depth="quick"
                )
            )
        kitchen = KITCHEN5K.write_csv(Path(directory))
        repository.save(
            dsbf.profile(
                kitchen,
                name="kitchen5k",
                dataset="kitchen5k",
                time_index=KITCHEN5K.time_series_column,
            )
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
