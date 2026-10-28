"""Build the run store the Playwright tests browse (``fixture_store.py <DSBF_HOME>``).

Two runs of titanic: a full scan, and a 300-row sample that also runs a check that
raises and a plugin that fails to import, under a policy that fails.
"""

import sys
import warnings
from pathlib import Path

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


if __name__ == "__main__":
    main(Path(sys.argv[1]))
