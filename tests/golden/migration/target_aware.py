"""What differs between `mutual_information` and the task it replaces.

The task scored each feature against a configured target and was silent without one.
No golden run gives the legacy tasks a target (their snapshots are made without), so
on the one dataset that has a target for the core checks, the comparison is the core
check's result against an empty one.
"""

from __future__ import annotations

from tests.golden.migration.findings import columns_read, compare_levels
from tests.golden.migration.harness import Difference, Result

# The task's labels for the normalised score (MI over ln n): moderate from 0.05.
_NOTABLE = ("high", "moderate")


def compare_mutual_information(legacy: Result, core: Result) -> list[Difference]:
    old = legacy["result"]["data"]["mi_scores"]
    features = {f["feature"]: f for f in core["output"]["features"]}
    found: list[Difference] = []
    if not old:
        found.append(
            Difference(
                "",
                "target",
                f"legacy had no target, core measured {len(features)} features "
                f"against {core['output']['target']!r}",
            )
        )
    else:
        found += columns_read(old, features)
        for column in sorted(set(old) & set(features)):
            was = old[column]["strength"] in _NOTABLE
            now = features[column]["notable"]
            if was != now:
                found.append(
                    Difference(
                        column,
                        "notable",
                        f"legacy {old[column]['strength']}, core "
                        f"{'notable' if now else 'not notable'}",
                    )
                )
    return found + compare_levels(legacy, core)
