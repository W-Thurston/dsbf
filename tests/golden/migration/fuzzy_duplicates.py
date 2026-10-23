"""The fuzzy duplicate check against the task it replaces."""

from __future__ import annotations

from tests.golden.migration.findings import LEGACY_DATASET, columns_read, legacy_levels
from tests.golden.migration.harness import Difference, Result

_RANK = {"info": 0, "warn": 1, "error": 2}


def compare_fuzzy_duplicates(legacy: Result, core: Result) -> list[Difference]:
    """The columns compared, the pairs flagged and the strength of the finding.

    The task scored the whole row string with `difflib` and kept every pair above
    0.85 that shared a token; the check scores each column and averages them, so only
    the counts and the columns are comparable.
    """
    result = legacy["result"]
    output = core["output"] or {}
    found = columns_read(
        (result.get("metadata") or {}).get("string_cols_used", ()),
        output.get("columns", ()),
    )
    was = (result.get("summary") or {}).get("fuzzy_pair_count", 0)
    now = output.get("pairs_flagged", 0)
    if was != now:
        found.append(Difference("", "pairs_flagged", f"legacy {was}, core {now}"))
    before = legacy_levels(result.get("guidance") or {}).get(LEGACY_DATASET)
    after = max(
        (f["severity"] for f in core["findings"]), key=_RANK.__getitem__, default=None
    )
    if before != after:
        found.append(Difference("", "finding", f"legacy {before}, core {after}"))
    return found
