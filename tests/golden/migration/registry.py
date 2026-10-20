"""Which comparator checks each core check, and which differences are accepted."""

from __future__ import annotations

from tests.golden.migration.accepted import ACCEPTED
from tests.golden.migration.harness import Accepted, Comparator
from tests.golden.migration.zeros import compare_zeros

# Core check id to its comparator. The legacy task each one replaces is
# ``dsbf.legacy.SUPERSEDED_BY``'s key for it.
COMPARATORS: dict[str, Comparator] = {
    "zeros": compare_zeros,
}

__all__ = ["ACCEPTED", "COMPARATORS", "Accepted"]
