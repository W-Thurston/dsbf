"""Compare a core check with the legacy task it replaces, on one golden dataset.

A ``Comparator`` takes what the legacy task stored (through the adapter) and what the
core check returned, as the dicts the golden child process writes, and lists every
``Difference``. Each one must be an ``Accepted`` difference with a reason, or the
comparison fails; and an accepted difference that no longer occurs fails too, so the
registry cannot go stale.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

# The engine's statuses in the words ``engine_run.json`` records for the legacy task.
Result = Mapping[str, Any]


@dataclass(frozen=True, order=True)
class Difference:
    """One way the core check's result differs from the legacy task's."""

    subject: str
    kind: str
    detail: str

    def describe(self) -> str:
        return f"{self.subject or 'dataset'}: {self.kind}: {self.detail}"


@dataclass(frozen=True)
class Accepted:
    """A difference that is intended, and why the core check is right.

    ``check`` is the legacy task whose result differs: two tasks can share a core check,
    and each is judged against its own entries. ``dataset`` is a golden dataset name;
    ``subject`` is the column the difference is about, or empty for the whole check.
    """

    check: str
    dataset: str
    subject: str
    kind: str
    detail: str
    reason: str

    @property
    def difference(self) -> Difference:
        return Difference(self.subject, self.kind, self.detail)


@dataclass(frozen=True)
class AcceptedKind:
    """A kind of difference accepted wherever it occurs, for a check calibrated since.

    The core checks were migrated to match their tasks, and the differences listed one
    by one in the ``accepted_*`` modules are the ones that migration chose. A check
    that was calibrated afterwards against planted truth (``tests/calibration``) differs
    from its task in every row its new method touches, thousands for a test that now
    takes another p-value, so the kinds it changed are accepted for all datasets at
    once, with the reason, and the calibration suite and the golden snapshots guard
    the check from here on.
    """

    check: str
    kinds: tuple[str, ...]
    reason: str

    def covers(self, check: str, difference: Difference) -> bool:
        return check == self.check and difference.kind in self.kinds


# A comparator gets the legacy task's stored entry (status, notes, finding count and
# result, as ``engine_run.json`` has it) and the core check's result as a dict.
Comparator = Callable[[Result, Result], list[Difference]]


def compare(
    check: str,
    comparator: Comparator,
    legacy: Result | None,
    core: Result,
) -> list[Difference]:
    """Every difference between the two, the statuses first.

    A legacy task the engine did not run is itself a difference, since there is
    nothing to compare the core check with.
    """
    if legacy is None:
        return [Difference("", "legacy_not_run", f"{check} has no legacy result")]
    legacy_status = _legacy_word(legacy["status"])
    core_status = core["status"]
    if legacy_status != core_status:
        return [Difference("", "status", f"legacy {legacy_status}, core {core_status}")]
    if core_status != "ok":
        return []
    return sorted(comparator(legacy, core))


def _legacy_word(status: str) -> str:
    """The adapter records the engine's status, which is already the core's word."""
    return status


@dataclass(frozen=True)
class Verdict:
    """How the differences found compare with the ones accepted."""

    unexpected: list[Difference]
    stale: list[Accepted]

    @property
    def clean(self) -> bool:
        return not (self.unexpected or self.stale)


def judge(
    check: str,
    dataset: str,
    found: list[Difference],
    accepted: tuple[Accepted, ...],
    kinds: tuple[AcceptedKind, ...] = (),
) -> Verdict:
    """The differences nobody accepted, and the accepted ones that did not occur.

    A difference of a kind in ``kinds`` is accepted, and a listed one of such a kind
    is not stale for failing to occur: the calibrated check may no longer produce it.
    """
    listed = [a for a in accepted if a.check == check and a.dataset == dataset]
    wanted = {a.difference for a in listed}
    present = set(found)

    def calibrated(difference: Difference) -> bool:
        return any(k.covers(check, difference) for k in kinds)

    return Verdict(
        unexpected=[d for d in found if d not in wanted and not calibrated(d)],
        stale=[
            a
            for a in listed
            if a.difference not in present and not calibrated(a.difference)
        ],
    )
