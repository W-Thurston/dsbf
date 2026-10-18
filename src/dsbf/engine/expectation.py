# dsbf/engine/expectation.py
"""What a plan expects of each check, judged before anything runs."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from dsbf.engine.check import Check
from dsbf.engine.eligibility import eligibility
from dsbf.engine.extras import install_hint, missing_extras
from dsbf.model import DatasetProfile


class Expected(StrEnum):
    """How a check is expected to end, with the reason in its ``Expectation``."""

    WILL_RUN = "will_run"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Expectation:
    """The expected status of one planned check, and why.

    ``columns`` are the columns the check would read, or None when the plan was made
    without a profile (or the check reads none).
    """

    status: Expected
    reason: str | None = None
    hint: str | None = None
    columns: tuple[str, ...] | None = None
    blocked_by: tuple[str, ...] = ()


def expect(
    check: Check,
    profile: DatasetProfile | None,
    upstream: Mapping[str, Expectation],
) -> Expectation:
    """The status ``check`` is expected to end with.

    A missing extra comes first, then a hard requirement that cannot run (``upstream``
    holds the expectations of the checks already judged), then a profile with
    nothing for the check to read. Without a ``profile`` only the first two are
    judged.
    """
    spec = check.spec
    missing = missing_extras(spec.extras)
    if missing:
        modules = ", ".join(m for modules in missing.values() for m in modules)
        names = ", ".join(repr(extra) for extra in missing)
        return Expectation(
            Expected.UNAVAILABLE,
            reason=f"Needs {modules}, which is not installed ({names} extra).",
            hint=install_hint(missing),
        )
    blockers = {
        needed: upstream[needed].status
        for needed in spec.requires.checks
        if needed in upstream
        and upstream[needed].status in (Expected.UNAVAILABLE, Expected.SKIPPED)
    }
    if blockers:
        listed = ", ".join(f"{name!r} ({status})" for name, status in blockers.items())
        return Expectation(
            Expected.SKIPPED,
            reason=f"Requires {listed}, which cannot run.",
            blocked_by=tuple(blockers),
        )
    if profile is None:
        return Expectation(Expected.WILL_RUN)
    return _from_profile(check, profile)


def _from_profile(check: Check, profile: DatasetProfile) -> Expectation:
    eligible = eligibility(check.spec, profile)
    if not eligible.applicable:
        return Expectation(
            Expected.NOT_APPLICABLE, reason=eligible.reason, columns=eligible.columns
        )
    return Expectation(Expected.WILL_RUN, columns=eligible.columns)
