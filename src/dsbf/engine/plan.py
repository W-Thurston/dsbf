# dsbf/engine/plan.py
"""The ordered set of checks a run will perform, and how it is resolved."""

from collections.abc import Iterator
from dataclasses import dataclass

from dsbf.engine.check import Check
from dsbf.engine.errors import PlanError
from dsbf.engine.expectation import Expectation, expect
from dsbf.engine.registry import Registry
from dsbf.engine.selection import Selection, select
from dsbf.engine.validation import Params, resolve_params
from dsbf.model import CheckParams, DatasetProfile, Note, Severity


@dataclass(frozen=True)
class PlanEntry:
    """One planned check and where it sits among the others."""

    check: Check
    selected: bool
    required_by: tuple[str, ...]
    requires: tuple[str, ...]
    after: tuple[str, ...]
    level: int
    params: CheckParams
    expected: Expectation

    @property
    def id(self) -> str:
        return self.check.id


@dataclass(frozen=True)
class Plan:
    """The checks a run will perform, in an order every dependency respects.

    ``requires`` on an entry are checks it cannot run without: if one doesn't
    complete, the entry is skipped. ``after`` are checks it runs behind but never
    depends on. An entry's ``level`` is one more than the highest level among both,
    so the entries of one level are independent of each other.

    ``notes`` carries what went wrong loading plugins, and ``plan_empty`` when no
    check was chosen. Each entry carries its validated ``params`` and an ``expected``
    status. When the plan was made without a profile (``profiled`` is False) that
    status judges only the installed extras and the requirements between checks, not
    the columns.
    """

    entries: tuple[PlanEntry, ...]
    selection: Selection
    notes: tuple[Note, ...] = ()
    profiled: bool = False

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.entries)

    @property
    def levels(self) -> tuple[tuple[PlanEntry, ...], ...]:
        count = 1 + max((entry.level for entry in self.entries), default=-1)
        return tuple(
            tuple(entry for entry in self.entries if entry.level == level)
            for level in range(count)
        )

    def get(self, check_id: str) -> PlanEntry:
        for entry in self.entries:
            if entry.id == check_id:
                return entry
        raise KeyError(check_id)

    def __contains__(self, check_id: object) -> bool:
        return any(entry.id == check_id for entry in self.entries)

    def __iter__(self) -> Iterator[PlanEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def make_plan(
    registry: Registry,
    selection: Selection | None = None,
    *,
    params: Params | None = None,
    profile: DatasetProfile | None = None,
) -> Plan:
    """Plan a run: select checks, pull in what they require, order and validate them.

    The plugins named in ``selection`` go into ``registry`` first. That is safe to
    repeat, and a plugin that fails to load leaves a note on the registry instead of
    stopping the plan.

    ``params`` maps a check id to the parameters it runs with; each is validated
    against the check's model. ``profile`` lets the plan judge which columns each
    check would read.

    Raises:
        PlanError: a selection mistake, a required check that is missing or
            excluded, a dependency cycle, or a parameter that is unknown or out of
            bounds. Parameter mistakes are all listed together.
    """
    selection = selection or Selection()
    for path in selection.plugins:
        registry.load_directory(path)
    if selection.entry_points:
        registry.load_entry_points()

    selected = {check.id: check for check in select(registry, selection)}
    planned, required_by = _close_over_requirements(registry, selected, selection)
    levels = _order(planned)
    resolved = resolve_params(registry, planned.values(), params)

    entries: list[PlanEntry] = []
    expected: dict[str, Expectation] = {}
    for check_id in sorted(planned, key=lambda key: (levels[key], key)):
        planned_check = planned[check_id]
        expected[check_id] = expect(planned_check, profile, expected)
        entries.append(
            PlanEntry(
                check=planned_check,
                selected=check_id in selected,
                required_by=tuple(sorted(required_by.get(check_id, ()))),
                requires=planned_check.spec.requires.checks,
                after=tuple(a for a in planned_check.spec.after if a in planned),
                level=levels[check_id],
                params=resolved[check_id],
                expected=expected[check_id],
            )
        )
    return Plan(
        tuple(entries),
        selection,
        (*registry.notes, *(() if entries else _empty_note())),
        profiled=profile is not None,
    )


def _close_over_requirements(
    registry: Registry, selected: dict[str, Check], selection: Selection
) -> tuple[dict[str, Check], dict[str, set[str]]]:
    planned = dict(selected)
    required_by: dict[str, set[str]] = {}
    problems: list[str] = []
    queue = list(selected.values())
    while queue:
        current = queue.pop()
        for needed in current.spec.requires.checks:
            if needed in selection.exclude:
                problems.append(
                    f"{current.id!r} requires {needed!r}, which was excluded; "
                    f"exclude {current.id!r} too, or keep {needed!r}"
                )
            elif needed not in registry:
                problems.append(
                    f"{current.id!r} requires {needed!r}, which is not registered"
                )
            else:
                required_by.setdefault(needed, set()).add(current.id)
                if needed not in planned:
                    planned[needed] = registry.get(needed)
                    queue.append(planned[needed])
    if problems:
        raise PlanError(*sorted(set(problems)))
    return planned, required_by


def _order(planned: dict[str, Check]) -> dict[str, int]:
    """The level of each check; a cycle among hard and soft edges is an error."""
    upstream = {
        check_id: [u for u in check.spec.upstream if u in planned]
        for check_id, check in planned.items()
    }
    levels: dict[str, int] = {}
    remaining = set(planned)
    level = 0
    while remaining:
        ready = {c for c in remaining if all(u in levels for u in upstream[c])}
        if not ready:
            raise PlanError(_describe_cycle(planned, upstream, remaining))
        for check_id in ready:
            levels[check_id] = level
        remaining -= ready
        level += 1
    return levels


def _describe_cycle(
    planned: dict[str, Check], upstream: dict[str, list[str]], stuck: set[str]
) -> str:
    """Walk upstream edges from a stuck check until one repeats."""
    walk = [min(stuck)]
    while True:
        step = min(u for u in upstream[walk[-1]] if u in stuck)
        if step in walk:
            cycle = [*walk[walk.index(step) :], step]
            break
        walk.append(step)
    parts = []
    for later, earlier in zip(cycle, cycle[1:], strict=False):
        kind = "requires" if earlier in planned[later].spec.requires.checks else "after"
        parts.append(f"{later} {kind} {earlier}")
    return "dependency cycle: " + ", ".join(parts)


def _empty_note() -> tuple[Note, ...]:
    return (
        Note(
            code="plan_empty",
            message="The selection matched no checks, so nothing will run.",
            severity=Severity.WARN,
            hint="Widen the packs, depth, dimensions or tags, or name checks "
            "with include.",
        ),
    )
