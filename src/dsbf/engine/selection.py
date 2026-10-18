# dsbf/engine/selection.py
"""What a user asks a run to include, and how that picks checks from a registry."""

import difflib
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from dsbf.engine.check import Check, _require_name
from dsbf.engine.errors import PlanError
from dsbf.engine.registry import Registry
from dsbf.model import Cost, Dimension, Stage


class Depth(StrEnum):
    """How much work a run is willing to do, as a ceiling on check cost."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"

    @property
    def max_cost(self) -> Cost:
        return _DEPTH_COST[self]


_DEPTH_COST = {
    Depth.QUICK: Cost.CHEAP,
    Depth.STANDARD: Cost.MODERATE,
    Depth.DEEP: Cost.HEAVY,
}


def _tuple(values: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True, init=False)
class Selection:
    """Which checks a run wants.

    The candidates are the checks in ``packs``, or every check when no pack is named.
    The cost ceiling (the lower of ``depth`` and ``max_cost``), ``dimensions``,
    ``tags`` and ``exclude_stages`` then narrow them. Within one filter a check
    matches if it has any of the values. ``include`` adds checks by id whatever the
    filters say, and ``exclude`` removes by id last.
    """

    packs: tuple[str, ...]
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    depth: Depth
    max_cost: Cost | None
    dimensions: tuple[Dimension, ...]
    tags: tuple[str, ...]
    exclude_stages: tuple[Stage, ...]
    plugins: tuple[str, ...] = field(
        metadata={"doc": "Plugin directories to load into the registry first."}
    )
    entry_points: bool = field(
        metadata={"doc": "Load checks and packs that installed packages offer."}
    )

    def __init__(
        self,
        *,
        packs: Iterable[str] | str = (),
        include: Iterable[str] | str = (),
        exclude: Iterable[str] | str = (),
        depth: Depth | str = Depth.STANDARD,
        max_cost: Cost | str | None = None,
        dimensions: Iterable[Dimension | str] | str = (),
        tags: Iterable[str] | str = (),
        exclude_stages: Iterable[Stage | str] | str = (),
        plugins: Iterable[str | Path] | str | Path = (),
        entry_points: bool = True,
    ) -> None:
        if isinstance(plugins, (str, Path)):
            plugins = (plugins,)
        packs, include, exclude = _tuple(packs), _tuple(include), _tuple(exclude)
        for what, names in (
            ("pack", packs),
            ("include", include),
            ("exclude", exclude),
        ):
            for name in names:
                _require_name(what, name)
        values = {
            "packs": packs,
            "include": include,
            "exclude": exclude,
            "depth": _enum(Depth, depth),
            "max_cost": None if max_cost is None else _enum(Cost, max_cost),
            "dimensions": tuple(_enum(Dimension, v) for v in _tuple(_text(dimensions))),
            "tags": _tuple(tags),
            "exclude_stages": tuple(
                _enum(Stage, v) for v in _tuple(_text(exclude_stages))
            ),
            "plugins": _tuple(str(path) for path in plugins),
            "entry_points": entry_points,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def cost_ceiling(self) -> Cost:
        """The most expensive check the run allows."""
        ceiling = self.depth.max_cost
        return min(ceiling, self.max_cost) if self.max_cost else ceiling


def _text(values: Iterable[object] | str) -> Iterable[str]:
    return (values,) if isinstance(values, str) else (str(v) for v in values)


def _enum[E: StrEnum](kind: type[E], value: object) -> E:
    try:
        return kind(value)
    except ValueError:
        choices = ", ".join(member.value for member in kind)
        raise PlanError(
            f"{value!r} is not a {kind.__name__.lower()}; choose from {choices}"
        ) from None


def select(registry: Registry, selection: Selection) -> tuple[Check, ...]:
    """The checks ``selection`` picks, ordered by id.

    Raises:
        PlanError: a pack or check id that the registry doesn't hold, or an id both
            included and excluded. Every such mistake is listed.
    """
    problems = _unknown(registry, selection)
    overlap = sorted(set(selection.include) & set(selection.exclude))
    problems += [f"{name!r} is both included and excluded" for name in overlap]
    if problems:
        raise PlanError(*problems)

    packs = registry.packs
    if selection.packs:
        names = {i for pack in selection.packs for i in packs[pack]}
        candidates = [registry.get(i) for i in sorted(names)]
    else:
        candidates = list(registry.checks)
    chosen = {c.id: c for c in candidates if _passes(c, selection)}
    for check_id in selection.include:
        chosen[check_id] = registry.get(check_id)
    for check_id in selection.exclude:
        chosen.pop(check_id, None)
    return tuple(chosen[key] for key in sorted(chosen))


def _unknown(registry: Registry, selection: Selection) -> list[str]:
    problems = []
    packs = registry.packs
    for name in selection.packs:
        if name not in packs:
            problems.append(f"unknown pack {name!r}{_hint(name, packs, 'packs')}")
    for what, ids in (("include", selection.include), ("exclude", selection.exclude)):
        for check_id in ids:
            if check_id not in registry:
                problems.append(
                    f"{what} names unknown check {check_id!r}"
                    f"{_hint(check_id, [c.id for c in registry], 'checks')}"
                )
    return problems


def _hint(name: str, known: Iterable[str], plural: str) -> str:
    pool = sorted(known)
    close = difflib.get_close_matches(name, pool, n=1)
    if close:
        return f"; did you mean {close[0]!r}?"
    return (
        f"; registered {plural}: {', '.join(pool)}"
        if pool
        else f"; no {plural} are registered"
    )


def _passes(candidate: Check, selection: Selection) -> bool:
    spec = candidate.spec
    return (
        spec.cost <= selection.cost_ceiling
        and (
            not selection.dimensions
            or bool(set(spec.dimensions) & set(selection.dimensions))
        )
        and (not selection.tags or bool(set(spec.tags) & set(selection.tags)))
        and spec.stage not in selection.exclude_stages
    )
