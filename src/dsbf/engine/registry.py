# dsbf/engine/registry.py
"""The set of checks a run can choose from."""

import difflib
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path

from dsbf.engine import plugins
from dsbf.engine.check import Check
from dsbf.engine.errors import RegistryError
from dsbf.engine.pack import Pack
from dsbf.model import Note, Severity

DIRECT = "direct"

# Called with (new check, held check, held source, new source) on an id clash.
ConflictHandler = Callable[[Check, Check, str, str], None]


class Registry:
    """Checks by id, the packs they belong to, and where each came from.

    Nothing registers itself: a check joins a registry only through ``register``,
    ``register_pack`` or one of the loaders. Loading a plugin never raises for a
    broken plugin; it records a ``Note`` in ``notes`` and carries on, so the run can
    report it.
    """

    def __init__(self, checks: Iterable[Check] = ()) -> None:
        self._checks: dict[str, Check] = {}
        self._sources: dict[str, str] = {}
        self._pack_members: dict[str, dict[str, None]] = {}
        self._loaded: set[str] = set()
        self._notes: list[Note] = []
        self.register(*checks)

    def register(self, *checks: Check) -> None:
        """Add checks. The same check twice is a no-op.

        Raises:
            RegistryError: a different check already holds one of the ids. Nothing
                is added in that case.
        """
        self._add(checks, DIRECT, conflict=_raise)

    def register_pack(self, pack: Pack) -> None:
        """Add a pack's checks and record them as its members.

        Raises:
            RegistryError: as for ``register``.
        """
        self._add_pack(pack, DIRECT, conflict=_raise)

    def load_directory(self, path: str | Path) -> None:
        """Import a local plugin directory and add the checks and packs in it.

        Loading the same directory again does nothing, in this registry or another
        in the same process. Whatever goes wrong is recorded in ``notes``.
        """
        resolved = Path(path).expanduser().resolve()
        self._load(
            f"directory {resolved}", lambda: plugins.discover_directory(resolved)
        )

    def load_entry_points(self) -> None:
        """Add what installed packages offer under ``dsbf.checks`` and ``dsbf.packs``.

        Loading again does nothing for entry points already loaded; a package
        installed since is picked up.
        """
        for group in (plugins.CHECKS_GROUP, plugins.PACKS_GROUP):
            for point in plugins.entry_points(group):
                self._load(
                    f"entry point {group}:{point.name}",
                    lambda point=point, group=group: plugins.discover_entry_point(
                        point, group
                    ),
                )

    def get(self, check_id: str) -> Check:
        """The check with this id.

        Raises:
            RegistryError: no such check; the message suggests close ids.
        """
        try:
            return self._checks[check_id]
        except KeyError:
            close = difflib.get_close_matches(check_id, self._checks, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise RegistryError(f"unknown check {check_id!r}{hint}") from None

    def source_of(self, check_id: str) -> str:
        """Where the check came from: ``direct``, an entry point or a directory."""
        self.get(check_id)
        return self._sources[check_id]

    @property
    def checks(self) -> tuple[Check, ...]:
        """Every check, ordered by id."""
        return tuple(self._checks[key] for key in sorted(self._checks))

    @property
    def packs(self) -> Mapping[str, tuple[str, ...]]:
        """Pack name to the ids of its checks, from ``Pack`` objects and each spec."""
        members: dict[str, dict[str, None]] = {
            name: dict(ids) for name, ids in self._pack_members.items()
        }
        for check in self._checks.values():
            for name in check.spec.packs:
                members.setdefault(name, {})[check.id] = None
        return {name: tuple(sorted(members[name])) for name in sorted(members)}

    @property
    def notes(self) -> tuple[Note, ...]:
        """What went wrong loading plugins, in the order it happened."""
        return tuple(self._notes)

    def __contains__(self, check_id: object) -> bool:
        return check_id in self._checks

    def __iter__(self) -> Iterator[Check]:
        return iter(self.checks)

    def __len__(self) -> int:
        return len(self._checks)

    def _add(
        self, checks: Iterable[Check], source: str, conflict: ConflictHandler
    ) -> None:
        pending: dict[str, Check] = {}
        for new in checks:
            held = pending.get(new.id) or self._checks.get(new.id)
            if held is None:
                pending[new.id] = new
            elif held is not new:
                conflict(new, held, self._sources.get(new.id, source), source)
        for check_id, new in pending.items():
            self._checks[check_id] = new
            self._sources[check_id] = source

    def _add_pack(self, pack: Pack, source: str, conflict: ConflictHandler) -> None:
        self._add(pack.checks, source, conflict)
        members = self._pack_members.setdefault(pack.name, {})
        for member in pack.checks:
            if self._checks.get(member.id) is member:
                members[member.id] = None

    def _load(self, source: str, discover: Callable[[], plugins.Discovery]) -> None:
        if source in self._loaded:
            return
        self._loaded.add(source)
        discovery = discover()
        self._notes.extend(discovery.notes)
        for item in discovery.contributions:
            if isinstance(item, Pack):
                self._add_pack(item, source, conflict=self._note_conflict)
            else:
                self._add((item,), source, conflict=self._note_conflict)

    def _note_conflict(
        self, new: Check, held: Check, held_source: str, new_source: str
    ) -> None:
        self._notes.append(
            Note(
                code="check_id_conflict",
                message=(
                    f"{new_source} declares check {new.id!r}, already declared by "
                    f"{held_source}; kept the first and ignored this one."
                ),
                severity=Severity.WARN,
                hint="Give the plugin's check an id of its own, "
                "or load only one of the two.",
            )
        )


def _raise(new: Check, held: Check, held_source: str, new_source: str) -> None:
    raise RegistryError(
        f"check {new.id!r} is already registered by {held_source}; "
        "a check id names one check. Give the second an id of its own."
    )
