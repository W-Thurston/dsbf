# dsbf/engine/pack.py
"""A named group of checks that ships together."""

from collections.abc import Iterable
from dataclasses import dataclass

from dsbf.engine.check import Check, _require_name
from dsbf.engine.errors import CheckDefinitionError


@dataclass(frozen=True, init=False)
class Pack:
    """Checks a plugin offers as one unit, such as ``healthcare``.

    A check also joins a pack by naming it in its own ``packs``; a ``Pack`` is what
    the ``dsbf.packs`` entry-point group loads, and registering it registers its
    checks.
    """

    name: str
    checks: tuple[Check, ...]

    def __init__(self, name: str, checks: Iterable[Check]) -> None:
        _require_name("pack", name)
        members = tuple(checks)
        for member in members:
            if not isinstance(member, Check):
                raise CheckDefinitionError(
                    f"pack {name}: {member!r} is not a Check; build one with @check"
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "checks", tuple(dict.fromkeys(members)))
