# dsbf/sinks/registry.py
"""The registry of sinks: named writers that turn a ``Run`` into one file."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dsbf.model import Run


class RunReadError(ValueError):
    """A stored run could not be read; the message names the file and the problem."""


@dataclass(frozen=True)
class Sink:
    """One output format: the name a caller asks for, its file, and its writer."""

    name: str
    filename: str
    write: Callable[[Run, Path], None]


_SINKS: dict[str, Sink] = {}
_ALIASES: dict[str, tuple[str, ...]] = {}


def register_sink(sink: Sink) -> None:
    """Add a sink; a name can be registered once."""
    if sink.name in _SINKS:
        raise ValueError(f"sink {sink.name!r} is already registered")
    _SINKS[sink.name] = sink


def register_alias(alias: str, names: tuple[str, ...]) -> None:
    """Let ``alias`` stand for several sinks, for example one format in many files."""
    if alias in _SINKS or alias in _ALIASES:
        raise ValueError(f"sink {alias!r} is already registered")
    _ALIASES[alias] = names


def known_sinks() -> tuple[str, ...]:
    """The names a caller may ask for: the sinks in registration order, then aliases."""
    return (*_SINKS, *_ALIASES)


def get_sinks(names: tuple[str, ...]) -> list[Sink]:
    """The sinks for ``names``, aliases expanded; every unknown name is in one error."""
    unknown = [name for name in names if name not in _SINKS and name not in _ALIASES]
    if unknown:
        raise ValueError(
            f"unknown sink {', '.join(map(repr, unknown))}; "
            f"known sinks: {', '.join(known_sinks())}"
        )
    expanded = [m for name in names for m in _ALIASES.get(name, (name,))]
    return [_SINKS[name] for name in dict.fromkeys(expanded)]
