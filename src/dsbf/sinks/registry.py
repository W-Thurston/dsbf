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


def register_sink(sink: Sink) -> None:
    """Add a sink; a name can be registered once."""
    if sink.name in _SINKS:
        raise ValueError(f"sink {sink.name!r} is already registered")
    _SINKS[sink.name] = sink


def known_sinks() -> tuple[str, ...]:
    """The names of the registered sinks, in registration order."""
    return tuple(_SINKS)


def get_sinks(names: tuple[str, ...]) -> list[Sink]:
    """The sinks for ``names``; every unknown name is listed in one error."""
    unknown = [name for name in names if name not in _SINKS]
    if unknown:
        raise ValueError(
            f"unknown sink {', '.join(map(repr, unknown))}; "
            f"known sinks: {', '.join(_SINKS)}"
        )
    return [_SINKS[name] for name in dict.fromkeys(names)]
