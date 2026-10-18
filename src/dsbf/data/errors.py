# dsbf/data/errors.py
"""Typed errors for sources DSBF cannot load, each carrying a hint for the caller."""


class SourceError(Exception):
    """A source could not be loaded. ``hint`` says what to try next."""

    def __init__(self, message: str, *, source: str, hint: str | None = None) -> None:
        self.message = message
        self.source = source
        self.hint = hint
        super().__init__(message if hint is None else f"{message}\nHint: {hint}")


class UnsupportedSourceError(SourceError):
    """The source is a kind or format DSBF does not load (yet)."""


class SourceReadError(SourceError):
    """The source is a supported kind but is missing, empty, corrupt or malformed."""


class SourceDependencyError(SourceError):
    """Loading the source needs an optional package that is not installed."""


class SamplingError(ValueError):
    """A sample was asked for that DSBF cannot draw as requested.

    Unlike the source errors this is about the request, not the data: a missing
    stratification column, or a seed given to a strategy that has none.
    """
