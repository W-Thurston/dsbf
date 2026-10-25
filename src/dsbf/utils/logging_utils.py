# dsbf/utils/logging_utils.py

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from rich.logging import RichHandler
from rich.text import Text

# Custom levels between WARNING and DEBUG
STAGE_LEVEL = 25
INFO2_LEVEL = 15
QUIET_LEVEL = 100  # Custom level above CRITICAL

ROOT_LOGGER_NAME = "dsbf"

# DSBF's own level names, applied by its handlers; the global level-name table
# belongs to the host application and is left alone.
_LEVEL_NAMES = {STAGE_LEVEL: "STAGE", INFO2_LEVEL: "INFO2"}

_LEVELS = {
    "quiet": QUIET_LEVEL,
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "stage": STAGE_LEVEL,
    "info": INFO2_LEVEL,
    "debug": logging.DEBUG,
}

# A library installs no output of its own: with this handler a host that has not
# configured logging sees nothing, and records still reach any handler it adds.
logging.getLogger(ROOT_LOGGER_NAME).addHandler(logging.NullHandler())


def level_from_name(level: str) -> int:
    """Map a verbosity name (``quiet`` to ``debug``) to a logging level."""
    return _LEVELS.get(level.lower(), INFO2_LEVEL)


def _level_name(record: logging.LogRecord) -> str:
    return _LEVEL_NAMES.get(record.levelno, record.levelname)


class _ConsoleHandler(RichHandler):
    """A Rich handler that prints messages verbatim, whatever brackets they hold.

    Messages carry user column names such as ``[/x]``, so they must never be
    parsed as Rich markup.
    """

    def __init__(self, level: int) -> None:
        super().__init__(level=level, rich_tracebacks=True, markup=False)
        self.setFormatter(logging.Formatter("%(message)s"))

    def get_level_text(self, record: logging.LogRecord) -> Text:
        name = _level_name(record)
        return Text.styled(name.ljust(8), f"logging.level.{name.lower()}")


@contextmanager
def _attached(handler: logging.Handler) -> Iterator[logging.Handler]:
    """Attach ``handler`` to the ``dsbf`` logger, then remove and close it."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    previous_level = logger.level
    if previous_level == logging.NOTSET or handler.level < previous_level:
        logger.setLevel(handler.level)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)


@contextmanager
def console_logging(level: str = "info") -> Iterator[None]:
    """Print ``dsbf`` log records to the console, for the length of the block.

    Only an application (the CLI) should call this; the library never does.
    ``quiet`` prints nothing.
    """
    logging_level = level_from_name(level)
    if logging_level >= QUIET_LEVEL:
        yield
        return
    with _attached(_ConsoleHandler(logging_level)):
        yield
