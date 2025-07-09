# dsbf/utils/logging_utils.py

import logging
import os
from typing import cast

from rich.logging import RichHandler

# Custom levels between WARNING and DEBUG
STAGE_LEVEL = 25
INFO2_LEVEL = 15
QUIET_LEVEL = 100  # Custom level above CRITICAL

logging.addLevelName(STAGE_LEVEL, "STAGE")
logging.addLevelName(INFO2_LEVEL, "INFO2")
logging.addLevelName(QUIET_LEVEL, "QUIET")


class DSBFLogger(logging.Logger):
    def stage(self, msg, *args, **kwargs):
        if self.isEnabledFor(STAGE_LEVEL):
            self._log(STAGE_LEVEL, msg, args, **kwargs)

    def info2(self, msg, *args, **kwargs):
        if self.isEnabledFor(INFO2_LEVEL):
            self._log(INFO2_LEVEL, msg, args, **kwargs)


# Register globally
logging.setLoggerClass(DSBFLogger)


def setup_logger(
    name="dsbf", level: str = "info", output_dir: str | None = None
) -> DSBFLogger:
    """
    Create a DSBF logger with optional Rich console and file output.

    Args:
        name (str): Logger name.
        level (str): One of "quiet", "warn", "stage", "info", "debug".
        output_dir (str | None): If provided, also logs to output_dir/run.log

    Returns:
        DSBFLogger
    """
    level_map = {
        "quiet": QUIET_LEVEL,
        "warn": logging.WARNING,
        "stage": STAGE_LEVEL,
        "info": INFO2_LEVEL,
        "debug": logging.DEBUG,
    }
    logging_level = level_map.get(level.lower(), INFO2_LEVEL)

    # Ensure logger is built as DSBFLogger
    logger = logging.getLogger(name)
    if not isinstance(logger, DSBFLogger):
        logging.setLoggerClass(DSBFLogger)
        logging.Logger.manager.loggerDict.pop(name, None)
        logger = logging.getLogger(name)

    logger.setLevel(logging_level)
    logger.handlers.clear()

    if logging_level >= QUIET_LEVEL:
        logger.disabled = True
        return cast(DSBFLogger, logger)

    # Console (Rich)
    rich_handler = RichHandler(rich_tracebacks=True, markup=True)
    rich_handler.setLevel(logging_level)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    # Optional file logging
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "run.log")
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    logger.propagate = False
    return cast(DSBFLogger, logger)
