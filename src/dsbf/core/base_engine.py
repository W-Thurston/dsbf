# dsbf/core/base_engine.py

"""
Base Engine module for DSBF.

Defines the abstract `BaseEngine` class used to implement modular data processing
engines (e.g., ProfileEngine). Handles run tracking, output directory setup, logging,
and metadata.
"""

import abc
import os
import platform
import subprocess
from datetime import datetime
from typing import Any

from dsbf.paths import default_outputs_dir
from dsbf.utils.logging_utils import DSBFLogger, get_log_fn, setup_logger
from dsbf.utils.versioning import get_dsbf_version


class BaseEngine(abc.ABC):
    """
    Initialize the base engine with a given configuration.

    Args:
        config (dict): Engine and metadata configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.message_verbosity = config.get("metadata", {}).get(
            "message_verbosity", "info"
        )

        output_dir_override = config.get("output_dir") or config.get(
            "metadata", {}
        ).get("output_dir")
        if output_dir_override:
            self.output_dir = output_dir_override
        else:
            self.output_dir = self._create_output_dir()
        self.timestamp = os.path.basename(self.output_dir)
        self.fig_path = os.path.join(self.output_dir, "figs")
        self.layout_name = config.get("metadata", {}).get("layout_name", "default")

        self.logger: DSBFLogger = setup_logger(
            "dsbf", self.message_verbosity, self.output_dir
        )

        metadata_cfg = config.get("metadata", {})
        self.run_metadata = {
            "engine": self.__class__.__name__,
            "timestamp": self.timestamp,
            "layout_name": self.layout_name,
            "dataset_name": metadata_cfg.get("dataset_name", ""),
            "dataset_source": metadata_cfg.get("dataset_source", ""),
            "profiling_depth": metadata_cfg.get("profiling_depth", ""),
            "message_verbosity": metadata_cfg.get("message_verbosity", ""),
            "visualize_dag": metadata_cfg.get("visualize_dag"),
        }

        # Append additional runtime environment metadata
        self.run_metadata.update(
            {
                "dsbf_version": get_dsbf_version(),
                "git_sha": self._get_git_sha(),
                "host": platform.node(),
                "os": platform.platform(),
                "python": platform.python_version(),
            }
        )

    def _log(self, msg: str, level: str = "info", task_name: str | None = None) -> None:
        """
        Verbosity-aware logger with indentation and Rich + file support.

        Args:
            msg (str): Message to log.
            level (str): One of "warn", "stage", "info", "debug".
        """
        INDENTATION = {
            "warn": "",
            "stage": "",
            "info": "  ",
            "debug": "   ",
        }
        prefix = INDENTATION.get(level, "  ")
        log_fn = get_log_fn(self.logger, level)
        log_fn(f"{prefix}{msg}")

    def _create_output_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = default_outputs_dir() / timestamp
        output_path.mkdir(parents=True, exist_ok=True)
        return str(output_path)

    @abc.abstractmethod
    def run(self):
        pass

    def _get_git_sha(self) -> str:
        """
        Attempt to retrieve the current Git commit SHA (short form).

        Returns:
            str: Short SHA of the current Git commit, or "unknown" if unavailable.
        """
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode("utf-8")
                .strip()
            )
        except Exception:
            return "unknown"
