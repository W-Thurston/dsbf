# dsbf/core/base_engine.py
"""
Base Engine module for DSBF.

Defines the abstract `BaseEngine` class used to implement modular data processing
engines (e.g., ProfileEngine). Handles run tracking, output directory setup, logging,
and metadata.
"""
import abc
import json
import os
import platform
import subprocess
from datetime import datetime

MESSAGE_VERBOSITY_LEVELS = {"quiet": 0, "info": 1, "debug": 2}


class BaseEngine(abc.ABC):
    """
    Initialize the base engine with a given configuration.

    Args:
        config (dict): Engine and metadata configuration dictionary.
    """

    def __init__(self, config):
        self.config = config
        self.message_verbosity = config.get("metadata", {}).get(
            "message_verbosity", "info"
        )
        self.verbosity_level = MESSAGE_VERBOSITY_LEVELS.get(self.message_verbosity, 1)
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

        metadata_cfg = config.get("metadata", {})
        self.run_metadata = {
            "engine": self.__class__.__name__,
            "timestamp": self.timestamp,
            "layout_name": self.layout_name,
            "dataset_name": metadata_cfg.get("dataset_name", ""),
            "dataset_source": metadata_cfg.get("dataset_source", ""),
            "profiling_depth": metadata_cfg.get("profiling_depth", ""),
            "message_verbosity": metadata_cfg.get("message_verbosity", ""),
            "visualize_dag": self.config.get("metadata", {}).get("visualize_dag"),
        }

        # Append additional runtime environment metadata
        self.run_metadata.update(
            {
                "dsbf_version": "0.1.0",  # TODO: Replace with real version
                "git_sha": self._get_git_sha(),
                "host": platform.node(),
                "os": platform.platform(),
                "python": platform.python_version(),
            }
        )

    def _log(self, msg, level="info"):
        if self.verbosity_level >= MESSAGE_VERBOSITY_LEVELS[level]:
            print(f"[{self.__class__.__name__}] {msg}")

    def _create_output_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("dsbf/outputs", timestamp)
        os.makedirs(output_path, exist_ok=True)
        return output_path

    def record_run(self):
        record_path = os.path.join("dsbf_run.json")

        # Load existing history
        if os.path.exists(record_path):
            try:
                with open(record_path, "r") as f:
                    history = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                history = []
        else:
            history = []

        # Append current run (avoid duplicates by timestamp)
        timestamps = {r.get("timestamp") for r in history}
        if self.run_metadata.get("timestamp") not in timestamps:
            history.append(self.run_metadata)

        # Save updated history
        with open(record_path, "w") as f:
            json.dump(history, f, indent=2)

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
