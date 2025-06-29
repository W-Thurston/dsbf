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
from datetime import datetime

MESSAGE_VERBOSITY_LEVELS = {"quiet": 0, "info": 1, "debug": 2}


class BaseEngine(abc.ABC):
    def __init__(self, config):
        self.config = config
        self.message_verbosity = self.config.get("metadata", {}).get(
            "message_verbosity", "info"
        )
        self.verbosity_level = MESSAGE_VERBOSITY_LEVELS.get(self.message_verbosity, 1)
        self.output_dir = self._create_output_dir()
        self.timestamp = os.path.basename(self.output_dir)
        self.fig_path = os.path.join(self.output_dir, "figs")
        self.layout_name = self.config.get("metadata", {}).get("layout_name", "default")

        self.run_metadata = {
            "engine": self.__class__.__name__,
            "timestamp": self.timestamp,
            "layout_name": self.layout_name,
            "config": self.config,
        }

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
