# dsbf/interfaces/api.py

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd
import polars as pl
import yaml

from dsbf.config import load_default_config
from dsbf.eda.profile_engine import ProfileEngine


class EDA:
    def __init__(
        self,
        dataset: str | pd.DataFrame | pl.DataFrame,
        config: dict[str, Any] | None = None,
    ):
        """
        Args:
            dataset: Path to CSV or in-memory DataFrame.
            config: Optional path to config YAML or pre-loaded dict.
        """
        if isinstance(config, str):
            with open(config) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config or load_default_config()

        self.config = cast(dict[str, Any], self.config)

        if isinstance(dataset, pd.DataFrame | pl.DataFrame):
            self.df = dataset
            self.config["metadata"]["dataset_path"] = None
        elif isinstance(dataset, str) and Path(dataset).exists():
            self.df = None
            self.config["metadata"]["dataset_path"] = dataset
        else:
            raise ValueError("Invalid dataset input. Must be path or DataFrame.")

        self.engine = ProfileEngine(self.config)

    def run(self):
        if self.df is not None:
            self.engine._log("Using in-memory DataFrame input.", "stage")
            self.engine._load_data = cast(
                Callable[[], pd.DataFrame | pl.DataFrame], lambda: self.df
            )
        self.engine.run()
        return self.engine.get_all_results()
