# dsbf/config/__init__.py

from importlib.resources import files
from typing import Any

import yaml


def load_default_config() -> dict[str, Any]:
    config_text: str = (
        files("dsbf.config").joinpath("default_config.yaml").read_text(encoding="utf-8")
    )
    return yaml.safe_load(config_text)
