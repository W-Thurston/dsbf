# dsbf/config/__init__.py

from pathlib import Path
from typing import Any, Dict

import yaml


def load_default_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent / "default_config.yaml"
    with config_path.open("r") as f:
        return yaml.safe_load(f)
