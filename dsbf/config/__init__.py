from pathlib import Path

import yaml


def load_default_config():
    config_path = Path(__file__).parent / "default_config.yaml"
    with config_path.open("r") as f:
        return yaml.safe_load(f)
