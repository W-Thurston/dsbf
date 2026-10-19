# dsbf/config/__init__.py

from importlib.resources import files
from typing import Any

import yaml

from dsbf.config.loader import (
    ENV_PREFIX,
    PROJECT_FILE_NAME,
    ConfigError,
    ConfigIssue,
    find_project_file,
    load_config,
)
from dsbf.config.profile import (
    DEFAULT_PACKS,
    Backend,
    LimitsConfig,
    ProfileConfig,
    SampleConfig,
    SampleStrategy,
)

__all__ = [
    "DEFAULT_PACKS",
    "ENV_PREFIX",
    "PROJECT_FILE_NAME",
    "Backend",
    "ConfigError",
    "ConfigIssue",
    "LimitsConfig",
    "ProfileConfig",
    "SampleConfig",
    "SampleStrategy",
    "find_project_file",
    "load_config",
    "load_default_config",
]


def load_default_config() -> dict[str, Any]:
    config_text: str = (
        files("dsbf.config").joinpath("default_config.yaml").read_text(encoding="utf-8")
    )
    return yaml.safe_load(config_text)
