# dsbf/paths.py
"""
Locations DSBF writes to at run time.

Everything lives under one home directory, resolved in this order:
    1. DSBF_HOME environment variable
    2. The per-user data directory (platformdirs), e.g. ~/.local/share/dsbf

The home is read on every call and is never created here, so importing DSBF
has no side effects and a changed environment takes effect immediately.
"""

import os
from pathlib import Path

from platformdirs import user_data_dir

HOME_ENV_VAR = "DSBF_HOME"


def dsbf_home() -> Path:
    """Return the absolute DSBF home directory."""
    override = os.environ.get(HOME_ENV_VAR)
    base = Path(override) if override else Path(user_data_dir("dsbf"))
    return base.expanduser().resolve()


def default_outputs_dir() -> Path:
    """Return the directory that holds one subdirectory per run."""
    return dsbf_home() / "outputs"
