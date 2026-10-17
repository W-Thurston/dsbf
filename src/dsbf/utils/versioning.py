# dsbf/utils/versioning.py

import subprocess


def get_dsbf_version() -> str:
    """
    Return the current DSBF version, using the latest Git tag if available.
    Falls back to 'dev' if Git or tag is not available.

    Returns:
        str: Version string (e.g., 'v0.15.0' or 'dev')
    """
    try:
        version = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
        return version.decode("utf-8").strip()
    except Exception:
        return "dev"
