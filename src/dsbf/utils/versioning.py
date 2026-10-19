# dsbf/utils/versioning.py

from importlib.metadata import PackageNotFoundError, version

UNKNOWN_VERSION = "0+unknown"


def get_dsbf_version() -> str:
    """
    Return the version of the installed ``dsbf`` package.

    It is read from the package metadata, so it names DSBF and not whatever
    repository the working directory happens to be in. Where the package is not
    installed (a bare source tree on ``PYTHONPATH``), returns ``UNKNOWN_VERSION``.

    Returns:
        str: Version string (e.g., '0.20.0')
    """
    try:
        return version("dsbf")
    except PackageNotFoundError:
        return UNKNOWN_VERSION
