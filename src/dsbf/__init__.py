# dsbf/__init__.py
"""DSBF profiles a table and reports what it finds.

``dsbf.profile(df)`` returns the ``Run``. The names below load on first use, so
``import dsbf.model`` stays light.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dsbf.library import Sample, profile

__all__ = ["Sample", "__version__", "profile"]

_LAZY = {
    "profile": "dsbf.library",
    "Sample": "dsbf.library",
    "__version__": "dsbf.utils.versioning",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'dsbf' has no attribute {name!r}")
    if name == "__version__":
        value = import_module(module).get_dsbf_version()
    else:
        value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
