# dsbf/engine/extras.py
"""Which optional-dependency groups are installed, so a plan can say what is missing."""

from collections.abc import Iterable
from importlib.util import find_spec

from dsbf.engine.check import Extra

# The importable modules that show an extra is installed; all must be present.
EXTRA_MODULES: dict[Extra, tuple[str, ...]] = {
    "ml": ("sklearn",),
    "timeseries": ("statsmodels",),
    "text": ("rapidfuzz",),
    "excel": ("fastexcel",),
    "pandas": ("pandas",),
    "sql": ("ibis",),
    "server": ("fastapi", "uvicorn", "sqlalchemy", "alembic"),
}


def missing_modules(extra: Extra) -> tuple[str, ...]:
    """The modules of ``extra`` that cannot be imported."""
    return tuple(module for module in EXTRA_MODULES[extra] if find_spec(module) is None)


def missing_extras(extras: Iterable[Extra]) -> dict[Extra, tuple[str, ...]]:
    """The extras among ``extras`` that are not fully installed, with what's missing."""
    missing = {extra: missing_modules(extra) for extra in extras}
    return {extra: modules for extra, modules in missing.items() if modules}


def install_hint(extras: Iterable[Extra]) -> str:
    """How to install ``extras``, such as ``pip install 'dsbf[ml,text]'``."""
    names = ",".join(dict.fromkeys(extras))
    return f"Install it with `pip install 'dsbf[{names}]'`."
