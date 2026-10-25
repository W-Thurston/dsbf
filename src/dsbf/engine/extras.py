# dsbf/engine/extras.py
"""Which optional-dependency groups are installed, so a plan can say what is missing."""

from collections.abc import Iterable
from importlib import import_module
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


# What a check imports when it runs, because the core pack loads without its extra.
# ``preload_extras`` imports these on the calling thread before checks run side by
# side: a first import on a worker thread can hang. Loading a C extension takes the
# loader's lock, and scikit-learn's thread-pool controller holds that lock while it
# waits for the interpreter lock that the importing thread holds.
# ``tests/test_packaging.py`` checks the table against the imports in ``src``.
PRELOAD: dict[Extra, tuple[str, ...]] = {
    "ml": ("sklearn.ensemble", "sklearn.feature_selection", "sklearn.mixture"),
    "timeseries": (
        "statsmodels.stats.diagnostic",
        "statsmodels.tsa.seasonal",
        "statsmodels.tsa.stattools",
    ),
    "text": ("rapidfuzz.fuzz", "rapidfuzz.process"),
}


def preload_extras(extras: Iterable[Extra]) -> None:
    """Import the modules of each installed extra in ``extras`` on this thread."""
    for extra in dict.fromkeys(extras):
        if not missing_modules(extra):
            for module in PRELOAD.get(extra, ()):
                import_module(module)


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
