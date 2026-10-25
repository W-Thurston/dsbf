"""The legacy engine is gone, and the check interface needs no application layer."""

import importlib.util
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

REMOVED = ("dsbf.eda", "dsbf.core", "dsbf.legacy", "dsbf.api", "dsbf.storage")


@pytest.mark.parametrize("name", REMOVED)
def test_the_removed_packages_do_not_exist(name):
    assert importlib.util.find_spec(name) is None


def test_importing_the_engine_pulls_in_no_interface_module():
    code = (
        "import sys, dsbf.engine; "
        "names = ('dsbf.interfaces', 'dsbf.library'); "
        "loaded = [m for m in names if m in sys.modules]; "
        "print(','.join(loaded))"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == ""
