"""The check interface depends on the data and the profile, not on the legacy engine."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

LEGACY = ("dsbf.eda", "dsbf.core", "dsbf.interfaces", "dsbf.api", "dsbf.storage")


def test_importing_the_engine_pulls_in_no_legacy_module():
    code = (
        "import sys, dsbf.engine; "
        f"loaded = [m for m in {LEGACY!r} if m in sys.modules]; "
        "print(','.join(loaded))"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == ""
