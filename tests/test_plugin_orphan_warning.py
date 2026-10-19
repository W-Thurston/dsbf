# tests/test_plugin_orphan_warning.py

import tempfile
import textwrap
from pathlib import Path

import pytest

from dsbf.eda.task_registry import load_task_group

pytestmark = pytest.mark.unit


def test_warns_on_plugin_file_with_no_task(caplog):
    # -- Create orphan plugin file --
    with tempfile.TemporaryDirectory() as tmp_dir:
        plugin_dir = Path(tmp_dir)
        orphan_file = plugin_dir / "orphan_plugin.py"

        orphan_file.write_text(
            textwrap.dedent(
                """
            def not_a_task():
                return "I do nothing"
        """
            )
        )

        # -- Run loader --
        load_task_group(str(plugin_dir))

        # -- Check logged records --
        assert "did not register any tasks" in caplog.text
        assert "orphan_plugin.py" in caplog.text
