# tests/test_plugin_orphan_warning.py

import re
import tempfile
import textwrap
from pathlib import Path

from dsbf.eda.task_registry import load_task_group


def strip_ansi_and_whitespace(text: str) -> str:
    """Remove ANSI codes and compress whitespace for easier matching."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)  # Strip ANSI color codes
    return " ".join(text.split())  # Collapse all whitespace to single spaces


def test_warns_on_plugin_file_with_no_task(capfd):
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

        # -- Check captured output --
        out, _ = capfd.readouterr()
        clean_out = strip_ansi_and_whitespace(out)

        assert "did not register any tasks" in clean_out
        assert "orphan_plugin.py" in clean_out
