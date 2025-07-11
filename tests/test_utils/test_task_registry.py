# tests/test_utils/test_task_registry.py

import re

from dsbf.eda.task_registry import _import_local_python_file


def strip_ansi_and_whitespace(text: str) -> str:
    """Remove ANSI codes and compress whitespace for easier matching."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)  # Strip ANSI color codes
    return " ".join(text.split())  # Collapse all whitespace to single spaces


def test_plugin_file_without_tasks_warns(capfd, tmp_path):
    plugin_path = tmp_path / "no_tasks_plugin.py"
    plugin_path.write_text("# no tasks defined\n")

    _import_local_python_file(plugin_path)

    captured = capfd.readouterr()
    clean_output = strip_ansi_and_whitespace(captured.out)

    assert "did not register any tasks" in clean_output
    assert "no_tasks_plugin.py" in clean_output
