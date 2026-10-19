# tests/test_utils/test_task_registry.py

import pytest

from dsbf.eda.task_registry import _import_local_python_file

pytestmark = pytest.mark.unit


def test_plugin_file_without_tasks_warns(caplog, tmp_path):
    plugin_path = tmp_path / "no_tasks_plugin.py"
    plugin_path.write_text("# no tasks defined\n")

    _import_local_python_file(plugin_path)

    assert "did not register any tasks" in caplog.text
    assert "no_tasks_plugin.py" in caplog.text
