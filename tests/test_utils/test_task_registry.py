# tests/test_utils/test_task_registry.py

from dsbf.eda.task_registry import _import_local_python_file


def test_plugin_file_without_tasks_warns(capfd, tmp_path):
    plugin_path = tmp_path / "no_tasks_plugin.py"
    plugin_path.write_text("# no tasks defined\n")

    _import_local_python_file(plugin_path)
    captured = capfd.readouterr()
    assert "did not register any tasks" in captured.out
