# dsbf/eda/task_loader.py

import importlib
import pkgutil
from pathlib import Path
from types import ModuleType


def load_all_tasks(tasks_package: str = "dsbf.eda.tasks") -> None:
    """
    Dynamically imports all Python modules in the given tasks package.
    Ensures that any @register_task decorators are executed.
    """
    module: ModuleType = importlib.import_module(tasks_package)
    module_file = getattr(module, "__file__", None)

    if module_file is None:
        raise ValueError(f"Could not determine __file__ for package: {tasks_package}")

    package_path = Path(module_file).parent

    for _, module_name, is_pkg in pkgutil.iter_modules([str(package_path)]):
        if not is_pkg and not module_name.startswith("_"):
            importlib.import_module(f"{tasks_package}.{module_name}")
