# dsbf/utils/display_registered_tasks.py

from dsbf.eda.task_loader import load_all_tasks
from dsbf.eda.task_registry import describe_registered_tasks

if __name__ == "__main__":
    load_all_tasks()
    describe_registered_tasks()
