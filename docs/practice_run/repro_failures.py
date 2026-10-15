"""Reproduce task crashes from the kitchen-sink run with full tracebacks."""

import os
import sys
import traceback

import pandas as pd

from dsbf.eda.task_registry import TASK_REGISTRY
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

path = sys.argv[1]
names = sys.argv[2:]
df = pd.read_csv(path)

for name in names:
    cls = TASK_REGISTRY[name].cls
    ctx, _ = make_ctx_and_task(
        task_cls=cls,
        current_df=df,
        global_overrides={"output_dir": os.path.abspath("runs/repro")},
    )
    try:
        r = run_task_with_dependencies(ctx, cls)
        print(f"== {name}: status={r.status}")
    except Exception:
        tb = traceback.format_exc().strip().splitlines()
        frames = [line for line in tb if 'File "/home' in line and "dsbf/" in line]
        print(f"== {name}: {tb[-1]}")
        for f in frames[-3:]:
            print("   ", f.strip())
