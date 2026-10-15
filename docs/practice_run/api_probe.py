# ruff: noqa: E501  (temporary reference script; superseded by the tests/perf harness)
"""Time and size every dashboard API call for each run in the practice DB."""

import json
import sqlite3
import sys
import time

from fastapi.testclient import TestClient

from dsbf.api.main import app

db = sys.argv[1]
client = TestClient(app)
runs = (
    sqlite3.connect(db)
    .execute(
        "SELECT r.run_key, d.name, d.source_path FROM runs r JOIN datasets d ON d.id=r.dataset_id"
    )
    .fetchall()
)


def hit(path, **params):
    t0 = time.perf_counter()
    r = client.get(path, params=params)
    ms = (time.perf_counter() - t0) * 1000
    return r.status_code, len(r.content), round(ms, 1), r


for run_key, name, src in runs:
    print(f"\n=== {name} ({run_key}) source_path={src}")
    # What RunDetailView loads on open, plus the per-tab calls.
    calls = [
        (f"/api/runs/{run_key}", {}),
        (f"/api/runs/{run_key}/tasks", {}),
        (f"/api/runs/{run_key}/figures", {}),
        (f"/api/runs/{run_key}/dq-status", {}),
        (f"/api/runs/{run_key}/ml-readiness", {}),
        (f"/api/runs/{run_key}/associations", {}),
        (f"/api/runs/{run_key}/sample", {"n": 10}),
    ]
    tasks = {}
    for path, params in calls:
        code, size, ms, r = hit(path, **params)
        print(
            f"  {code}  {size / 1024:>9.1f} KB  {ms:>8.1f} ms  {path.split(run_key)[-1] or '/'}"
        )
        if path.endswith("/tasks") and code == 200:
            tasks = r.json()
        if path.endswith("/figures") and code == 200:
            figs = r.json()
            missing = 0
            for f in figs[:400]:
                c, *_ = hit(f"/api/figures/{f['id']}/file")
                missing += c != 200
            print(
                f"       figures: {len(figs)} rows, {missing} of {min(len(figs), 400)} files not servable"
            )
    cols = list((tasks.get("infer_types", {}).get("data") or {}).keys())
    if not cols:
        cols = list(((tasks.get("summarize_nulls") or {}).get("data") or {}).keys())
    if len(cols) >= 2:
        code, size, ms, r = hit(f"/api/runs/{run_key}/column-data", cols=cols[:2])
        print(
            f"  {code}  {size / 1024:>9.1f} KB  {ms:>8.1f} ms  /column-data?cols={cols[:2]}"
        )
        col = cols[0]
        for ep in ("associations", "correlations"):
            code, size, ms, r = hit(f"/api/runs/{run_key}/{ep}/{col}")
            print(f"  {code}  {size / 1024:>9.1f} KB  {ms:>8.1f} ms  /{ep}/{col}")
    code, size, ms, r = hit(f"/api/runs/{run_key}/column-data", cols=["does_not_exist"])
    print(
        f"  {code}  {size / 1024:>9.1f} KB  {ms:>8.1f} ms  /column-data?cols=['does_not_exist'] -> {r.text[:120]}"
    )

print("\n=== dataset list")
code, size, ms, r = hit("/api/datasets")
print(f"  {code} {size} B {ms} ms", json.dumps(r.json())[:400])
