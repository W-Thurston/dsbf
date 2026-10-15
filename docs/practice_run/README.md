# Practice-run scripts (2026-10-15)

These scripts produced the measurements in [`../WALKTHROUGH.md`](../WALKTHROUGH.md). They're
kept as-is for reference until the performance benchmark harness in `tests/perf/` replaces them.

| Script | Purpose |
|---|---|
| `make_data.py` | Generates `kitchen_5000.csv`, `kitchen_50000.csv` and `adversarial.csv` into `./data` |
| `run_stages.py` | Runs one instrumented profile: stage timers, cProfile, per-task durations, output sizes |
| `api_probe.py` | Times every dashboard API endpoint (needs `httpx`; the walkthrough used `curl` against uvicorn instead) |
| `repro_failures.py` | Re-runs named tasks with full tracebacks |

Run them from a scratch directory with
`DSBF_DB_PATH=$PWD/practice.db DSBF_AUTO_EXPORT_METADATA=0 PYTHONPATH=<repo>` so nothing touches
the real database or outputs:

    python make_data.py
    python run_stages.py titanic titanic
    python run_stages.py kitchen5k data/kitchen_5000.csv --ts normal_date
