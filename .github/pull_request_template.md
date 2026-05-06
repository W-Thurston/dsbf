## What does this PR do?

<!-- One or two sentences. What problem does it solve or what does it add? -->

## Why?

<!-- Context for the change. Link to the relevant issue if one exists. Closes #____ -->

## Type of change

- [ ] Bug fix
- [ ] New task
- [ ] Dashboard / UI change
- [ ] Refactor (no behaviour change)
- [ ] Documentation
- [ ] CI / tooling

## How was it tested?

<!-- Which tests cover this change? Did you run the full suite? -->

```bash
poetry run pytest tests/path/to/relevant_test.py -v
```

## Checklist

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `poetry run pytest` passes (or failing tests are documented below)
- [ ] Docstrings added or updated on new public classes and methods
- [ ] Type annotations on new function signatures

**If this PR adds or changes a task:**
- [ ] `task_metadata.yaml` updated
- [ ] Uses `get_columns_by_intent()` — not `df.columns` or `select_dtypes()`
- [ ] Zero-eligible-columns edge case handled (returns `status="success"` with empty data)
- [ ] Unit test covers the happy path and the empty-input case
- [ ] Validation suite run locally before opening PR: `poetry run python tests/validation/run_validation.py`

**If this PR changes the Quality or ML Readiness scoring:**
- [ ] `data_quality_scorer._level()` thresholds understood and intentionally changed
- [ ] Existing validation dataset assertions still pass or updated assertions included

## Notes for reviewer

<!-- Anything that needs extra attention, known limitations, or follow-up work. -->
