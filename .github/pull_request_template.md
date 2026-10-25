## What does this PR do?

<!-- One or two sentences. What problem does it solve or what does it add? -->

## Why?

<!-- Context for the change. Link to the relevant issue if one exists. Closes #____ -->

## Type of change

- [ ] Bug fix
- [ ] New check
- [ ] Dashboard / UI change
- [ ] Refactor (no behaviour change)
- [ ] Documentation
- [ ] CI / tooling

## How was it tested?

<!-- Which tests cover this change? Did you run the full suite? -->

```bash
uv run pytest tests/path/to/relevant_test.py -v
```

## Checklist

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `uv run pytest` passes (or failing tests are documented below)
- [ ] Docstrings added or updated on new public classes and methods
- [ ] Type annotations on new function signatures

**If this PR adds or changes a check:**
- [ ] The check is in `core_pack()`, and its `version` is bumped if its results changed
- [ ] Columns come from `applies_to` and `ctx.columns` — not `df.columns` or `select_dtypes()`
- [ ] Every finding states `n_used` and the method
- [ ] Unit test covers the happy path and an edge case
- [ ] Archetype test added or updated in `tests/validation/test_archetypes.py`

**If this PR changes the Quality or ML Readiness summaries (`quality_dimensions`, `ml_readiness`):**
- [ ] The level rules in `checks/synthesis.py` are understood and intentionally changed
- [ ] Existing archetype tests still pass or updated ones are included, and the `calibration` label is on the PR

## Notes for reviewer

<!-- Anything that needs extra attention, known limitations, or follow-up work. -->
