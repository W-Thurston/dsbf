# Dashboard checklists for the archetype datasets

What to look at in the browser after profiling an archetype by hand. The validation suite
(`test_archetypes.py`) checks the findings and levels behind these screens, not the
screens. The dashboard is rebuilt in Phase 6 unit by unit, and each unit turns its part
of this checklist into Vitest component tests and a Playwright test (`frontend/e2e`),
then deletes the lines it replaced.

Generate an archetype (`tests/generators/dsbf_test_dataset_generator.py --only clean`),
keep a run of it in a scratch store, and open the dashboard on it:

```bash
export DSBF_HOME=$(mktemp -d)
uv run dsbf profile tests/datasets/clean.csv -s store
(cd frontend && pnpm build)
uv run dsbf dashboard
```

On every screen: switch to the light theme and check the text is readable, and check the
browser console shows no uncaught error.

## The overview (built)

Expected values are what `dsbf profile` printed on 2026-10-28.

## `clean`

- [ ] Run health: 33 ok, 13 not applicable, nothing else; every not-applicable check lists its reason
- [ ] Findings: 12, all `info`; no policy panel; no sampling notice
- [ ] Run notes: none

## `tiny`

- [ ] Run health: 36 ok, 10 not applicable; 25 rows × 6 columns in the header
- [ ] Findings: one `warn` (`too_few_rows_linear_model`), the rest `info`; the page renders with no console error

## `near_clean`

- [ ] Findings: 3 `warn` (`heavy_tails`, `outliers_detected`, `skewed`, all distribution), 23 `info`; highest severity badge reads `warn`
- [ ] Run health: 34 ok, 12 not applicable

## `all_categorical`

- [ ] Run health: 20 ok, 26 not applicable, and the not-applicable reasons name what is missing (a numeric column, or the unset `time_index` or `target`)
- [ ] Findings: 2 `warn`, 15 `info`

## `high_missingness`

- [ ] Findings: 1 `error` (`high_null_rate`, completeness), 6 `warn`; the `error` is first in Top findings and the highest badge reads `error`
- [ ] Findings by dimension: completeness has 5, with the `error` in its row

## `severe_multicollinearity`

- [ ] Findings: 1 `error` (`high_null_rate`), 9 `warn` (`redundant_columns`, `multicollinearity`); the `error` is first in Top findings
- [ ] Run health: 24 ok, 22 not applicable

## `single_column`

- [ ] Header reads 500 rows × 1 column; Run health: 19 ok, 27 not applicable (each with a reason); no crash on one column

## `wide`

- [ ] Header reads 1,000 rows × 100 columns; Findings: 49 `warn`, 221 `info`; Top findings lists 10 and says 'Showing 10 of 270'
- [ ] Opening the run stays under 200 KB (the Playwright test measures titanic)

## Findings explorer

Covered by `frontend/e2e/findings-explorer.spec.ts` and the Vitest tests beside the
components, so nothing here is checked by hand:

- filters, sort and page live in the URL and survive a reload;
- a filtered count equals the API's, which `tests/test_service/test_endpoints.py` holds
  to `run.findings.filter`, a pair finding under both of its columns;
- an opened finding shows its evidence and guidance, and a top finding on the overview
  opens it;
- a finding's artifact loads only on request and pages by row range.

## Tabs not yet built

Each belongs to the unit in `ROADMAP.md` that builds it (column view,
relationships, quality and ML readiness, time series). The wording is the old dashboard's
and is rewritten when its unit converts it into tests. Lines that mention a tab by its
old name ("Distributions", "Quality") describe what that tab must still show.

### `clean`

- [ ] Quality: trust banner shows 'Looking Good'
- [ ] Quality: all five dimension section headers show green dots
- [ ] Quality: clicking each dimension header opens an 'All clear' body (no findings)
- [ ] Quality: Clean Columns section contains all 9 columns
- [ ] Distributions: selecting each column shows no warning badges in section headers
- [ ] Distributions: Outlier Analysis shows 'Nothing flagged' for all columns
- [ ] Distributions: Normality section badge shows 'Consistent with normal' for numeric cols
- [ ] Relationships: Summary card shows 0 collinearity and 0 leakage warnings
- [ ] ML Readiness: gate banner shows '✓ Ready for Modeling' (green)
- [ ] ML Readiness: all five dimension summary cards show 'All clear'

### `tiny`

- [ ] Quality: all sections render (including with potentially sparse findings)
- [ ] Distributions: percentile table renders with 25-row data (some percentiles may duplicate)
- [ ] Distributions: selecting a column doesn't crash the detail panel
- [ ] Relationships: association table either shows pairs or an appropriate 'no data' state
- [ ] ML Readiness: no dimension shows an error-level finding triggered by tiny n alone
- [ ] No tab shows a blank white panel or uncaught error in the browser console

### `near_clean`

- [ ] Quality: Completeness section auto-opens (income ~6% null)
- [ ] Quality: Usability, Validity, Redundancy, Leakage sections stay collapsed (All clear — plan_type 71% is below 95% flag threshold)
- [ ] ML Readiness: gate banner shows '⚠ Needs Work Before Modeling' (amber)
- [ ] ML Readiness: Transformations amber (purchase_amount log-transform warn)
- [ ] ML Readiness: Encoding amber (region/plan_type/segment encoding warn)
- [ ] ML Readiness: no dimension is red — all findings are warn-level, gate should be needs_work not not_ready
- [ ] ML Readiness: Column Status — no columns in Needs attention bucket

### `all_categorical`

- [ ] Quality: Usability section open (dominant column flagged); all other sections collapsed
- [ ] Distributions: selecting any column shows no Outlier Analysis, no Normality section
- [ ] Distributions: selecting any column shows no Skewness findings in guidance panel
- [ ] Relationships: association table shows only Cramér's V — no Pearson r, no eta squared
- [ ] ML Readiness: Transformations dimension shows All clear (no continuous columns to transform)
- [ ] ML Readiness: Encoding dimension shows warn for tag (high-cardinality)
- [ ] No tab shows a blank white panel or uncaught JS error

### `high_missingness`

- [ ] Quality: Completeness section auto-opens; findings for device_type, premium_score, notes
- [ ] Distributions: selecting income or age shows Outlier Analysis and Normality with reliability warning (sparse data)
- [ ] Distributions: selecting device_type or notes shows Value Counts with high null proportion visible
- [ ] ML Readiness: Missingness dimension is red, gate is '⚠ Not Ready' (notes 75% null is an error-level blocker)
- [ ] ML Readiness: expanding a Missingness finding shows imputation action suggestions
- [ ] No tab shows a blank white panel or uncaught JS error

### `severe_multicollinearity`

- [ ] Quality: Redundancy section auto-opens (9 cols flagged VIF > 10); Leakage section auto-opens (base/copy_a r≈0.99 flagged)
- [ ] Quality: revenue and click_rate appear in Clean Columns section (VIF ~1.0 — genuinely independent despite extreme scale difference)
- [ ] Relationships: correlation heatmap shows obvious block structure in {base, copy_a, copy_b, combo} cluster and separate {factor_x, factor_y, factor_z} cluster
- [ ] ML Readiness: gate shows Not Ready (error-level leakage from near-perfect copies)
- [ ] ML Readiness: Leakage dimension red; no Redundancy dimension (collinearity is a Quality concern, not ML Readiness)
- [ ] No tab shows a blank white panel or uncaught JS error

### `single_column`

- [ ] Distributions: column browser shows only 'value'; selecting it shows skewness finding and outlier analysis
- [ ] Relationships: association table shows 'No associations' or empty state — 0 pairs with 1 column
- [ ] Relationships: correlation heatmap shows empty state or single-cell 1x1 matrix — no crash
- [ ] ML Readiness: Transformations shows log-transform warn for 'value'
- [ ] No tab shows a blank white panel or uncaught JS error

### `wide`

- [ ] Distributions: column browser scrolls through 100 columns without layout break or JS error
- [ ] Distributions: selecting cont_clean_00 shows no warnings in any section
- [ ] Distributions: selecting cont_skew_00 shows log-transform warn in guidance
- [ ] Relationships: correlation heatmap renders for the 60 continuous columns without crash
- [ ] Relationships: association table populates for any selected column
- [ ] ML Readiness: Transformations section shows cont_skew_* columns; Encoding shows cat_bal_* and cat_hc
- [ ] No tab shows a blank white panel or uncaught JS error at 100-column scale
