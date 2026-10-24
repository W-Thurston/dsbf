# Dashboard checklists for the archetype datasets

What to look at in the browser after profiling an archetype by hand. These were printed
by `run_validation.py` until the validation suite moved to pytest on `dsbf.profile()`
(`test_archetypes.py`), which checks the findings and levels behind these screens but
not the screens. The wording is that of the current dashboard, which still reads legacy
results; the dashboard is rebuilt in Phase 6, where these become end-to-end tests.

Profile the dataset (`uv run dsbf profile tests/datasets/clean.csv --name clean`, after
generating it with `tests/generators/dsbf_test_dataset_generator.py --only clean`), start
the API and the frontend as the README's dashboard section describes, and open the run.

## `clean`

- [ ] Overview: trust banner shows 'Looking Good' (green)
- [ ] Overview: DataHealthBar — all five traffic-light dots are green
- [ ] Overview: sample size adequacy metric is green (2,000 rows is adequate)
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

## `tiny`

- [ ] Overview: sample size adequacy metric shows a warning (25 rows is too small)
- [ ] Overview: DataHealthBar renders without errors
- [ ] Quality: all sections render (including with potentially sparse findings)
- [ ] Distributions: percentile table renders with 25-row data (some percentiles may duplicate)
- [ ] Distributions: selecting a column doesn't crash the detail panel
- [ ] Relationships: association table either shows pairs or an appropriate 'no data' state
- [ ] ML Readiness: no dimension shows an error-level finding triggered by tiny n alone
- [ ] No tab shows a blank white panel or uncaught error in the browser console

## `near_clean`

- [ ] Overview: trust banner shows amber ('A Few Things to Note')
- [ ] Overview: DataHealthBar — Completeness dot amber, all others green
- [ ] Quality: Completeness section auto-opens (income ~6% null)
- [ ] Quality: Usability, Validity, Redundancy, Leakage sections stay collapsed (All clear — plan_type 71% is below 95% flag threshold)
- [ ] ML Readiness: gate banner shows '⚠ Needs Work Before Modeling' (amber)
- [ ] ML Readiness: Transformations amber (purchase_amount log-transform warn)
- [ ] ML Readiness: Encoding amber (region/plan_type/segment encoding warn)
- [ ] ML Readiness: no dimension is red — all findings are warn-level, gate should be needs_work not not_ready
- [ ] ML Readiness: Column Status — no columns in Needs attention bucket

## `all_categorical`

- [ ] Overview: DataHealthBar — Usability dot amber/red (dominant column)
- [ ] Overview: no numeric columns in Column Types visualization
- [ ] Quality: Usability section open (dominant column flagged); all other sections collapsed
- [ ] Distributions: selecting any column shows no Outlier Analysis, no Normality section
- [ ] Distributions: selecting any column shows no Skewness findings in guidance panel
- [ ] Relationships: association table shows only Cramér's V — no Pearson r, no eta squared
- [ ] ML Readiness: Transformations dimension shows All clear (no continuous columns to transform)
- [ ] ML Readiness: Encoding dimension shows warn for tag (high-cardinality)
- [ ] No tab shows a blank white panel or uncaught JS error

## `high_missingness`

- [ ] Overview: DataHealthBar — Completeness dot red
- [ ] Quality: Completeness section auto-opens; findings for device_type, premium_score, notes
- [ ] Distributions: selecting income or age shows Outlier Analysis and Normality with reliability warning (sparse data)
- [ ] Distributions: selecting device_type or notes shows Value Counts with high null proportion visible
- [ ] ML Readiness: Missingness dimension is red, gate is '⚠ Not Ready' (notes 75% null is an error-level blocker)
- [ ] ML Readiness: expanding a Missingness finding shows imputation action suggestions
- [ ] No tab shows a blank white panel or uncaught JS error

## `severe_multicollinearity`

- [ ] Overview: DataHealthBar — Redundancy red, Leakage red, Completeness amber
- [ ] Quality: Redundancy section auto-opens (9 cols flagged VIF > 10); Leakage section auto-opens (base/copy_a r≈0.99 flagged)
- [ ] Quality: revenue and click_rate appear in Clean Columns section (VIF ~1.0 — genuinely independent despite extreme scale difference)
- [ ] Relationships: correlation heatmap shows obvious block structure in {base, copy_a, copy_b, combo} cluster and separate {factor_x, factor_y, factor_z} cluster
- [ ] ML Readiness: gate shows Not Ready (error-level leakage from near-perfect copies)
- [ ] ML Readiness: Leakage dimension red; no Redundancy dimension (collinearity is a Quality concern, not ML Readiness)
- [ ] No tab shows a blank white panel or uncaught JS error

## `single_column`

- [ ] Overview: DataHealthBar renders without errors (1 column)
- [ ] Overview: Column Types visualization shows a single continuous bar — no crash on 1-column dataset
- [ ] Distributions: column browser shows only 'value'; selecting it shows skewness finding and outlier analysis
- [ ] Relationships: association table shows 'No associations' or empty state — 0 pairs with 1 column
- [ ] Relationships: correlation heatmap shows empty state or single-cell 1x1 matrix — no crash
- [ ] ML Readiness: Transformations shows log-transform warn for 'value'
- [ ] No tab shows a blank white panel or uncaught JS error

## `wide`

- [ ] Overview: DataHealthBar — Completeness amber, Encoding amber/red
- [ ] Distributions: column browser scrolls through 100 columns without layout break or JS error
- [ ] Distributions: selecting cont_clean_00 shows no warnings in any section
- [ ] Distributions: selecting cont_skew_00 shows log-transform warn in guidance
- [ ] Relationships: correlation heatmap renders for the 60 continuous columns without crash
- [ ] Relationships: association table populates for any selected column
- [ ] ML Readiness: Transformations section shows cont_skew_* columns; Encoding shows cat_bal_* and cat_hc
- [ ] No tab shows a blank white panel or uncaught JS error at 100-column scale
