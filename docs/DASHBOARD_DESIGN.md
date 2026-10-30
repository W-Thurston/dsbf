# Dashboard design (Phase 6b)

The design record for the Phase 6b dashboard. The old dashboard is the baseline: every page unit in
Phase 6b starts from the inventory below and states and justifies each deviation from it. U6b-00
wrote this document; each page unit cites its section and updates the decision column when the user
signs a page off.

Status: the inventory's decisions were agreed with the user on 2026-10-29 (U6b-00). A page unit
that deviates from a decision records the change and its reason in the decision column. The visual
language is decided in U6b-01.

## Baseline sources

- **Source:** `git show 15a6ac1^:frontend/src/<path>`, the last commit before the TypeScript
  rebuild (Vue 3, ECharts, Plotly for backend figures). It was read in full for this inventory. The
  old build (`v0.20.0`, `c880348`, 2026-05-06) was not run.
- **Screenshot:** `docs/assets/Overview_Tab.png` (an NFL dataset, dark theme).
- **Where the README and the source disagree, the source wins:** the five traffic lights are a
  persistent bar on every tab, not part of the Overview; the correlation heatmap was a placeholder
  ("D3 heatmap visualization coming in the next refactor"); fuzzy duplicates are on the Quality tab,
  not Relationships; the Q-Q plot was a placeholder ("Q-Q plot coming soon"). Where the old dashboard
  only intended an element, the baseline is that intent.

## How to read the inventory

Each row is one element of the old dashboard.

- **Showed / carried by:** what the person learned and the visual that carried it (chart, traffic
  light, coloured cell, badge, text).
- **Data now:** `have` (the `/v2` API supplies it, with the source named), `need` (an additive field
  the API lacks) or `drop` (no `/v2` equivalent and not worth adding).
- **Decision:** keep, change or drop, with the reason for any deviation.

## Inventory

### Shell (every page)

| Element | Showed / carried by | Data now | Decision |
|---|---|---|---|
| Breadcrumb header | DSBF › dataset › run key; text links | `have`: run header | Keep. One persistent header, not a crumb line repeated per page as in Phase 6. |
| Data health bar | Five quality dimensions (Completeness, Validity, Usability, Redundancy, Leakage) as **traffic-light dots** with "All clear" or "N cols", above the tabs on every tab; hover shows the definition and "N of M columns (x%)"; dims the dimensions irrelevant to the current tab | `have`: `quality_dimensions` (level, coverage, findings, `assessed`) | Keep. Change: a fourth, grey state for an unassessed dimension (leakage without a target), which the old bar could not show. |
| Tab bar | Six tabs inside one run page; client state, not in the URL | `have` | Keep the six tabs as the run's structure. Change: each tab is a route, so links, reloads and the back button work (Phase 6 deep links to findings and columns depend on this). |
| Theme toggle | Light / dark, dark by default | `have`: theme store | Keep. |
| Dataset list | Card grid with search: runs, latest run date, quality score | `have`: `DatasetSummary` (`n_runs`, `last_run_at`, `last_max_severity`); the score is `drop` | Keep the grid. Change: the old README dropped aggregate scores, so the score becomes five dimension dots for the latest run, which is `need`: the latest run's dimension levels on `DatasetSummary`. |
| Run list | Searchable table of a dataset's runs | `have`: dataset runs | Keep. |

### Overview tab

| Element | Showed / carried by | Data now | Decision |
|---|---|---|---|
| Metric strip | Rows, columns, memory, missing %, duplicate rows (n and %), sample adequacy (coloured adequate / limited / insufficient), depth, stage; **text KPIs** | `have`: `n_rows`, `n_columns`, `approx_memory_bytes`, null counts (missing % derived), `duplicate_rows`, `sample_size.verdict`, `plan.depth`. Stage is `drop` (no `/v2` concept) | Keep, without stage. Add sampling (`SamplingInfo`) when the run was sampled. |
| Dataset sample | First 5/10/25/50 rows, nulls highlighted; table | `have`: `/runs/{id}/snapshot` (404 when no snapshot was kept) | Keep, with an empty state for runs without a snapshot. |
| Run information | Run date and key | `have`: header | Keep, folded into the header. |
| Dataset lineage | Dataset, file name, path, size, modified time | `have`: location, format, fingerprint, lineage; size and modified time are `drop` | Change: show location and format, plus whether the content changed since the previous run (fingerprint), which tells the person more than a file size. |
| Dataset metadata table | Per column: physical type, intent badge, null % (**coloured** at 5% and 20%), unique, mode, mode share (coloured at 70% and 90%), cardinality (the same number as unique), skew (coloured beyond ±1), range; sortable, filterable | `have`: name, semantic type, physical type, nulls, distinct (`/columns`). `need`: mode, mode share, skewness, min and max per column without a request per column | Keep. Drop the cardinality column (duplicates unique). Add an inline chart per row (a coarse histogram for numeric columns, a top-values bar for categorical ones) from the column sketch (U6b-02). |
| Data signals | Alerts derived in the browser from task outputs, filterable by priority (P1–P3) and type, each one an icon and a sentence | `have`: findings (severity, code, check, title, subject) | Keep as a ranked list of findings with filters; the findings explorer is its full version. Change: chart the counts by severity and dimension above the list (a U6-19 carry-in). |
| Column types | **Stacked bar** of columns per type (Plotly figure rendered by the backend) | `have`: semantic types from `/columns` | Keep, drawn in ECharts in the browser. |
| Missingness matrix | **Rows × columns nullity image** (static PNG from the backend) | Row matrix: `drop` as stored data (the snapshot could give one for its sample rows). `have`: per-column nulls and null co-occurrence pairs (`missingness.pairs`, Jaccard) | Change: a per-column null bar plus a null co-occurrence heatmap. It works for every run, with or without a snapshot, and shows which columns go missing together; a row matrix over a sample of rows says less. |
| Dtype optimisations | Collapsible table, current → suggested type and estimated saving; opens itself at ≥ 1 MB total | `have`: `dtype_optimizations` | Keep. |

### Quality tab

| Element | Showed / carried by | Data now | Decision |
|---|---|---|---|
| Trust banner | Overall level as a **coloured banner** with a label ("Looking good", "Worth investigating") and a sentence; columns checked, fully clean, dimensions flagged N/5 | `have`: `quality_dimensions` (`total_columns`, `clean_columns`, levels) | Keep. Change: "N of M assessed" when a dimension is unassessed. |
| Dimension cards | Five **traffic-light cards**: dot, affected count, % of dataset, issue-type chips; a click scrolls to the section | `have` | Keep. |
| Dimension sections | Collapsible per dimension; sortable findings table (column, finding, detail, severity badge); an expanded row shows metrics, sample values, guidance blurbs and action chips; a ☆ marks a row reviewed (browser only, not saved) | `have`: dimension findings with evidence and guidance | Keep the sections and expanded rows. Drop the reviewed star: unsaved, it was lost on reload; saving it per user is carried into Phase 8. |
| Clean columns | Chip grid of columns with no finding | `have`: `clean_columns` | Keep. |
| Missingness mechanism | Per column: evidence, **bars** for the predictors' correlation with missingness, caveats | `have`: `missingness_mechanism` (predictors with effect size, p-value and adjusted p) | Keep. |
| Fuzzy duplicates | Near-duplicate row pairs with a **similarity bar** | `have`: `fuzzy_duplicates` (`top_pairs`, coverage) | Keep, on the Quality tab where the source had it. |

### Distributions tab

| Element | Showed / carried by | Data now | Decision |
|---|---|---|---|
| Column browser | Left pane: columns grouped by type, search, a warning dot, a **null % bar** per column | `have`: `/columns`, findings by subject | Keep. |
| Column heading | Name, type badge, physical type | `have` | Keep. |
| Stats strip | Numeric: count, mean, median, std, min, max, skew, IQR, null %, outliers, kurtosis. Categorical: count, unique, top value, top %, null % | `have`: profile stats, quantiles, top values; `outliers` | Keep. |
| Distribution chart | Numeric: **histogram** with mean and median lines (bins approximated from percentiles), toggling to a **box plot** with fences. Categorical: **top-value bar chart** with a zoom slider | `have`: `Histogram` (true edges and counts), quantiles, top values | Keep. Change: use the real histogram, a strict improvement on bins derived from percentiles. |
| Guidance | Expandable insight blurbs with action chips, beside the chart | `have`: the column's findings with `guidance` | Keep. |
| Bimodal callout | A warning box with the BIC improvement | `have`: `bimodality` (means, stds, weights) | Keep. Overlaying the two fitted components on the histogram is prototyped in U6b-06. |
| Percentiles / value counts | Collapsible table; text-length card for text | `have`: quantiles, top values, lengths | Keep. |
| Outlier analysis | IQR, z-score, MAD and Isolation Forest counts with fences, plus a consensus banner | `have`: `outliers` (IQR, z, MAD, consensus); Isolation Forest is the separate `multivariate_outliers` check | Keep three methods and the consensus; U6b-07 places the multivariate check on Relationships or Quality. |
| Normality | Badge, KS and Jarque-Bera p-values, prose; Q-Q plot "coming soon" | `have`: `normality`, including `QQData` | Keep, and build the Q-Q plot (fulfils the old intent). |
| Top correlations | The column's strongest correlations; a click selects that column | `have`: `/runs/{id}/relationships` | Keep. |

### Relationships tab

| Element | Showed / carried by | Data now | Decision |
|---|---|---|---|
| Relationships summary | **Stacked strength bar** (strong / moderate / weak pairs); collinearity and leakage chips that filter the browser | `have`: relationship matrix, `collinearity`, `duplicate_columns`, `target_leakage` | Keep. |
| Correlation heatmap | Placeholder | `have`: relationship matrix | Build it (fulfils the old intent). |
| Column browser + associations table | Pick a column, then its associations with every other column (metric by type pair), filterable by type and strength | `have` | Keep. |
| Pair detail | Stats strip, warnings (VIF, leakage), **pair plot** (scatter, box or bar by type pair, from sampled raw rows) | `have`: `/snapshot/pair`, pair checks | Keep (Phase 6's `PairPlot` already draws it). |

### ML Readiness tab

| Element | Showed / carried by | Data now | Decision |
|---|---|---|---|
| Gate banner | Ready / needs work / not ready as a **coloured banner**, with total columns, ready to use and dimensions blocking | `have`: `ml_readiness` (`gate`, `complete`, `unassessed`) | Keep. |
| Dimension cards | Five **traffic-light cards**: Transformations needed, Encoding required, Missingness impact, Leakage risk, Unusable features | `have`: the same five dimensions and labels | Keep. |
| Dimension sections | Findings table (column, source, finding, level); an expanded row shows metrics and suggested actions, each with *what it does*, *trade-off*, *after this* and a reference link | `have`: findings and `guidance.actions` (method, condition, detail). The *what / trade-off / after / reference* text was 38 entries of static frontend content (`utils/actionMeta.js`) | Keep, and port the action text as a frontend dictionary keyed by method (U6b-08). |
| Transformation preview | Per skewed column, skewness before and after each candidate transform, the recommended one marked | `have`: `transformations` (`skewness_before`, candidates with `skewness_after`) | Keep. Chart: before/after bars. |
| Ready features | Chip grid | `have`: `clean_columns` | Keep. |

### Time Series tab

| Element | Showed / carried by | Data now | Decision |
|---|---|---|---|
| Disabled / no-index states | How to enable it, with the config to add | `have` (Phase 6 empty state) | Keep. |
| Header strip | Index column, frequency, value columns | `have`: time-series checks | Keep. |
| ACF / PACF | **Bar charts** with confidence bands per column, significant lags, a summary table | `have`: `autocorrelation` | Keep (Phase 6 draws these). |
| Stationarity | ADF and KPSS table with a note on reading the two together | `have`: `stationarity` | Keep. |
| STL decomposition | **Four-panel sparklines** per column with a strength table | `have`: `seasonal_decomposition` | Keep (Phase 6 draws these). |

### Phase 6 pages with no old counterpart

| Page or part | Closest old element | Decision |
|---|---|---|
| Findings explorer (filters, paging, detail, artifact viewer) | Data signals and the Quality sections | Keep as its own page, reached from the run page and from every finding count. |
| Run health (check statuses and issues) and policy verdict | The per-section "did not run" states | Keep; chart the status counts (U6-19 carry-in). |
| Compare, finding history and lineage | None | Keep as dataset-level pages outside the six run tabs. |
| Dataset settings and rerun | None | Keep as a dataset-level page. |
| Coverage (columns each check analysed) | None | Fold into Quality, beside the dimension coverage. |

## API needs

Both additive, both agreed, both built in U6b-02.

1. Per-column mode, mode share, skewness, min and max (and a small sketch, if wanted) on the
   `/runs/{id}/columns` summary, so the Overview table needs one request rather than one per column.
2. The latest run's five dimension levels on `DatasetSummary`, for dots on the dataset list.

## Questions the dashboard answers

Agreed with the user on 2026-10-29. A person answers each one from charts, without reading the JSON.

| Question | Where |
|---|---|
| Can I trust this data at a glance? | Health bar on every tab; Overview |
| What is wrong, worst first? | Overview's data signals; Quality; findings explorer |
| What does each column look like? | Distributions; the Overview's column table |
| How do the columns relate? | Relationships |
| Is it ready to model, and what preparation does it need? | ML Readiness |
| How does it behave over time? | Time Series (runs with a time index) |
| What changed since the last run? | Compare and finding history (new in Phase 6; the old dashboard could not answer it) |

## Structure

Agreed with the user on 2026-10-29: each run has the old six tabs, each a route, under a persistent
header and the five-dimension health bar. The findings explorer is its own page, reached from every
finding count. Compare, history, lineage, settings and rerun are dataset-level pages outside the
tabs. The Phase 6 line of links on the run page is replaced.

## Visual language

Decided in U6b-01 and signed off by the user with the shell screenshots (2026-10-30). Starting point,
agreed with the user on 2026-10-29: the old dashboard's look (dark slate theme by default, a
centred fixed-width layout, bordered cards with uppercase titles, traffic-light dots), changed only
where a chart needs it. The user's complaint was the content (tables and text), not the theme.

**Tokens** (`frontend/src/styles/tokens.css`; `tokens.test.ts` checks every pair).
- Surfaces and text keep the old slate palette in both themes, AA-checked at 4.5:1.
- Meaning by colour has its own tokens, checked at 3:1 against the card surface (the WCAG minimum
  for a graphical object): `--level-green|blue|amber|red` for the traffic lights and `--level-none`
  for an unassessed dimension; `--chart-primary`, `--chart-secondary` and `--chart-reference` for
  chart marks. Light-theme values are darker (for example green `#15803d`) so a dot holds 3:1 on white.
- ECharts cannot read CSS variables, so `chartTheme.ts` repeats the marks per theme (`MARKS`), and
  `tokens.test.ts` fails when the two drift.

**Severity and level vocabulary.**
- Four levels, always with words: green "All clear", blue "Minor", amber "Warning", red "Serious".
  Colour never carries the state alone: every dot has an accessible label and the words beside it.
- A fifth, grey state means "not assessed" (a dimension no check could read, such as leakage
  without a target). It is an empty ring, not a filled dot, so it cannot be read as green.
- Finding severity (info, notice, warn, error) keeps its badges; the levels are the dimensions'
  roll-up, not a second scale of findings.

**Categorical palette.** The ECharts theme palette in `chartTheme.ts` (blue, green, orange, red,
purple, emerald, amber, sky, fuchsia, slate) is the series order. One series uses
`--chart-primary`; a second uses `--chart-secondary`; a reference line (mean, threshold) is always
`--chart-reference`, dashed.

**Layout and density.** A centred page (max 1400 px). A run page is a stack: header (dataset ›
run date, then rows × columns, depth, "sampled" when it was), the health bar, the six tabs, then
the tab. Cards have a 1 px border, 10 px radius and 20 px padding; card titles are 13 px uppercase.
The health bar is one compact card of five dots, with a hover or focus tooltip giving the
dimension's definition and "N of M columns (x%)"; the dimensions a tab does not speak to are dimmed
to 45%.

**Chart components** (`frontend/src/components/charts/`). Each takes plain arrays, builds its
option in `options.ts` (unit-tested without a canvas) and follows the dashboard theme unless a
caller pins one.

| Component | Draws | Notes |
|---|---|---|
| `HistogramChart` | bars on the stored bin edges, value axis | optional dashed reference lines (mean, median, thresholds) |
| `BoxPlotChart` | one horizontal box per row | shared axis, so spreads compare |
| `TopValuesChart` | horizontal bars, order as given | category labels on the y axis |
| `HeatmapChart` | a labelled matrix | signed scales centre on zero (correlation); unsigned run min to max |
| `SparklineChart` | inline 80 × 20 SVG line | not ECharts: a table row needs no chart chunk |
| `StrengthBar` | stacked CSS bar of segments | segments sized by value, empty ones omitted |
| `TrafficDot` | filled dot, or a ring for "not assessed" | accessible label |
| `TrafficCard` | a card with dot, title and one summary line | grey state has a dashed border |
| `LevelBanner` | a full-width banner with a level stripe | headline and detail |

`DsbfChart` stays one lazy chunk. Its registrations are trimmed to the chart types and components
the dashboard draws with (bar, line, heatmap, scatter, boxplot, custom; grid, tooltip, data zoom,
mark line, visual map). Legend, toolbox, title and mark area are no longer registered. The visual
map was used by the relationships heatmap but never registered, so its colour scale was not
applied; it is now. The chunk went from 690,875 to 670,670 bytes.

**Routes.** `/runs/:runId` is a shell (header, health bar, tabs); each tab is a child route
(`''`, `quality`, `distributions`, `relationships`, `ml-readiness`, `time-series`), and the findings
explorer and one-column pages are children with no tab. `/runs/:runId/columns` redirects to
`distributions`. The ML readiness cards moved from the Quality page to their own tab; U6b-04 and
U6b-08 redesign both.
