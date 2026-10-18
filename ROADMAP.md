# DSBF Roadmap

This is the backlog of upcoming work. Each **phase** is a program: its capability, motivation,
boundaries, sequence and success criteria are locked at the top of the phase. Work inside a phase
is broken into **units**. A unit is one branch, one PR, one commit and one session. The workflow for running a unit is in [`CLAUDE.md`](CLAUDE.md), and the active unit (at
most one) lives in [`PLAN.md`](PLAN.md).

- **Design:** [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md)
- **Findings cited as F-xx / N-xx:** [`docs/REVIEW_FINDINGS.md`](docs/REVIEW_FINDINGS.md)
- **How a run works today, with measurements:** [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md)

**Status values:**
- `todo`: ready once its dependencies are `done`
- `active`: currently in `PLAN.md`
- `done`: merged; the entry carries notes and the PR number. When a phase's last unit closes,
  its unit entries are condensed into a short program summary (see `.claude/commands/closeout.md`).
- `placeholder`: a later phase, detailed by that phase's planning unit

**Next unit:** the first `todo` unit, in order, whose dependencies are all `done`.

Phases 0–3 are broken down in full. Phases 4–9 are placeholders. Each starts with a `Ux-00`
planning unit that turns the placeholders into full units, using what earlier phases taught.

## Dependency upgrades

Upgrades ride with the unit that owns the code they affect, so a behaviour change is reviewed
once, next to the code that changes it (proposed 2026-10-16, after U0-16). When a unit starts,
list the open Dependabot PRs and run `uv lock --upgrade --dry-run`. Take the upgrades in the area
the unit touches, verified like any other change: the fast gate, the golden suite when engine
output can move, and a real run. An upgrade that changes results is not taken silently: it either
becomes its own reviewed step (golden snapshots regenerated, with the differences explained in the
PR) or stays behind a cap, recorded in the table below with the unit that revisits it. When a
unit takes an upgrade, it raises the cap and deletes the row.

| Held back (`pyproject.toml` cap) | Why | Revisit in |
|---|---|---|
| `pandas <3` (PR #47, closed 2026-10-16) | 3.0 fails the 3.12 and 3.13 test jobs (string dtype, copy-on-write); the legacy tasks run on pandas until they migrate | U4-E (the base cap; pandas 3 frames already load as an input type, U2-01) |
| `statsmodels <0.15` | 0.15's `variance_inflation_factor` includes the intercept: VIFs drop and four columns on the `clean` archetype stop being flagged | U4-B (VIF redesign, F-32) |
| `scikit-learn <1.8` | 1.8 changes `GaussianMixture` log-likelihood on discrete columns (titanic `pclass` BIC 1016 → 1763), so `detect_bimodal_distribution` reports other BIC values and finding titles | U4-B (bimodality is an analysis check) |
| `scipy <1.18` (found in U1-05, no PR) | 1.18 makes `kstest(x, "norm", args=(mu, sigma))` raise `TypeError: ndtr() takes from 1 to 2 positional arguments but 3 were given`, so `run_ks` and two normality tests fail on a fresh `pip install`; DSBF imported scipy without declaring it, so only the lock kept it on 1.16.3 | the unit that migrates the normality tests (Phase 4; U4-00 assigns it), or sooner if a unit touches `normality_tests.py` |
| `plotly <7` (PR #62, closed 2026-10-16) | 7 removes the `mapbox` layout property that `custom_plotly_templates.py` sets, so it fails at import; the backend plotting that uses plotly is deleted anyway | U4-E (plotly is deleted; drop the cap and the ignore entry) |
| `vue-router <5` (PR #69, ignored from 2026-10-17) | 5 is a major bump of the router the current dashboard is built on; checking it needs the dashboard viewed in a browser, and the dashboard is rebuilt in Phase 6 | U6-00 |
| `plotly.js-dist-min <4` (PR #70, ignored from 2026-10-17) | 4 is a major bump of the library behind the legacy charts; checking it needs the dashboard viewed in a browser, and the charts are rebuilt in Phase 6 | U6-00 |

`.github/dependabot.yml` carries an `ignore` entry for each row, so Dependabot doesn't re-raise the held-back version. Removing a cap removes its entry.

---

## Phase 0: Stabilise and baseline

**Done (2026-10-16), 17 units.** Detailed unit entries were condensed here at close-out; the per-unit notes that later units depend on were carried into those units as "Note from U0-xx" lines.

- **Capability delivered:** a trustworthy, measured baseline on the target toolchain. CI runs lint (ruff from `uv.lock`), type check (pyrefly with a baseline), the fast gate and the full suite on Python 3.12 and 3.13, CodeQL, and the `golden` and `perf` workflows (nightly, on demand and on labelled PRs). Dependencies are managed with uv. Tests run hermetically from a temporary directory, every registered task completes on a hostile frame, and ADRs D1–D11 are accepted.
- **Success criteria, measured:**
  - Every Phase 0 unit is `done`: 17 of 17.
  - Fast gate (`unit and not slow`) 29 s; full suite 5 min 14 s (858 tests, U0-06). The `titanic` quickstart from a temporary directory went from 4 failed tasks to 0 (U0-07); `kitchen5k` from 3 failed tasks to 0, with 5 skipped by design (U0-09).
  - Golden baselines: ten snapshots (the 8 archetypes, `titanic`, `kitchen5k`), 60 successful and 5 skipped tasks each, 0 errors; two full runs matched byte for byte (U0-13). Re-checked on 2026-10-16 against the upgraded dependency set: 10 of 10 pass.
  - Performance baseline, taken on `ubuntu-latest` (4 CPUs): `titanic` 26 s, adversarial frame 16 s, `kitchen5k` 147 s, `kitchen50k` 357 s with a 2.08 GB peak, 0 failed tasks (U0-14).
  - pyrefly baseline: 895 existing errors across `dsbf/`, `tests/` and `docs/practice_run/` (U0-12); 0 new.
  - ADRs D1–D11 are `Accepted`: [0001](docs/adr/0001-library-first-layering.md), [0002](docs/adr/0002-findings-as-primary-output.md), [0003](docs/adr/0003-polars-arrow-data-representation.md), [0004](docs/adr/0004-ibis-warehouse-pushdown.md), [0005](docs/adr/0005-pydantic-contracts.md), [0006](docs/adr/0006-report-only-failure-semantics.md), [0007](docs/adr/0007-entry-point-plugins.md), [0008](docs/adr/0008-sqlalchemy-alembic-fsspec-storage.md), [0009](docs/adr/0009-single-distribution-with-extras.md), [0010](docs/adr/0010-check-naming.md), [0011](docs/adr/0011-toolchain.md).
  - Security alerts: 0 open code-scanning alerts and 0 open Dependabot alerts, re-listed on 2026-10-16 after #59 merged (alerts 3–10 closed when CodeQL re-scanned `main`).
- **Key decisions:** uv with the `uv_build` backend and `uv.lock` (ADR 0011); pyrefly as a pinned dev dependency, `default` preset, committed baseline that only shrinks; Python 3.14 out of the CI matrix (nine locked packages had no cp314 wheel); golden runs pin `PYTHONHASHSEED=0` and rank value counts with a stable sort (`ranked_value_counts`); `titanic` is vendored for the golden suite (seaborn-data has no license); perf numbers are compared only against the CI-hardware baseline; U0-16 closed the code-scanning alerts by returning generic API error text and logging the exception, and by giving the workflows `permissions: contents: read`.
- **Deviations from the phase block:**
  - The phase promised "no new features" and cheap safety fixes only. U0-09 also fixed `RichHandler(markup=True)`, which made seven tasks crash on column names such as `[/x]` (the largest crash cause); U0-13 fixed unstable tie ordering in three tasks; U0-06 added API tests.
  - "Dependabot backlog merged" took three units: U0-04, U0-16 and U0-17. U0-16 was widened on 2026-10-16 from the security alerts to the whole backlog, so Phase 1 starts with no open PR. It took #34, #38, #42, #43 and #45, and closed #46 (polars 2) and #47 (pandas 3) as deferred. Dependabot then raised four more PRs as soon as #59 merged, so U0-17 took #60 (pandas 2.3.3, polars 1.44.2; its pyrefly baseline lost one entry the new typings fixed), #61 (ipykernel 7) and #63 (rich 15), all on green CI with golden 10 of 10, and closed #62 (plotly 7, which fails at import and is deleted with the legacy plotting in U4-E).
  - The golden suite showed that two upgrades in #43 change results, so they stay behind caps: `statsmodels <0.15` (VIF includes the intercept) and `scikit-learn <1.8` (`GaussianMixture` log-likelihood on discrete columns). With `pandas <3`, `polars <2` and `plotly <7` they are listed under "Dependency upgrades" at the top of this file, each with the unit that revisits it, and mirrored by `ignore` entries in `.github/dependabot.yml`.
  - The README's examples were all rebuilt (U0-11) because most no longer ran.
- **PRs:** #29, #31, #32, #40, #41, #48, #50, #51, #52, #53, #54, #55, #56, #57, #58, #59 (U0-16, merged as `751ad9d`), and #64 (U0-17, merged as `de863ab`). Dependency PRs merged along the way: #23, #26, #24, #28, #27, #20, #22 and #36 (U0-04); #38, #42, #45, #34 and #43 (U0-16); #61, #63 and #60 (U0-17).

---

## Phase 1: Contracts

**Done (2026-10-17), 11 units.** Detailed unit entries were condensed here at close-out; the per-unit notes that later units depend on were carried into those units as "Note from U1-xx" lines.

- **Capability delivered:** the `src/` layout, a top-level `frontend/` on pnpm with ESLint, Prettier and Vitest in CI, runtime paths under `DSBF_HOME`, the optional-dependency extras, and the versioned result contracts in `dsbf.model` (subjects, evidence, findings, check results, run health, profiles, manifests, `Run`, policy rules) with a layered config loader (`dsbf.config`). `dsbf schema export` publishes `schema/run.schema.json` and `frontend/src/generated/run.d.ts`, and a CI `contracts` job fails on a stale export or on a schema change its version number does not announce.
- **Success criteria, measured:**
  - Every Phase 1 unit is `done`: 11 of 11.
  - A hand-built `Run` validates, round-trips through JSON, validates against the exported schema with `jsonschema` and compiles against the generated types under `tsc --strict` (U1-11). The `ColumnProfile` union narrows exhaustively.
  - Size: profile plus manifests for titanic are about 10 KB against a 200 KB budget (U1-08); a 61-check, 610-finding run is 338 KB, down from 644 KB before `Run.findings` became derived (U1-11).
  - Breaking-change gate, exercised on a real field removal: stale export exit 1, unannounced removal exit 1, with `SCHEMA_VERSION` raised exit 0 (U1-11).
  - Packaging: the wheel's file list equals `git ls-files src` (U1-01); the wheel with `[all]` in a clean venv passes the suite, 1006 tests (U1-05).
  - Paths: a run started from a scratch directory writes only under the home, and 357 of 357 figures are served by the API from another directory; golden 10 of 10 unchanged (U1-04).
  - Non-finite values: 5 of 5 on the adversarial frame are nulled and flagged, and all output is strict JSON, where the legacy `report.json` has 269 `NaN` and 19 `Infinity` (U1-06).
  - Frontend: `pnpm lint && pnpm build && pnpm test:run` pass with 31 unit tests and 20 lint warnings recorded as follow-ups (U1-03).
- **Key decisions:** pydantic v2 contracts that are frozen, reject unknown fields and use tuples ([ADR 0005](docs/adr/0005-pydantic-contracts.md)); undefined numbers become `None` plus a flag, never an error; `Finding.id` hashes check, subject and code; the home is `DSBF_HOME`, else the per-user data directory, chosen by the user on 2026-10-17 so one build serves a laptop and a cloud job ([ADR 0008](docs/adr/0008-sqlalchemy-alembic-fsspec-storage.md)); extras follow [ADR 0009](docs/adr/0009-single-distribution-with-extras.md) while the base install is unchanged; Prettier was chosen over ESLint stylistic rules (U1-03); the config loader is hand-written, because `pydantic-settings` cannot name the layer behind a bad key; the published schema describes a dumped run (serialization mode, defaults required) and treats any change to a type constraint as breaking, with the version rule in `CONTRIBUTING.md`; `Run.findings` is derived from `results`, not serialised.
- **Deviations from the phase block:**
  - The phase promised "packaging, paths and models only". U1-04 also made the metadata export opt-in (`dsbf tasks export`) and removed `record_run`; U1-05 declared scipy, which a clean install showed was imported but never declared, and capped it below 1.18; U1-09 added the config loader and U1-10 the policy schema, which the phase block did not list by name.
  - The policy matcher set grew beyond the REFACTOR_PLAN §7.2 example (all six comparison operators; `check`, `confidence`, `tag` and `column` matchers; `then` accepts a list) because the example cannot scope a rule to a check or column (U1-10).
  - `identifier` and `unknown` were added to the profile variants because the legacy `infer_types` emits them (U1-08). `pyarrow` stayed in the base install, as the design lists it (U1-05).
  - `Policy` is not in the published schema, because its loose spellings normalise on load; it ships with its loader (carried into U5-00). The dashboard was not viewed in a browser, and Dependabot's vue-router 5 and plotly.js 4 are held back (rows above, revisited in U6-00).
  - Phase 2 is already broken into full units, so it has no `U2-00`; U2-01 is next.
- **PRs:** #66 (U1-01, `bfc43e1`), #67 (U1-02, `b99633a`), #68 (U1-03, `c82fe45`), #71 (U1-04, `77d1992`), #72 (U1-05, `9f9d2c8`), #74 (U1-06, `837aea9`), #75 (U1-07, `07c2f53`), #76 (U1-08, `4b76948`), #77 (U1-09, `4af82d8`), #78 (U1-10, `3f39519`), and #79 (U1-11, `3b76aa7`).

---

## Phase 2: Data and profile

- **Capability:** Tier-1 sources (in-memory data and files), a `Dataset` with fingerprint and sampling provenance, and a single profiling pass that replaces per-task rescans.
- **Motivation:** "Measure once, decide many times" (REFACTOR_PLAN §2). It removes the repeated per-column computation and the per-task pandas copies, and fixes type inference cost, listwise deletion and histogram honesty at the source.
- **Boundaries:** No check API and no changes to legacy tasks. The profile is new code, compared against legacy outputs.
- **Sequence and dependencies:** Depends on Phase 1 contracts. Sources, then dataset, then type inference, then column statistics, then dataset-level profile, then parity, then the performance budget.
- **Success criteria:** Profile parity with legacy outputs is documented, and the profile pass runs in under 2 s at 52k × 60 on CI hardware.

### U2-01 · Source adapters for in-memory data and files (tier 1)
- **Status:** done (2026-10-17) · **Size:** M · **Depends on:** U1-08
- **Goal:** `dsbf.data.load(source)` accepts pandas, Polars or Arrow objects, or CSV/Parquet/Excel/JSON/NDJSON paths, and returns a Polars-backed `Dataset`. Unreadable input raises a typed error with a hint.
- **Done:** `dsbf.data` (`load`, `Dataset`, `SourceError` and its three subclasses) loads all of them; Polars is 2.0.0 (cap lifted, golden 10 of 10 unchanged); the Excel reader is `fastexcel` in a new `dsbf[excel]` extra; pandas 3 frames load, with a `pandas3` CI job running the adapter tests on them. Deviations: the `excel` extra and its ADR 0009 / REFACTOR_PLAN §3.1 rows were not in the unit; pyarrow stays in the base install (Polars' parquet, CSV and Excel readers don't need it, only pandas and Arrow input do; its home stays with U4-00). Dependabot #73 was left open (outside this unit's area). On 13 real datasets the loaded shape, column names and null counts match `pd.read_csv`; kitchen50k loads in 1.16 s (pandas 0.76 s).
- **PR:** #80 (`876a17c`)
- **Commit:** `feat(data): add tier-1 source adapters`

### U2-02 · Dataset fingerprint, sampling provenance and value cache
- **Status:** done (2026-10-17) · **Size:** M · **Depends on:** U2-01 · **Resolves:** F-09
- **Goal:** `Dataset.fingerprint`, `Dataset.sample(...)` with a recorded `SamplingInfo` (head, seeded random, stratified-or-error), and `Dataset.values(col, dropna=True)` over a byte-bounded cache.
- **Done:** `fingerprint` hashes each column (name, dtype, values) with `Series.hash`: 0.06 s on kitchen50k, 0.90 s at 1M × 60, with a pinned value so a Polars upgrade that changes the hash fails a test. A sample keeps its source's fingerprint (`Dataset.source_fingerprint`) and records `SamplingInfo`; stratified sampling is proportional by largest remainder, seeded, exact in row count, and null is its own group. `values` returns read-only arrays in an LRU bounded at 256 MiB by default. Deviations: the `SamplingInfo` contract now requires a seed for stratified samples as well as random ones (a validator change; the published schema is unchanged). The U2-01 pandas note is resolved by documentation, not a copy: `Dataset` states that its frame is read-only after load, because a copy would cost 0.13 s and 400 MB at 1M × 50 and undo the zero-copy load. The legacy `data_sampling` still has its F-09 crash; its removal is noted under U3-12. `load(dataset, name=...)` returns a copy that rebuilds its fingerprint and cache on demand.
- **PR:** #81 (`62efb95`)
- **Commit:** `feat(data): add dataset fingerprint, sampling provenance and value cache`

### U2-03 · Sample-based semantic type inference
- **Status:** done (2026-10-17) · **Size:** M · **Depends on:** U2-02 · **Resolves:** F-37
- **Goal:** Semantic types are inferred from a sample, with confirmation on the full column only where needed. The results agree with `infer_types` on the golden datasets, or the differences are documented.
- **Note from U2-01:** `load` leaves date-like CSV text as `String` (nothing parses dates at read time), so this unit is where it becomes a datetime. A CSV column with no values is `String` in Polars and `float64` in pandas (kitchen5k's `constant_null`, `mixed_null_types`, `empty_vs_nan`), so decide what an all-null column's semantic type is. Polars decodes a non-UTF-8 CSV lossily, so a UTF-16 file reads with garbled names and no error; consider flagging U+FFFD in names and values.
- **Context:** F-37. `_is_datetime_like` took 8.6 s of `infer_types`' 9.0 s on kitchen5k and grew 10× at 50k rows.
- **Acceptance:** Parity is checked against U0-13 baselines, and the differences are listed in the PR. Inference time is under 0.5 s on kitchen50k.
- **Done:** `dsbf.profile.infer_column_types(dataset)` returns a `ColumnType` per column (semantic type, confidence, sample rows, and the date format or epoch unit that U2-05 needs). Parsing and string-shape rules read a seeded sample of at most 1000 values, and a date column is confirmed on the whole column; counts, distinct values and the median come from the whole column. kitchen50k takes 0.28–0.30 s against 25.95 s for the legacy method. An all-null column is `unknown`, whatever dtype held it. On 11 datasets the types match the legacy ones except Polars `Boolean` columns (`boolean`, where legacy said `categorical`) and kitchen5k's `datetime_with_gaps` (years past 2262 overflow pandas, so legacy called it `id`; now `datetime`). Deviations: the legacy dateutil fallback is replaced by a fixed list of 19 formats, so a format outside it is not a date, and `%Y%m%d` is left out because eight-digit identifiers parse under it; the identifier rule (over 90% distinct) is kept for parity, so near-unique free text is `identifier`.
- **PR:** #82 (`83a62b1`)
- **Commit:** `feat(profile): add sample-based semantic type inference`

### U2-04 · Column profile I: counts and numeric statistics
- **Status:** done (2026-10-17) · **Size:** M · **Depends on:** U2-03 · **Resolves:** F-32 (per-column statistics)
- **Note from U0-09:** `detect_collinear_features` now only guards on complete-row count (infinities treated as missing). Per-column handling, dropping high-missingness columns first and recording `n_used` stay with the VIF redesign in Phase 4.
- **Goal:** One pass computes counts, nulls, distinct (exact or sketched), zeros, empty strings, non-finite values, and numeric min, max, mean, std, skew, kurtosis, MAD and quantiles, per column on that column's non-null values.
- **Acceptance:** Values match `summarize_numeric`, `summarize_nulls` and `summarize_unique` on the baselines within tolerance.
- **Done:** `dsbf.profile.profile_columns(dataset, types=None, *, quantiles=…)` computes every column's counts and statistics in one Polars `select` and returns a `ColumnProfile` per column: nulls, exact distinct, zeros, non-finite values, empty strings, boolean split, and for numeric columns min, max, mean, std, skewness, kurtosis and the seven legacy percentiles, all over that column's own finite non-null values (F-32). Null counts, distinct counts and the numeric statistics equal the legacy `summarize_nulls`, `summarize_unique` and `summarize_numeric` within `rel=1e-9` on all 10 golden datasets, with no accepted differences. kitchen50k takes 0.033 s (types supplied), 1M × 50 floats 2.0 s. Deviations: distinct counts are always exact (the sketched option is left to U2-08), and the contract gives categorical columns no numeric statistics, so integer codes the legacy task summarised have none here (see U2-07).
- **PR:** #83 (`c472927`)
- **Commit:** `feat(profile): add count and numeric column statistics`

### U2-05 · Column profile II: histograms, top-k, text and datetime
- **Status:** done (2026-10-17) · **Size:** M · **Depends on:** U2-04 · **Resolves:** F-12
- **Goal:**
  - equal-width histogram bins, with the rule recorded;
  - top-k values with counts;
  - string length and pattern statistics;
  - datetime range, inferred frequency, gaps and timezone.
- **Acceptance:** Histogram counts sum to `n_non_null`. The adversarial frame profiles without error. Values match `summarize_value_counts` and `categorical_length_stats`.
- **Done:** `profile_columns` now returns complete profiles, with `top_k` and `max_bins` options. Numeric columns get an equal-width histogram (Freedman–Diaconis, Sturges when the IQR is 0, clipped to 100 bins: 4 of 119 golden numeric columns hit the cap; whole-number widths for integers; counts sum to the finite non-null values and equal `np.histogram` on the same edges), so F-12 is fixed in the profile data. Categorical and text columns get top values (default 10), and string columns lengths, average word length and a new `TextShape` (character-class counts, pattern signatures). Datetime columns get range, time zone, frequency (fixed spans and calendar months, quarters, years) and gaps, matching `pd.infer_freq` on regular series. Top values and lengths match the legacy tasks on all 10 golden datasets with one listed difference (pandas cuts a field at a NUL byte). `SCHEMA_VERSION` 0.1.0 → 0.2.0 (additive: `TextShape`, `PatternCount`, `shape` on the two string profiles). kitchen50k takes 0.39 s with types supplied. Deviations: pattern and character-class statistics were added to the categorical profile as well as the text one; strings over 256 characters are never listed as top values (one such column made a 12.5 KB entry); text columns list only repeated values.
- **PR:** #84 (`2a15752`)
- **Commit:** `feat(profile): add histograms, top-k, text and datetime profiles`

### U2-06 · Dataset profile and stage inference
- **Status:** done (2026-10-17) · **Size:** S · **Depends on:** U2-05 · **Resolves:** F-08
- **Goal:** Shape, a hash-based duplicate-row count, a missingness co-occurrence summary, and stage inference from the profile with one vocabulary shared with check metadata.
- **Acceptance:** Stage inference gives the same answer for pandas and Polars input of the same data (the F-08 regression test).
- **Done:** `dsbf.profile.profile_dataset(dataset)` returns a full `DatasetProfile`: shape, memory, duplicate rows (64-bit row hash), rows with any null, and a `Missingness` summary (`NullPair` records of columns null together, ranked by Jaccard; the 100 columns with most nulls, 20 pairs). `infer_stage(profile, StageThresholds)` reads only the profile, so pandas and Polars input agree; `StageThresholds.from_config` reads `engine.stage_inference`; `Stage` (`raw`, `cleaned`, `modeling`, `report`, `any`) is the vocabulary check metadata will share. Duplicates, any-null rows and pair counts equal pandas on all 10 golden datasets; at 1M × 50 the dataset-level parts add about 0.26 s. `SCHEMA_VERSION` 0.2.0 → 0.3.0 (additive). Deviations: legacy `model_ready` is `modeling` and `exploratory` is `raw`; high cardinality counts categorical columns only (legacy counted all, so 9 of 10 golden datasets were `exploratory`); the legacy `eda/stage_inference.py` is untouched until the engine switches over.
- **PR:** #85 (`7fc2643`)
- **Commit:** `feat(profile): add dataset-level profile and stage inference`

### U2-07 · Profile parity report against legacy outputs
- **Status:** done (2026-10-18) · **Size:** M · **Depends on:** U2-06
- **Note from U2-06:** stage differences to list as accepted: legacy `model_ready` is `modeling`; legacy `exploratory` is `raw`; high cardinality counts categorical columns only; legacy ignored the configured thresholds (wrong config path) and compared Polars dtypes to strings, so pandas and Polars input gave different answers (`single_column`: `model_ready` vs `exploratory`).
- **Note from U2-05:** top values and lengths are compared in `tests/test_profile/test_column_parity.py` (`ACCEPTED_VALUES`, one entry). List these as accepted as well: the top-k default is 10 where the legacy task shows 5; the legacy table counts null as a value and the profile does not; text columns list only repeated values and no string over 256 characters is listed; `avg_word_length` is non-space characters per word where legacy divides all characters by words; pandas' CSV float parser can differ from Polars' in the last digit (compared to `rel=1e-12`); pandas cuts a field at a NUL byte (kitchen5k `control_character_text`). Frequency aliases name the spacing, not the anchor (`W` is any weekday, `MS`/`ME` any month), and jittered timestamps with no dominant step get no frequency or gap count.
- **Note from U2-04:** the legacy `summarize_numeric` summarises every numeric dtype, not only `continuous` columns, so it also covers integer codes the profile types `categorical` (titanic's `pclass`, `sibsp`, `parch`, `survived`); `CategoricalProfile` has no numeric statistics. List this as accepted and decide whether a categorical numeric column should carry min and max. Column-statistics parity is in `tests/test_profile/test_column_parity.py` with no accepted differences on the golden datasets.
- **Note from U2-03:** semantic-type differences to list as accepted, kept in `tests/test_profile/test_type_parity.py`: a Polars `Boolean` column is `boolean` (legacy `categorical`), and kitchen5k's `datetime_with_gaps` is `datetime` (legacy `identifier`, because pandas cannot hold years past 2262). Vocabulary: legacy `continuous` is `numeric`, `id` is `identifier`.
- **Note from U2-01:** load-level differences from `pd.read_csv` / `pd.read_excel` to list as accepted: an all-null column is `String`, not `float64`; Polars keeps an empty CSV header as `''` where pandas writes `Unnamed: 6`; the Excel reader names an empty header `__UNNAMED__6`; an Excel column of all-midnight datetimes reads as `Date`.
- **Goal:** An automated comparison between the new profile and the legacy `summarize_*` and `infer_types` outputs on every golden dataset, plus a committed `docs/parity/profile.md` listing accepted differences with reasons.
- **Acceptance:** The parity test passes with only the listed differences.
- **Done:** `tests/test_profile/accepted.py` is the one list of accepted differences (23 slugs in 7 areas), and `docs/parity/profile.md` explains each; `test_parity_report.py` fails if the two drift. New parity tests: dataset level (shape, duplicate rows, null-cell fraction against the golden snapshots), stage (each dataset's legacy and new stage pinned) and load (golden CSVs against `pd.read_csv`, plus the all-null, empty-header and Excel cases); the column and type parity tests now read their accepted cells from the registry. Deviations: the U2-06 stage note was too narrow: 9 of 10 golden datasets differ in name, 5 only by vocabulary (`exploratory` is `raw`, `model_ready` is `modeling`) and 4 (`clean`, `near_clean`, `severe_multicollinearity`, `titanic`) because high cardinality counts categorical columns only. Integer codes typed `categorical` (4 in titanic, 8 in kitchen5k) stay without min and max. Excel parity is pinned on `load` only, since `pd.read_excel` needs `openpyxl`. No change under `src/`.
- **Commit:** `test(profile): add legacy parity comparison and accepted differences`

### U2-08 · Profile performance budget
- **Status:** todo · **Size:** S · **Depends on:** U2-07, U0-14 · **Resolves:** F-30 (profiling part)
- **Note from U2-05:** with histograms, top values, string shape and datetime parts, `profile_columns` takes 0.39 s on kitchen50k (types supplied) and 3.3 s on 1M × 50 floats (2.0 s before), about 1.3 s of it histogram binning (`searchsorted` over 1M values per column); arithmetic binning with an edge fix-up is the option if that scenario needs the speed. `value_counts` (a `group_by` per categorical or text column) is the main cost on kitchen50k.
- **Note from U2-04:** `profile_columns` takes 0.033 s on kitchen50k and 2.0 s on 1M × 50 floats (0.77 s without the seven quantiles, which stay Polars expressions: 0.83 s against 1.11 s for `np.quantile` and 1.80 s sorting first). Distinct counts are always exact (`distinct_exact` is always `True`); add a sketched count above a row threshold only if the profile pass measures `n_unique` as a cost.
- **Note from U2-03:** `infer_column_types` takes 0.28–0.30 s on kitchen50k and computes each column's exact `n_unique` and null count, which the count pass of U2-04 computes again. Share them if the profile pass measures that as a cost.
- **Note from U1-04:** `record_run` no longer exists, so `tests/perf/baseline.json` still carries a `record_run` stage (0.0 s) in each scenario. Regenerate the baseline with the perf harness when this unit adds the profile pass, which drops the entry.
- **Note from U0-14:** `tests/perf/scenarios.py` has no 1M x 50 scenario, because today's engine would take hours on it. Add it here if the new profile pass can run it; the REFACTOR_PLAN §11 budget names it. The baseline shows where the legacy time goes: `fuzzy_duplicate_detection` (65% of titanic, 46% of kitchen5k) and `detect_stationarity` (40% of kitchen50k).
- **Note from U0-17:** the perf comparison is noisy at task level on shared CI runners. Two runs of the same tree (PR #64, 2026-10-16) flagged different single tasks: the first `generate_dataset_summary_plots` (+52% on kitchen5k, +28% on kitchen50k, about 1.1 s each, just over the 1 s floor), the second `detect_stationarity` on kitchen50k (89 s → 121 s, +36%), and neither flagged a stage or a total. A local A/B of the plotting task under the old and new dependencies showed no difference (median 1.11 s vs 1.03 s). Before the budget is enforced as a gate, make the check robust: compare the median of repeated runs for the kitchen scenarios, or give long tasks a wider tolerance, so a clean tree doesn't fail by chance.
- **Goal:** Add the profile pass to the perf harness, with the budget enforced.
- **Acceptance:** Profile pass under 2 s at kitchen50k (52.5k × 60) on CI hardware, and under 2.5× peak memory over the in-memory data size. Or record the actual numbers and amend the budget with reasons.
- **Commit:** `test(perf): enforce profiling pass budget`

---

## Phase 3: Engine and check API

- **Capability:** The check API, registry and packs, the planner, an isolated, parallel and cached executor, library-friendly logging, a legacy adapter running every existing task, and the public `dsbf.profile()` entry point.
- **Motivation:** "Choose your own analysis" and report-only failure handling (REFACTOR_PLAN §1, R1 and R2) live in the engine. The adapter lets checks migrate one at a time against a running system.
- **Boundaries:** No check migrations (Phase 4), and no storage, API or dashboard changes (Phase 6).
- **Sequence and dependencies:** Depends on Phase 2. Interface, registry, planner, executor (isolation, then timeouts and events, then parallelism), cache, logging, adapter, entry point.
- **Success criteria:** Golden runs reproduce through the adapter. A forced exception in any check still yields a complete `Run` with a note. Changing one parameter re-executes only the affected checks.

### U3-01 · Check interface and check context
- **Status:** todo · **Size:** M · **Depends on:** U2-08, U1-09
- **Goal:** The `@check` decorator and metadata (id, version, kind, params, `applies_to`, `requires`, `after`, cost, extras, packs, dimensions). `CheckContext` provides filtered `columns`, `values()`, `profile`, `finding()`, `note()` and `output()`.
- **Note from U1-07:** `CheckResult.output` is an open `dict[str, Any]` (made JSON-safe, undefined numbers nulled and named in `non_finite_outputs`). Typed per-check output models belong here.
- **Assumptions to confirm:** The decorator vs. class-based API (REFACTOR_PLAN §6.2 shows the decorator); naming per ADR D10.
- **Acceptance:** Three toy checks (rule, analysis, synthesis) run in a test harness and produce valid `CheckResult`s. A check never sees columns outside `applies_to`.
- **Commit:** `feat(engine): add check interface and check context`

### U3-02 · Registry, packs and plugin discovery
- **Status:** todo · **Size:** M · **Depends on:** U3-01 · **Resolves:** F-23
- **Goal:** An explicit `Registry` object (no import side effects), the `dsbf.checks` and `dsbf.packs` entry-point groups, local plugin directories imported under unique module names, and reload safety.
- **Acceptance:**
  - A test plugin package installed in a venv is discovered.
  - Two plugin dirs that each contain a `utils.py` coexist.
  - Loading twice in one process is a no-op.
- **Commit:** `feat(engine): add registry, packs and entry-point plugin discovery`

### U3-03 · Planner I: selection and dependency resolution
- **Status:** todo · **Size:** M · **Depends on:** U3-02 · **Resolves:** F-21
- **Note from U2-06:** `infer_stage` rates data by its mean column null ratio, so titanic is `cleaned` although `cabin` is 77% null and 709 of 891 rows have a null. When the inferred stage drives selection here, decide whether a per-column rule (the worst column, or the share of columns over a null limit) is needed.
- **Goal:** Select checks by packs, include/exclude lists, depth presets, cost budget, dimension and tag. Resolve hard requirements (auto-included and marked) and soft `after` ordering. Report cycles as plan errors.
- **Acceptance:** Table-driven tests cover selection. A soft-after dependency that errors doesn't skip its dependent (the F-21 scenario).
- **Commit:** `feat(engine): add plan selection and dependency resolution`

### U3-04 · Planner II: validation, expected statuses and `dsbf plan`
- **Status:** todo · **Size:** M · **Depends on:** U3-03 · **Resolves:** F-07, F-22
- **Goal:**
  - Validate parameters against each check's model, with precise errors.
  - Annotate the plan with `will_run`, `not_applicable` or `unavailable` (missing extra), with reasons.
  - Print a readable plan via a new `dsbf plan` CLI command.
- **Acceptance:**
  - A misspelled parameter is an error naming the check, key and allowed keys (the F-07 class).
  - A missing extra shows as `unavailable` with an install hint.
  - `dsbf plan data.csv` output has a snapshot test.
- **Commit:** `feat(engine): validate plans and add dsbf plan`

### U3-05 · Executor I: isolation, statuses and notes
- **Status:** todo · **Size:** M · **Depends on:** U3-04 · **Resolves:** F-06, F-20 (engine side)
- **Goal:** Sequential execution where every exception becomes `error` with a note (type, message, hint, traceback artifact), hard dependents become `skipped` with the blocker named, and the run always completes with `RunHealth`.
- **Acceptance:** A forced exception in each toy check yields a complete `Run`. The health summary lists the reason. A check returning a failed result is never counted as ok (the F-06 regression test).
- **Commit:** `feat(engine): execute checks in isolation with explicit statuses`

### U3-06 · Executor II: timeouts, resource accounting and events
- **Status:** todo · **Size:** M · **Depends on:** U3-05 · **Resolves:** F-24
- **Goal:** Per-check timeouts (→ `timeout` plus a note); an optional run budget; wall, CPU and sampled peak memory per check, labelled approximate; and an event hook API (`check_started`, `check_finished`, `note`).
- **Assumptions to confirm:** How timeouts work for thread-based checks (cooperative cancellation vs. process isolation for heavy checks).
- **Acceptance:** A sleeping toy check times out without stalling the run. Events arrive in order. Memory figures are within reason of `/usr/bin/time` on a perf dataset.
- **Commit:** `feat(engine): add timeouts, resource accounting and execution events`

### U3-07 · Executor III: parallel execution
- **Status:** todo · **Size:** M · **Depends on:** U3-06
- **Goal:** Independent checks at each DAG level run concurrently (threads for compiled-code checks, a process pool for pure-Python ones). Results are deterministic regardless of worker count.
- **Acceptance:** The same `Run` (excluding timing) with 1 worker and N workers. A speed-up is measured on a perf dataset.
- **Commit:** `feat(engine): run independent checks in parallel`

### U3-08 · Result cache
- **Status:** todo · **Size:** M · **Depends on:** U3-07
- **Goal:** Results cached by `(fingerprint, check id, check version, params hash, sampling)` on local disk under `DSBF_HOME`, with invalidation of dependents.
- **Acceptance:** Changing one parameter re-executes only that check and its dependents (a test asserts the execution counts). Bumping a check version invalidates its entries.
- **Commit:** `feat(engine): cache check results by data fingerprint and parameters`

### U3-09 · Library-friendly logging
- **Status:** todo · **Size:** S · **Depends on:** U3-05 · **Resolves:** F-19
- **Goal:**
  - a `dsbf` logger hierarchy with no global logging changes;
  - user data never interpreted as markup;
  - an `error` level;
  - the CLI attaches handlers;
  - per-run file logs, with handlers closed afterwards.
- **Acceptance:** Importing `dsbf` leaves `logging.getLoggerClass()` unchanged. Logging a column named `[/x]` produces no logging error. Task-level messages appear in the run log file.
- **Commit:** `refactor(logging): make logging library-friendly and markup-safe`

### U3-10 · Legacy task adapter
- **Status:** todo · **Size:** M · **Depends on:** U3-08, U3-09
- **Note from U1-07:** blurbs carry no `code`, `dimension`, `confidence` or (always) `n_used`; the adapter supplies them. A `code` must be unique within a check and subject and must not depend on blurb order (the contract test's `<phase>_<index>` shifts when a task reorders blurbs). Map `good` to `Severity.INFO` plus the tag `good`, `__dataset__` to `Subject.dataset()`, and legacy `column`/`columns` on an action to `Action.columns`. `tests/test_model/test_legacy_guidance.py` holds the reference mapping.
- **Goal:** `LegacyTaskCheck` wraps any `BaseTask`. It provides a legacy `AnalysisContext` shim (data, config, `semantic_types` from the new profile) and maps `TaskResult` statuses onto the new statuses.
- **Assumptions to confirm:** Which `context.metadata` keys legacy tasks read from each other (grep `get_metadata`, `context.results`).
- **Acceptance:** Five representative legacy tasks (a rule-like task, a pandas-heavy one, a plotting one, a synthesis scorer, and a time-series one) run through the new engine with their outputs unchanged.
- **Commit:** `feat(engine): add legacy task adapter`

### U3-11 · All legacy tasks through the new engine with findings
- **Status:** todo · **Size:** M · **Depends on:** U3-10
- **Note from U1-06:** `tests/perf/baseline.json` still records the HTTP 500s on the adversarial run (`/associations` and `/sample` emit `NaN`; the U0-14 note is kept under U1-06). The legacy writer and API don't use `dsbf.model` yet. Once legacy results are routed through `Evidence` (nulled and flagged), or the API stops emitting non-finite floats, regenerate the baseline with the perf harness so the benchmark guards the 200s.
- **Note from U1-07:** use the mapping and the rules in the U3-10 note; the tests there convert a real titanic sample without loss.
- **Goal:** Every registered legacy task runs on the new engine. Guidance blurbs convert into `Finding`s with evidence where it can be derived.
- **Acceptance:** Golden runs (U0-13) reproduce through the adapter, with differences listed in the PR. The adversarial frame produces a complete `Run` with no unhandled errors.
- **Commit:** `feat(engine): run all legacy tasks through the new engine`

### U3-12 · `dsbf.profile()` entry point and run health output
- **Status:** todo · **Size:** M · **Depends on:** U3-11 · **Resolves:** F-16, F-17, F-18, F-34
- **Note from U2-06:** call `dsbf.profile.profile_dataset` and `infer_stage` (with `StageThresholds.from_config`) from the entry point, and remove the legacy `eda/stage_inference.py` and its call in `ProfileEngine` when the engine path is replaced.
- **Note from U2-03:** the identifier and period-label rules use the share of distinct values, so a row sample can type a column differently from the full data (kitchen50k's `high_cardinality_random` is `categorical` whole and `identifier` in a 2000-row random sample). Decide whether `sample=…` types the full dataset first and profiles the sample, or types the sample and records it in `semantic_sample_rows`.
- **Note from U2-02:** `Dataset.sample(rows, strategy=, seed=, stratify_by=)` is the sampling to build `sample=…` on. The legacy `data_sampling` still has the F-09 crash and the `stratified` alias for head; remove it when the legacy engine path is replaced. A stratified or random sample records its seed in `SamplingInfo`.
- **Note from U2-01:** the legacy `dsbf profile` still fails on a parquet with a `datetime64` column (`TypeError: Object of type Timestamp is not JSON serializable`, in the legacy report writer). It goes away when results are written from the `Run` contract; add a datetime64 parquet to this unit's end-to-end test to confirm.
- **Note from U1-01:** `dsbf version` runs `git describe --tags` (`utils/versioning.py`), so outside a checkout it prints `dev` plus a `fatal: not a git repository` line, even from an installed wheel. Read the version from `importlib.metadata`.
- **Goal:** The public `dsbf.profile(source, packs=…, checks=…, target=…, time_index=…, sample=…)` returns an in-memory `Run` with no filesystem writes. Version comes from `importlib.metadata`. `run.health.summary()` matches REFACTOR_PLAN §7.1.
- **Acceptance:**
  - A test asserts no files are created during `dsbf.profile()`.
  - The README library example works.
  - Phase 3 exit checks pass: golden via the adapter, forced exception gives a complete `Run`, a parameter change re-executes only the affected checks.
- **Commit:** `feat: add dsbf.profile() library entry point`
- **Note from U1-09:** `dsbf.config.load_config(model, ...)` is generic over any model. This unit defines the root config model (engine, limits, `profiling_depth`, `backend`, `sample_strategy` as enums, so F-22's third bullet is covered) and builds `profile(...)` keyword arguments into the `overrides` layer. The caller's config must not be mutated (F-22's second bullet): `load_config` copies its inputs.

---

## Phase 4: Migrate checks (placeholder)

- **U4-00 · Plan Phase 4** (`todo`, depends on U3-12): break down the batches below into full units using Phase 3 learnings and golden-run evidence. Update REFACTOR_PLAN if needed. Carry in from U0-15 (ADR 0009): shrink the base install to the core dependency set as the legacy tasks migrate, moving the rest behind extras and deleting dependencies nothing imports (for example matplotlib, plotly, seaborn, networkx, graphviz, pydot and missingno, to be verified by grep). Carry in from U1-05: (1) each extra needs a test that the core runs without it and reports `unavailable` (ADR 0009), added by the unit that removes its package from `dependencies`; (2) CI runs the extras as an install matrix, one leg per extra, once the base shrinks and an extra's absence is observable; (3) decide pyarrow's home: the design keeps it in the base, while the U0-04 note wanted it in the extra for parquet input (pandas is its only parquet user today); (4) declare `numpy` (imported in 45 places, never declared) with a tested range when the core dependency set is fixed, and declare each other directly imported package at the same time; (5) the `scipy <1.18` cap: change the `kstest` call in `normality_tests.py` when migrating it, then lift the cap.
- U4-A · Rule checks onto the profile, about 20 tasks (`placeholder`). From U2-07: a categorical column of integer codes has no min and max, where the legacy `summarize_numeric` summarised it; the check that replaces it decides whether to read such codes as numeric. No golden dataset gives `summarize_text_fields` output, so `avg_word_length` was never compared to the legacy task; add a text-heavy golden dataset when the text checks move. From U2-05: a categorical column with mostly unique strings lists first-seen one-off values as top values (text columns skip values seen once); decide with the checks that consume top values whether categorical should also require a repeat. From U2-03: `load` decodes a non-UTF-8 CSV lossily with no error (a UTF-16 file reads with U+FFFD and NUL characters in names and values), so add a rule check that flags U+FFFD and NUL in column names and values; a column that mixes date formats (kitchen5k's `inconsistent_datetime`) is typed `categorical`, so a check could report the mix. From U0-03: the migrated guards only check that some column matches; tasks still loop over every numeric or object dtype column, not `matched_cols`, so they also analyse columns whose semantic type was excluded. And a migrated task run without `infer_types` now skips instead of analysing all numeric columns. Decide the intended behaviour per check. From U0-08: the time-series test configs still carry an inert nested `time_series.tasks.<name>` block that no task reads (tasks read `tasks.<task name>`); drop it when those tests migrate. From U0-07: `get_recommendation_tip` (`dsbf/utils/reco_engine.py`) still runs `eval` on the YAML `condition` strings (F-31); replace it with a declarative condition format (`{"field": ..., "op": ">", "value": ...}`) when the 14 tasks that call it migrate
- U4-B · Analysis checks with statistical fixes: F-11, F-13, confidence intervals, bias-corrected Cramér's V (`placeholder`). From U0-09: `summarize_modes` iterates all columns rather than `matched_cols`, and the leakage correlation scan uses `select_dtypes` (F-11); the collinear check needs per-column VIF handling and `n_used` (F-32). From U0-16: when the VIF check is redesigned, lift the `statsmodels <0.15` cap and decide the intended VIF definition (centred, with an intercept, as 0.15 computes it, or the uncentred one the baselines record); when bimodality is migrated, lift `scikit-learn <1.8` and decide how the BIC comparison is reported (the absolute BIC and `delta / abs(bic1)` both depend on the log-likelihood convention; the delta itself did not move). Regenerate those golden snapshots in the same PR, with the differences explained
- U4-C · Heavy checks redesigned with budgets: fuzzy duplicates (F-35), time series (F-39), drift against reference profiles (`placeholder`). From U0-13: `fuzzy_duplicate_detection` builds candidate pairs from a token index over sets of strings and stops at `max_candidates`, so which pairs it keeps (and the count) follows `PYTHONHASHSEED`; make the order deterministic, then the golden runner no longer needs the pinned seed. On kitchen5k, `decompose_time_series` cannot infer a seasonal period (frequency `None` for a daily index with repeated dates), and the golden baseline records that.
- U4-D · Synthesis: quality dimensions, ML readiness, target-based leakage (F-26) (`placeholder`). From U2-03: the identifier rule treats any string column over 90% distinct as `identifier`, so near-unique free text (kitchen50k's `long_text`) is not `text`; revisit the identifier-before-text order with the checks that consume the type.
- U4-E · Remove the legacy adapter, backend plotting and plotting dependencies (N-A2, F-38) (`placeholder`). From U0-16: once no legacy task needs pandas, lift the `pandas <3` cap (PR #47 was closed as deferred; the golden suite and the pandas extra's tests must pass on 3.x)

## Phase 5: Policy, sinks, CLI v2. First release for embedded use (placeholder)

- **U5-00 · Plan Phase 5** (`todo`, depends on the Phase 4 units). Carry in from U1-09: CLI v2 builds its `cli` layer for `dsbf.config.load_config` and needs a `--set a.b=c` parser (U1-09 left it out); `dsbf run -c` still uses the YAML as the whole config (F-22, first bullet) and goes away with the legacy CLI. Carry in from U0-13 and U1-04: `tests/validation/run_validation.py` without `--report` looks for the newest `report.json` under `dsbf/outputs/`, which DSBF stopped writing to in U1-04 (runs now go under `$DSBF_HOME/outputs/`), so it finds nothing new, and used to take whatever dataset's report was newest (it checked `near_clean` against a stale 9,217-row run on 2026-10-16), and the README keeps a warning about it; make it profile into its own output directory, as the golden runner does, then drop the warning. Its `clean` checklist also says 10 columns where the dataset has 9. Carry in from U0-11: (1) `dsbf profile` defaults `--depth` to `standard`, while `default_config.yaml`, `dsbf quickstart` and `dsbf run` use `full`: pick one default for CLI v2. (2) The default config's `message_verbosity` comment lists `quiet | info | debug`, but the logger also accepts `warn` and `stage`. (3) Dead config keys `metadata.output_format`, `engine.severity_thresholds` and `metadata.layout_name` (F-28) go with the config redesign. Carry in from U1-04: a configured `output_dir` is used as given, so a relative one still stores relative figure paths (only the default location is absolute); resolve it to absolute in the config redesign. Carry in from U1-10: (1) the policy loader (`--policy`) turns the validation failures of `yaml.safe_load` + `Policy.model_validate` into a `ConfigError`-style report with the rule name; (2) the evaluator decides what `new_since: previous_run` does without a previous run; (3) a range on one setting (`>0.1` and `<0.5`) needs a list of comparisons per `metric` key, `severity` or `confidence`: add it if users ask. Carry in from U1-11: publish the `Policy` schema (`schema/policy.schema.json`, with the loose spellings documented) together with its loader, because the model's own schema describes only the normalised form.
- Placeholders: policy evaluator (report-only default) · JSON/NDJSON sinks · Parquet sinks with a consumer schema guide · static HTML report · CLI v2 (`profile`, `check`, `plan`, `schema export`, run health summary) · pipeline and consumer docs · install smoke test from a temp dir · release `1.0.0a1`

## Phase 6: Store, service and dashboard v2. Standalone local (placeholder)

- **U6-00 · Plan Phase 6** (`todo`, depends on the Phase 5 units). Carry in from U1-11: (1) the dashboard is JavaScript, so it can use `frontend/src/generated/run.d.ts` only through JSDoc `import()` types or after a move to TypeScript; (2) a TypeScript consumer needs its own rank table for `Severity` and `Confidence`, which the schema cannot carry; (3) revisit the `vue-router <5` and `plotly.js-dist-min <4` caps in the held-back table.
- Placeholders: run repository on SQLite with Alembic · artifact store and per-run data snapshot · `dsbf runs …` · API v2 · local job runner · generated TS client · dashboard shell and run health · findings explorer · column view (reuse `wip/echarts-charts` components with real histogram bins; the profile now carries equal-width bins from U2-05, while the dashboard still draws the percentile-binned histogram of F-12) (U0-16 saw the light theme draw chart titles and card headings such as "Value Counts" almost white on a light background; same on echarts 5) · relationships · quality · ML readiness · time series · run compare and history · settings → rerun · frontend component and end-to-end tests (F-29) · clear the 20 frontend lint warnings from U1-03 and raise `no-unused-vars`, `vue/no-side-effects-in-computed-properties` and `vue/require-default-prop` to `error` (16 dead variables in 12 files, `acfSelectedColData` in `TimeSeriesTab.vue` assigns inside a `computed`, route props in `RunDetailView.vue` and `RunListView.vue`); revisit `vue/no-v-html` if another source appears · code-split the dashboard bundle (over 500 kB, Plotly and ECharts load eagerly) · bundle the dashboard into the wheel with `dsbf dashboard` · remove API v1 and the old store (F-15, F-25) · release `1.0.0`

## Phase 7: Scale and sources (placeholder)

- **U7-00 · Plan Phase 7**. Carry in from U0-15: re-check ADR 0004, that the profile interface can be expressed as an Ibis aggregation (with the approximate-statistics caveats), before committing to the pushdown design; amend or supersede the ADR if it can't.
- **Carry in from U2-01:** `dsbf.data.load` reads one local file, the first Excel sheet only, and rejects directories, globs and URLs (`UnsupportedSourceError`). Sheet choice, globs and `s3://`-style paths belong to T2.
- Placeholders: larger-than-memory files and object storage (T2) · Ibis sources with DuckDB/Postgres, then Snowflake/BigQuery (T3) · recurring datasets, history and profile-based drift (T4) · scheduling recipes

## Phase 8: Team and cloud (placeholder)

- **U8-00 · Plan Phase 8**
- Placeholders: Postgres repository · object-storage artifacts · auth (token, OIDC) · job queue · Docker image and Helm chart · GitHub Action · Airflow, Prefect and Dagster adapters · dbt recipe · notifiers and OpenLineage · MCP server

## Phase 9: Modalities and workflow phases (placeholder)

- **U9-00 · Plan Phase 9**
- Placeholders: text-corpus modality · domain packs · modelling-prep exports (ML guidance actions as sklearn pipelines) · modelling diagnostics
