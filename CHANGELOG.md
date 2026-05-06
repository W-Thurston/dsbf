# Changelog

All notable changes to DSBF are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

<a name="v0.20.0"></a>
## [v0.20.0](https://github.com/W-Thurston/dsbf/compare/v0.19.0...v0.20.0) — Current

This release represents the completion of the core EDA phase of DSBF — the full
stack from task engine through interactive dashboard is functional, validated, and
ready for real-world use.

### Added

**Data health framework (Quality tab)**
- Five independently-scored quality dimensions: Completeness, Validity, Usability,
  Redundancy, Leakage — replacing the original single numeric DQ score
- Four-state traffic-light system (green / blue / amber / red) where color maps to
  both severity and proportion: `info`-only findings below 10% of columns produce
  blue; any `warn`-or-above finding produces minimum amber regardless of proportion
- `DataHealthBar` component showing all five dimension dots in the Overview tab
- Trust banner with plain-English summary that updates with dataset state

**ML Readiness tab**
- Five preparation-action dimensions: Transformations Needed, Encoding Required,
  Missingness Impact, Leakage Risk, Unusable Features
- Severity-based gate banner (Ready / Needs Work / Not Ready) driven by
  error/warn finding counts, not proportion
- Column Status section (Needs Attention / Notes Available / Ready As-Is buckets)
- `model_sensitivity` tag strip on expanded findings (affected vs unaffected model
  families)

**Task implementations (Waves 2–4)**
- Unified outlier detection (`detect_outliers.py`) with Isolation Forest support
  via `__dataset__` sentinel key
- Missingness mechanism analysis with deliberate epistemic humility — confidence
  capped at "moderate", `consistent_with` language, never claims to confirm MNAR
- Time series suite: `compute_acf_pacf`, `detect_stationarity`,
  `decompose_time_series`; shared `_time_series_utils.py`; defaults to disabled
  with plain-English explanation
- `get_shared_param()` on `BaseTask` for correct access to shared config blocks
- `add_guidance()` gains optional `model_sensitivity` parameter

**Validation suite**
Eight purpose-built dataset archetypes with full assertion coverage:
`clean`, `tiny`, `near_clean`, `all_categorical`, `high_missingness`,
`severe_multicollinearity`, `single_column`, `wide` (100 columns)

The `severe_multicollinearity` dataset specifically targets four VIF failure modes
including the scale-mismatch regression test for the `add_constant` fix.

**Dashboard components**
- `MissingnessMechanismCard` — dedicated expandable card per column
- `MlReadinessTab` — full tab with gate banner, dimension cards, collapsible
  sortable findings, inline row expansion, column status section
- `DataHealthBar`, `QualityTab` — five-dimension traffic light replacing the
  original single-score display

### Fixed

- **VIF spurious inflation (critical):** `detect_collinear_features` was calling
  `variance_inflation_factor` without an intercept term. Features with nonzero
  means produced VIF of 10–35 regardless of actual collinearity. Fixed by adding
  `add_constant(numeric_df, has_constant="add")` before computation.
- **Data corruption bug:** `cli.py` quickstart was not nulling `dataset_path`,
  causing Titanic dataset paths to persist into subsequent NFL dataset runs.
  `writer.py` now determines `is_builtin` from `raw_path is None AND
  dataset_source in (seaborn, sklearn, openml)`. `_upsert_dataset` only updates
  `source_path` when the incoming value is non-null.
- **Semantic type filtering:** `detect_high_cardinality` and `detect_id_columns`
  were iterating `df.columns` instead of `matched_cols`, bypassing the semantic
  type filter entirely. Both now operate on `matched_cols` exclusively.
- **`_level()` severity floor:** The quality scorer's `_level()` function only
  prevented `error` findings from producing a green result. A single `warn`-level
  finding on a 100-column dataset would silently produce green (1% proportion,
  below the 5% amber threshold). Floor now covers `warn`-and-above.

### Changed

- `data_quality_scorer._level()` rewritten with four-state logic (see above)
- Associations: `kendalls_tau` absorbed into `compute_pairwise_associations` with
  `method="auto"` routing (Kendall's τ for n<30, Pearson for n≥30)
- Outlier detection consolidated — `outlier_detection_mad` deprecated in
  `task_metadata.yaml`; unified task handles all methods including Isolation Forest

---

<a name="v0.19.0"></a>
## [v0.19.0](https://github.com/W-Thurston/dsbf/compare/v0.18.0...v0.19.0) (2025-07-11)

<a name="v0.18.0"></a>
## [v0.18.0](https://github.com/W-Thurston/dsbf/compare/v0.17.0...v0.18.0) (2025-07-09)

<a name="v0.17.0"></a>
## [v0.17.0](https://github.com/W-Thurston/dsbf/compare/v0.16.0...v0.17.0) (2025-07-08)

<a name="v0.16.0"></a>
## [v0.16.0](https://github.com/W-Thurston/dsbf/compare/v0.15.0...v0.16.0) (2025-07-06)

### Added
- Complete milestone: plot integration, task validation, test stability

<a name="v0.15.0"></a>
## [v0.15.0](https://github.com/W-Thurston/dsbf/compare/v0.14.0...v0.15.0) (2025-07-04)

### Fixed
- Import statement typo

<a name="v0.14.0"></a>
## [v0.14.0](https://github.com/W-Thurston/dsbf/compare/v0.13.0...v0.14.0) (2025-07-04)

<a name="v0.13.0"></a>
## [v0.13.0](https://github.com/W-Thurston/dsbf/compare/v0.12.0...v0.13.0) (2025-07-04)

<a name="v0.12.0"></a>
## [v0.12.0](https://github.com/W-Thurston/dsbf/compare/v0.11.0...v0.12.0) (2025-07-04)

<a name="v0.11.0"></a>
## [v0.11.0](https://github.com/W-Thurston/dsbf/compare/v0.10.0...v0.11.0) (2025-07-04)

### Added
- Metadata filtering, task registry audit, and DAG config control

<a name="v0.10.0"></a>
## [v0.10.0](https://github.com/W-Thurston/dsbf/compare/v0.9.0...v0.10.0) (2025-07-04)

### Added
- Plugin architecture, task metadata, validation

<a name="v0.9.0"></a>
## [v0.9.0](https://github.com/W-Thurston/dsbf/compare/v0.8.0...v0.9.0) (2025-07-04)

<a name="v0.8.0"></a>
## [v0.8.0](https://github.com/W-Thurston/dsbf/compare/v0.7.0...v0.8.0) (2025-07-04)

<a name="v0.7.0"></a>
## [v0.7.0](https://github.com/W-Thurston/dsbf/compare/v0.6.0...v0.7.0) (2025-07-04)

### Added
- Centralized reliability flag logic; key tasks refactored to use it (Milestone 5)

<a name="v0.6.0"></a>
## [v0.6.0](https://github.com/W-Thurston/dsbf/compare/v0.5.0...v0.6.0) (2025-07-04)

<a name="v0.5.0"></a>
## [v0.5.0](https://github.com/W-Thurston/dsbf/compare/v0.4.0...v0.5.0) (2025-07-04)

### Added
- `detect_target_drift` task
- Audit and standardization of all EDA tasks and tests; feature drift detection

<a name="v0.4.0"></a>
## [v0.4.0](https://github.com/W-Thurston/dsbf/compare/v0.3.0...v0.4.0) (2025-07-04)

### Fixed
- Expanded task registry; standardized tasks; profiling depth moved from hardcoded
  `DEPTH_LEVELS` to per-task registry declarations; execution graph updated accordingly
- `ExecutionGraph` and `ProfileEngine` cleanup

### Chore
- Updated `ci.yml` and `.pre-commit-config.yaml`

<a name="v0.3.0"></a>
## [v0.3.0](https://github.com/W-Thurston/dsbf/compare/v0.2.0...v0.3.0) (2025-06-26)

### Changed
- Converted all EDA tasks to class-based implementation

<a name="v0.2.0"></a>
## [v0.2.0](https://github.com/W-Thurston/dsbf/compare/v0.1.0...v0.2.0) (2025-06-26)

### Added
- DAG layout now renders in waterfall layout

### Docs
- Initial CHANGELOG

<a name="v0.1.0"></a>
## v0.1.0 (2025-06-25)

### Added
- MVP profiling engine with full CI setup and metadata tracking
