# DSBF Refactor Plan

**Status:** proposal, 2026-10-15; decisions D1–D11 accepted 2026-10-16 ([ADRs](adr/README.md)); Phase 4 scope revised 2026-10-19 (U4-00)
**Inputs:** [`REVIEW_FINDINGS.md`](REVIEW_FINDINGS.md) (fixes F-01 to F-39, next steps N-*),
[`docs/WALKTHROUGH.md`](WALKTHROUGH.md) (measured practice run), and the maintainer's goals below.

This plan rebuilds DSBF's *scaffolding* (data handling, execution, contracts, storage, serving) and
keeps its *substance*: the statistical logic in the existing tasks, the guidance text, the
five-dimension quality framework, hedged language, and the validation archetypes. It replaces
incremental milestones M0 to M6 that were proposed in the review. The work itself is tracked as
single-session **units** in [`ROADMAP.md`](../ROADMAP.md).

---

## 1. Goals

DSBF should work in three ways, and every design decision below is checked against all three.

| Mode | Who | What they need from DSBF |
|---|---|---|
| **1. Embedded** | Teams with existing pipelines, orchestrators, warehouses and dashboards | A library and CLI that run selected checks and return **structured, versioned results** they can act on or load into their own dashboards. No DSBF UI, database or files required. |
| **2. Standalone** | A person or team with no data tooling | DSBF as the pipeline: point it at data, keep history, browse results in the bundled dashboard. Local first; team server later. |
| **3. Anything in between** | Everyone else | Any layer can be used on its own or swapped: your checks plus DSBF's store, DSBF's checks plus your dashboard, DSBF's dashboard reading results produced in CI, and so on. |

Requirements that follow from the maintainer's answers:

- **R1. Choose your own analysis.** Checks are individually selectable, grouped into packs, and resolve their own dependencies.
- **R2. Report-only by default.** A failing check never stops the run or the pipeline. Every failure, skip or unavailable check produces a **visible explanation in the results**, so nobody needs to read logs. Users *may* define their own rules that turn findings into exit codes or actions.
- **R3. Broad reach, built in tiers.** It should run from one person's laptop up to cloud-native pipelines in large organizations, and on tabular data first, then larger, remote and non-tabular data, without redesigning the core.
- **R4. Local first, team later.** Storage, auth and job execution sit behind interfaces, so the team version is new implementations, not a rewrite.
- **R5. Statistical honesty is enforced by structure**, not by convention.

### Assumptions (correct these if they're wrong; they change sizing, not shape)

- **A1.** There's one maintainer, sometimes with help, so work is sequenced into small, independently shippable PRs.
- **A2.** DSBF has never had a published release or external users (not on PyPI; the GitHub repo has no forks, checked 2026-10-15). Development-era schemas, configs, commands and data are replaced when the design requires it, with no compatibility shims. Runs in existing local databases are **not** migrated; you re-profile. Compatibility guarantees start with the published 1.x schema (§4.3).
- **A3.** `feature/viz_refactor` is parked after the histogram fix ([Phase 0](#phase-0-stabilise-and-baseline-sm)). The new dashboard is built on the new API in Phase 6.
- **A4.** Python ≥ 3.12, as today.

---

## 2. Design principles

1. **Library first.** The core is a pure function: data plus configuration in, results out. It does no file I/O, database access, global logging setup or network calls. Everything else is an optional layer that consumes core output.
2. **Findings are the product.** Every check emits `Finding` records with a common shape. Raw computations are attached as *evidence*, fetched on demand. Dashboards, policies, exports and agents all read findings.
3. **Contracts before code.** Results, configuration and plugin interfaces are typed, versioned schemas, published as JSON Schema and TypeScript types. External consumers (mode 1) can depend on the schema without depending on DSBF internals.
4. **Measure once, decide many times.** A single profiling pass computes per-column statistics. Most checks are cheap rules over that profile. Expensive analyses run separately, with budgets and caching.
5. **Failure is data.** Check outcomes are first-class statuses with reasons. The run always completes and always explains itself.
6. **Every layer is optional and swappable.** Optional features are installed through extras, and plugins are discovered through standard entry points.
7. **Honesty by construction.** A finding can't be created without its sample size, method and provenance. Uncertainty and sampling are part of the schema.

---

## 3. Target architecture

```
                   ┌──────────────────────────────────────────────────────────────┐
  Integrations     │ CLI │ Orchestrator adapters │ GitHub Action │ MCP server │ SDK │   (thin wrappers)
                   └──────────────┬───────────────────────────────────────────────┘
                                  │ all call the library API
 ┌────────────────────────────────▼─────────────────────────────────────────────┐
 │  L5 policy   user rules over findings → annotate / warn / fail / notify       │
 │  L6 sinks    in-memory Run · JSON/NDJSON · Parquet · HTML report · Store      │
 ├───────────────────────────────────────────────────────────────────────────────┤
 │  L4 engine   Planner (selection, dependencies, validation, budget) →          │
 │              Executor (isolation, parallelism, cache, timeouts, events)       │
 │  L3 checks   Rule checks · Analysis checks · Synthesis checks (packs, plugins)│
 │  L2 profile  one-pass ColumnProfile / DatasetProfile (+ cached value arrays)  │
 │  L1 data     Source adapters → Dataset (Arrow/Polars) · sampling · fingerprint│
 │  L0 model    Finding · Evidence · Note · CheckResult · Run · Profile · Config │  ← zero-heavy-dep contracts
 └───────────────────────────────────────────────────────────────────────────────┘
      ▲ core: pure, no I/O                                   │ optional
 ┌────┴──────────────────────────────────────────────────────▼──────────────────┐
 │  L7 store    RunRepository (SQLite → Postgres) · ArtifactStore (local → S3)   │
 │  L8 service  FastAPI: read API v2 · job API · auth (none → token/OIDC)        │
 │  L9 dashboard Vue 3 + ECharts, generated TS types, served by `dsbf dashboard` │
 └───────────────────────────────────────────────────────────────────────────────┘
```

How each mode uses the layers:

| Mode | Layers used |
|---|---|
| 1. Embedded, in process | L0–L4, plus the in-memory Run or a Parquet/JSON sink |
| 1. Embedded, any orchestrator | CLI (`dsbf check`) → sinks, plus an optional policy for exit codes |
| 1. Embedded, own dashboard | Parquet/NDJSON findings sink → their warehouse or BI tool, with the schema published |
| 2. Standalone, local | Everything, with SQLite and the local filesystem |
| 2. Standalone, team (later) | Everything, with Postgres, S3, auth and the job queue |
| 3. In between | Any combination. For example, CI runs `dsbf check --sink store` and the team browses `dsbf dashboard`. |

### 3.1 Packaging

DSBF stays **one distribution** with extras. Splitting into several packages would cost a solo
maintainer more than it saves.

| Install | Contents | Key deps |
|---|---|---|
| `dsbf` | L0–L6, core check pack (rules + basic statistics), CLI, JSON/Parquet/HTML sinks | polars, pyarrow, numpy, scipy, pydantic, typer |
| `dsbf[ml]` | Isolation forest, mutual information, ML-readiness extras | scikit-learn |
| `dsbf[timeseries]` | ACF/PACF, stationarity, decomposition | statsmodels |
| `dsbf[text]` | Fuzzy duplicates, string anomaly scoring | rapidfuzz |
| `dsbf[excel]` | Excel file sources (`.xlsx`, `.xls`) | fastexcel |
| `dsbf[pandas]` | pandas input/output convenience (pandas DataFrames are accepted via Arrow even without it) | pandas |
| `dsbf[sql]` | Warehouse and database sources with pushdown (Tier 3) | ibis-framework (+ backend extras) |
| `dsbf[server]` | Store, service, bundled dashboard | fastapi, uvicorn, sqlalchemy, alembic |
| `dsbf[all]` | Everything | |

If a selected check's extra isn't installed, it shows up in the run as **`unavailable`** with a note
such as "install `dsbf[timeseries]`". It never errors (R2).

Layout moves to `src/dsbf/`. Package data is loaded with `importlib.resources`, and runtime data
lives under `DSBF_HOME` (default from `platformdirs`). This removes F-16, F-17 and F-31 by
construction.

---

## 4. Core contracts (L0)

All models are pydantic v2, exported as JSON Schema (`dsbf schema export`), with TypeScript types
generated for the dashboard. Every serialised object carries `schema_version`.

### 4.1 Vocabulary

| Term | Meaning |
|---|---|
| **Source** | Where data comes from: a DataFrame, file, glob, SQL query or table, or (later) other modalities |
| **Dataset** | A loaded or lazily referenced Source, with identity, fingerprint, modality and sampling provenance |
| **Profile** | Per-column and dataset-level summary statistics from the profiling pass |
| **Check** | One unit of analysis (today's "task"). It has an id, version, parameters, requirements and cost |
| **Pack** | A named, versioned group of checks with default parameters (for example `core`, `ml`, `timeseries`, and later domain packs) |
| **Plan** | The resolved, validated set of checks to run. It can be inspected before running (`dsbf plan`) |
| **Finding** | One statement about the data: subject, dimension, severity, message, evidence and guidance |
| **Evidence** | The numbers behind a finding: metrics, `n_used`, method, assumptions, CI and provenance |
| **Note** | An operational message: why a check was skipped, errored, unavailable or degraded |
| **Run** | The result of executing a Plan on a Dataset: manifest, profile, check results, findings, notes, health |
| **Policy** | User rules that map findings and health to actions (annotate, warn, fail, notify) |
| **Sink** | Where a Run goes: memory, files, store, notifier |

**Rename "task" to "check" in the public API.** It's the word pipeline users already use. There's
no public alias. The legacy adapter (§6.2) is the only place the old name survives, and it's
deleted at the end of Phase 4.

### 4.2 Key models (sketch)

```python
class Subject(BaseModel):  # replaces "col1|col2" string keys (F-10)
    kind: Literal["dataset", "column", "column_pair", "column_group", "row_set"]
    columns: tuple[str, ...] = ()  # ordered; arbitrary characters allowed
    # extensible later: "document", "series", "partition", ...


class Evidence(BaseModel):
    metrics: dict[
        str, Scalar | tuple[Scalar, ...] | dict[str, Scalar]
    ]  # Scalar = bool | int | float | str | None; finite floats only (also inside a
    # list or mapping); NaN/inf/NaT → None, named in non_finite_metrics (F-36)
    non_finite_metrics: tuple[str, ...] = ()
    n_used: int
    n_total: int
    method: str  # e.g. "spearman_r", "vif", "shapiro_wilk"
    assumptions: tuple[
        AssumptionCheck, ...
    ] = ()  # contracts are frozen; sequences are tuples
    interval: Interval | None = None  # CI where defined
    sampled: SamplingInfo | None = None  # provenance (F-09, F-15)
    details_ref: str | None = None  # pointer to bulky evidence, fetched lazily (F-35)


class Action(BaseModel):  # a step, readable by a program (today's guidance actions)
    action: str  # the verb, e.g. "transform"
    columns: tuple[
        str, ...
    ] = ()  # the columns it applies to; may differ from the finding's
    method: str | None = None
    condition: str | None = None
    detail: str | None = None
    params: dict[str, Scalar] = {}  # anything else, e.g. from_dtype


class Guidance(BaseModel):
    phase: Literal["eda", "ml"]  # "eda" describes the data; "ml" says what to do first
    actions: tuple[Action, ...] = ()


class Finding(BaseModel):
    id: str  # stable hash(check_id, subject kind and columns, code) → diffable across runs
    check_id: str
    check_version: str
    code: str  # e.g. "high_null_rate", "near_duplicate_rows"
    subject: Subject
    dimension: Dimension  # completeness | validity | usability | redundancy | leakage | distribution | relationship | temporal
    severity: Severity  # info < notice < warn < error
    confidence: Confidence  # low < moderate < high; capped per check (hedged-language principle)
    title: str
    message: str  # self-contained, includes the observed values
    evidence: Evidence
    guidance: Guidance | None  # one phase of advice (today's add_guidance); a positive
    # legacy "good" level is Severity.INFO plus the tag "good"
    tags: tuple[str, ...] = ()


class Note(BaseModel):  # an operational message
    code: str
    message: str
    severity: Severity = Severity.INFO
    hint: str | None = None  # what the reader can do
    blocked_by: tuple[str, ...] = ()  # the checks that caused a skip
    exception_type: str | None = None
    detail: str | None = None


class CheckResult(BaseModel):
    check_id: str
    check_version: str
    status: CheckStatus  # ok | not_applicable | unavailable | skipped | error | timeout
    notes: tuple[Note, ...] = ()  # required when status != ok (R2, F-20)
    findings: tuple[Finding, ...] = ()  # only when status == ok
    output: dict | None = (
        None  # typed per check via output model; bulky parts → artifacts
    )
    non_finite_outputs: tuple[str, ...] = ()  # keys of output that held NaN/inf/NaT
    timing: Timing
    params: dict  # resolved parameters, recorded for reproducibility (finite only)


class RunHealth(BaseModel):  # derived from the results by RunHealth.from_results
    counts: StatusCounts  # one integer per status
    issues: tuple[
        HealthIssue, ...
    ]  # error, timeout, skipped, unavailable; with reasons
    findings_at_warn_or_above: int


class Run(BaseModel):
    schema_version: str
    run_id: UUID  # UUID-based (F-15)
    started_at: AwareDatetime
    finished_at: AwareDatetime
    dataset: DatasetManifest  # identity, fingerprint, source, shape, sampling
    plan: PlanManifest  # checks, versions, params, dsbf version (F-18); a check's pack is a name, packs have no version
    profile: DatasetProfile
    results: dict[str, CheckResult]
    health: RunHealth  # counts and reasons by status: the "never read logs" summary
    policy: PolicyOutcome | None = (
        None  # what the user's policy decided, if one was applied
    )
    # Run.findings is a property, not a field: every result's findings flattened and
    # ordered by severity, so a serialised run carries each finding once (under results).
    # Run.build derives health from the results and checks the parts agree
```

`ColumnProfile` is a union discriminated by `semantic_type` (`numeric | categorical | boolean |
datetime | text | identifier | unknown`); `identifier` and `unknown` mirror what the legacy
`infer_types` emits. `DatasetProfile.n_rows` counts the rows profiled, so for a sampled dataset it
equals `DatasetManifest.sampling.sampled_rows`.

### 4.3 Versioning rules

- `schema_version` follows semver. Additive changes are minor and breaking changes are major. The JSON Schema is checked in, and CI fails on an unannounced breaking diff.
- Each check has its own `version`. Changing its statistics or output bumps the version, which invalidates cache entries ([§6.4](#64-executor)) and is called out in golden-run diffs.
- `Finding.id` is stable for the same check, subject and code, so findings can be compared across runs ("new since yesterday", "resolved").

---

## 5. Data and profiling (L1–L2)

### 5.1 Source tiers

| Tier | Scope | Mechanism | Phase |
|---|---|---|---|
| **T1** | In-memory pandas, Polars or Arrow; CSV, Parquet, Excel, JSON/NDJSON that fit in RAM | Polars (pandas accepted via Arrow, zero-copy where possible) | 2 |
| **T2** | Larger-than-memory files and globs, local or object storage (`s3://`, `gs://`, `abfs://`) | Polars lazy/streaming scans; sampling at read time with recorded provenance; fsspec | 7 |
| **T3** | Databases and warehouses (Postgres, DuckDB, Snowflake, BigQuery, …) | Ibis: profile aggregations pushed down as SQL; samples pulled for analysis checks | 7 |
| **T4** | Recurring datasets (daily loads, partitions) | Dataset identity plus history; drift computed from **stored profiles** (histograms, top-k, sketches), so no raw data has to be kept | 7 |
| **T5** | Non-tabular: text corpora, panel time series, nested JSON, media metadata | New `modality` plus modality-specific profilers and check packs; nested data flattened to a tabular view where sensible | 9 |

`Dataset` is the boundary. Checks never see a file path or connection, only a `Dataset` and its
`Profile`. Adding a tier means adding a source adapter, not changing checks.

### 5.2 Data handling inside checks

- The internal representation is **Polars / Arrow**. Checks that need raw values get **NumPy arrays from a per-column cache** (`dataset.values(col, dropna=True)`), which is what scipy, statsmodels and sklearn take anyway. There are no per-check pandas copies (F-30).
- pandas is used only at the edges: accepting user DataFrames and optional export.
- Sampling happens **once, at the Dataset**. Its provenance (strategy, seed, rows) flows into every `Evidence.sampled`. Stratified sampling is either implemented properly or rejected (F-09).
- `fingerprint` is a content hash: a schema hash plus a streaming hash, or a cheap partition-level hash for T2/T3. It's used for caching and for detecting "file changed since profiled" (F-15).

### 5.3 The profiling pass

One scan (or a few, for T2/T3) produces a `ColumnProfile` per column:

- **Type:** physical dtype; semantic type inferred from a **sample** with confirmation (fixes F-37); confidence.
- **Counts:** rows, nulls, distinct (exact up to a threshold, otherwise HyperLogLog), zeros, empty strings, non-finite values.
- **Numeric:** min, max, mean, std, skew, kurtosis, quantiles (exact or t-digest), **equal-width histogram bins with the binning rule recorded** (fixes F-12).
- **Categorical/text:** top-k values with counts, length statistics, character-class profile, pattern signatures.
- **Datetime:** range, inferred frequency, gaps summary, timezone.
- **Dataset-level:** shape, duplicate-row count (hash-based), per-column missingness matrix summary.

Rules:
- Statistics are computed **per column on that column's non-null values** (fixes F-32). Multivariate checks choose their own deletion strategy and report `n_used`.
- The profile is serialisable and stored with every run. It powers the dashboard's charts (no backend image rendering, N-A2), drift across runs (T4), and most rule checks.

**Parity gate:** before any check moves to the profile, the profile must reproduce today's
`summarize_*` outputs on the validation archetypes, within tolerance, or differ in documented ways.

---

## 6. Checks and engine (L3–L4)

### 6.1 Check kinds

| Kind | Reads | Cost | Examples (current task → new check) |
|---|---|---|---|
| **Rule** | Profile only | microseconds | constant columns, zeros, high cardinality, ID-like, single dominant value, near-zero variance, null rate, skewness, kurtosis, entropy, boolean summary, dtype optimisation, string length |
| **Analysis** | Profile plus value arrays or the Dataset | moderate to heavy | outliers, normality, pairwise associations, contingency, ANOVA/Kruskal/Mann-Whitney, bimodality, missingness mechanism, mutual information, encoded/regex/mixed-type, fuzzy duplicates, time series |
| **Synthesis** | Other checks' findings and results | cheap | quality dimensions (today's `data_quality_scorer`), ML readiness, target leakage roll-up |

About 20 of today's 67 tasks become rule checks. In the practice run those were the tasks
together taking under 3 seconds; now they share one profile instead of each rescanning the data.

### 6.2 Check interface

```python
class OutlierParams(CheckParams):
    method: Literal["all", "iqr", "zscore", "mad", "isolation_forest"] = "all"
    flag_threshold: float = Field(0.01, ge=0, le=1)
    zscore_threshold: float = 3.0
    ...


@check(
    id="outliers",
    version="2.0.0",
    kind="analysis",
    params=OutlierParams,
    output=OutlierOutput,  # a CheckOutputModel, so the output has a schema
    applies_to=SemanticTypes(
        "numeric"
    ),  # engine filters columns (fixes F-11 by construction); None for a dataset-level check
    requires=Requires(profile=["quantiles"], checks=[]),
    after=[],  # soft ordering, no failure propagation (F-21)
    cost=Cost.MODERATE,
    extras=[],  # e.g. ["ml"] for isolation forest → "unavailable" if missing
    packs=["core"],
    dimensions=[Dimension.VALIDITY],
)
def outliers(ctx: CheckContext, params: OutlierParams) -> CheckOutput:
    for col in ctx.columns:  # already filtered by applies_to
        x = ctx.values(col)  # cached NumPy, non-null (analysis checks only)
        ...
        ctx.finding(
            code="outliers_iqr",
            subject=Subject.column(col),
            severity=...,
            confidence=...,
            title=...,
            message=...,
            evidence=...,
            guidance=...,
        )
    return ctx.output(OutlierOutput(...))


# A synthesis check reads other checks through ctx.result("other_check"), for the
# checks it declares in requires.checks (hard) or after (soft; None if it didn't run).
# A rule check reads ctx.profile only. run_check(check, context) calls a check and
# builds its CheckResult; the executor wraps it with isolation.
```

What this removes:
- **Parameters are declared:** unknown or ignored config keys are validation errors (F-07, F-22), and defaults, docs and the generated config reference all come from one place.
- **Column selection belongs to the engine:** `ctx.columns` is the only column list a check sees (F-11).
- **No try/except boilerplate:** the executor owns isolation (F-06, the ~60 copies).
- **Empty results have a defined shape,** because outputs are typed models (F-03).
- **Optional enrichment can't fail a check:** recommendation tips are attached after the check returns, and a failed tip lookup becomes a note (F-31).

**Legacy adapter (migration only):** `LegacyTaskCheck` wraps an existing `BaseTask`, gives it the
old `AnalysisContext` interface, and converts `TaskResult.guidance` into `Finding`s. This lets the new
engine run **every** existing task from Phase 3 on, so migration proceeds one check at a time
against a running system.

How the adapter works (`dsbf.legacy`; `legacy_checks(session)` returns one `Check` per registered task, and the package goes when the last task is migrated):
- A `LegacySession` holds what the tasks of one run share: the dataset, its profile, the legacy config, a directory for plots and the reference frame. It is built from the dataset and profile passed to `execute`, because a `CheckContext` shows a check only the columns it applies to and a legacy task picks its own.
- Each task is a check with the task's name as id, version `0.1.0`, `kind="analysis"`, `runs_in="thread"` (a run-time `Check` cannot be imported by a worker process) and pack `legacy`. `depends_on` becomes `requires.checks`. Three tasks read results they never declared, so they get them as `after`; a test scans the task sources and fails when one is missing. `applies_to` is the profile types the task's `expected_semantic_types` map to, so a dataset with none plans it `not_applicable` where the legacy engine called it and got an empty result.
- The task runs against a `LegacyContext`: the data as pandas or Polars (a private copy per check, since a task may edit its input and checks overlap), the config, the upstream `TaskResult`s rebuilt from the stored output, and `semantic_types` from the profile (legacy `continuous`, `id` and `categorical` for the profile's `numeric`, `identifier` and `categorical` or `boolean`) and `inferred_dtypes`.
- The output is the whole `TaskResult.to_dict()`, made JSON-plain (`LegacyTaskOutput.result`). A task that returns `failed` ends `error` with its `error_metadata` as the note; one that returns `skipped` ends `skipped` with its message. The parameters are the task's config, the shared config, a digest of the task's module and the shared code, and the output directory or reference digest for a task that writes files or reads the reference, so a cached result is reused only when all of them are the same. Findings are not derived yet.

How the entry point works (`dsbf.profile(source, ...)`, in `dsbf/library.py`; the subpackage that holds the profile pass is `dsbf.profiling`):
- The settings are a `ProfileConfig` (`dsbf.config.profile`): packs, include and exclude, `depth`, `target`, `time_index`, `sample`, `backend`, `limits`, `workers`, `plugins`, per-check `params` and per-task overrides. The keyword arguments are its `overrides` layer, a `config=` path is its project file, and the working directory is never read, so a library call is the same wherever it runs. A bad value, or a `target`, `time_index` or task name the data does not have, is a `ConfigError` with a suggestion.
- The flow is `load` (or the `Dataset` given), `Dataset.sample` if asked, the profile pass on the sampled rows, a `LegacySession`, `legacy_checks`, `make_plan` and `execute`. `checks` takes `+id` and a bare id to include and `-id` to exclude. The default pack is `core` alone (U4-15); `packs=["core", "legacy"]` adds the legacy tasks that no core check replaces, and a legacy task that a core check replaces is left out of that selection (`dsbf.legacy.supersedes`, below); the backend defaults to `pandas`, because the legacy tasks are written for pandas frames (on titanic 47 of 63 checks differ on Polars). The inferred stage is not computed, because no check reads it.
- **Core and legacy side by side** (`dsbf.checks`, `dsbf.legacy.supersedes`): `core_pack()` holds the migrated checks and is registered explicitly by `dsbf.profile()` and `dsbf plan`, never on import. `SUPERSEDED_BY` maps a legacy task to the core check that replaces it. When that check's pack is selected and the caller did not name the task in `checks`, the task is adapted into the `legacy_superseded` pack, so the default selection leaves it out, while a check that `requires` it still pulls it in. `make_legacy_plan` also adds a superseded task that a planned legacy task reads through `UNDECLARED_READS` (the planner pulls in only hard requirements). A superseded task that runs reports its output and no findings (`LegacyParams.superseded_by`, which is in the cache key, and a `findings_superseded` note), so each finding is reported once. The adapter can't use `PlanEntry.selected` for this: a check function sees only its context and parameters.
- Nothing is written unless asked: the result cache is off (`cache=True` is `DiskCache.default()`, or pass any `ResultCache`), and the two legacy checks that write plots are left out of a run with no output directory.
- `Run.notes` holds what happened to the run as a whole: plugin-load problems, plan notes, and `cache_unavailable` when the cache failed during the run. `Run.render_health()` prints them under the check issues. The version in the manifest and `dsbf.__version__` come from `importlib.metadata`.
- A `Run` is in memory and can be megabytes on text-heavy data: titanic at `deep` is 8.5 MB, 7.9 MB of it the output of `fuzzy_duplicate_detection`, which Phase 4 redesigns.

### 6.3 Planner

Input: Dataset, Profile (or the schema only, for a dry run), selection and configuration.
Output: a `Plan` that can be printed (`dsbf plan`) and is recorded in the run manifest.

1. **Select** by packs, include/exclude lists, depth presets (`quick | standard | deep`), cost budget, dimension, tag, and modality and semantic applicability.
2. **Resolve** hard requirements (pulled in automatically and marked as such) and soft `after` orderings. Cycles are an error at plan time.
3. **Validate** each check's parameters against its model. Errors name the check, key and allowed values.
4. **Annotate** each planned check with an expected status: `will_run`, `not_applicable` (no eligible columns), or `unavailable` (missing extra or source capability). Every status comes with a reason.

How steps 1 and 2 work in `dsbf.engine` (`Selection`, `make_plan`):
- `Depth` is a cost ceiling: `quick` keeps cheap checks, `standard` adds moderate ones, `deep` adds heavy ones. An explicit `max_cost` can only lower it. The candidates are the checks in the named packs (all checks when none is named); depth, dimension, tag and `exclude_stages` filters are ANDed; `include` adds by id past every filter, and `exclude` removes by id last.
- An unknown pack, an unknown `include` or `exclude` id, and an id in both lists are plan errors that suggest the closest name, and all of them are reported together. A pack with no checks is valid and selects nothing.
- Hard `requires.checks` are pulled in transitively past the filters and marked (`selected=False`, `required_by`); a requirement that is unregistered or excluded is a plan error naming both checks. A soft `after` orders checks that are in the plan and pulls nothing in. Hard and soft edges are laid out in levels (each sorted by id, so plans are deterministic and a level can run in parallel); a cycle is a plan error that prints the path and the kind of each edge.
- The inferred data stage does not select checks, as in the legacy engine. A check's `stage` is informational and filterable with `exclude_stages`.

How steps 3 and 4 work (`make_plan(registry, selection, params=..., profile=...)`):
- `params` maps a check id to its parameters, and each entry carries the validated `CheckParams`. Every mistake is reported in one `PlanError`: an unknown key (with the allowed keys and the closest name), a value outside its type or bounds, a check that takes no parameters, and an id the registry doesn't hold. Parameters for a registered check outside the plan are allowed, so one configuration serves any selection.
- Each entry has an `Expectation` (`will_run`, `not_applicable`, `unavailable` or `skipped`, with a reason, an install hint where one applies, and the columns the check would read). A missing extra comes first, then a hard requirement that is `unavailable` or `skipped` (so the dependent is `skipped`, as the executor will do), then a profile with nothing for the check to read: no column of the `applies_to` type, none with the `requires.profile` features it needs, or no dataset-level feature. A `not_applicable` requirement does not block its dependents. Without a profile only the first two are judged, and `Plan.profiled` is false.
- `dsbf plan DATA` loads and profiles the file, selects with `--pack`, `--include`, `--exclude`, `--depth`, `--max-cost`, `--dimension`, `--tag`, `--plugin` and `--param check.key=value`, and prints the plan by level. A plan error or an unreadable file exits 2; unavailable or skipped checks do not change the exit code.

### 6.4 Executor

- **Isolation (R2):** any exception becomes `status="error"` with a `Note` containing the error type, message, a **hint**, and the traceback in an artifact. Hard dependents become `skipped` with a note naming the blocker. Soft dependents run. Nothing propagates. Only *setup* errors (unreadable source, invalid config) stop a run, and even then the CLI writes a Run containing the error by default.
How isolation works in `dsbf.engine` (`execute(plan, dataset, profile, dsbf_version=...)` returns a `Run`):
- Checks run level by level, the checks of a level at the same time (see Parallelism). Each ends in an explicit status, decided in this order: a missing extra is `unavailable` (the check is not called); a hard `requires` upstream whose actual status is not `ok` or `not_applicable` makes it `skipped`, with `blocked_by` naming the blockers; otherwise the check runs and the status `run_check` returns is trusted, never inferred from "no exception escaped" (F-06). A `not_applicable` upstream leaves its dependents running, and a failed soft `after` upstream changes nothing.
- A check that finds, part way, that it cannot answer can say so without raising: `ctx.stop(status, code, message, ...)` ends it `not_applicable`, `unavailable`, `skipped` or `error` with that note, and `run_check` returns the status. Its findings are dropped as for an exception, and only `ok` and `not_applicable` leave its dependents running. `ok` and `timeout` are not the check's to choose.
- Any exception from a check, or from building its context, becomes an `error` result with a `check_failed` note: exception type, message, a hint, and the traceback in `Note.detail` (no artifact store exists before Phase 6). Findings recorded before the exception are dropped. `KeyboardInterrupt` and `SystemExit` propagate. A profile that doesn't belong to the dataset is a `ValueError` before anything runs.
- One `eligibility(spec, profile)` decides which columns a check reads. The plan's `Expectation`, the `CheckContext` (`columns`, the profile view) and `run_check` all use it, so a column lacking a required profile feature is out of all three, and a check whose required dataset feature is missing is `not_applicable` (`profile_feature_missing`) without being called.

- **Budgets:** per-check timeouts and an optional run-level budget. A check that exceeds its timeout becomes `timeout` with a note. A thread cannot be killed, so a threaded check stops when it next calls back into its context (`ctx.checkpoint()`); only a process worker can be terminated. Heavy checks declare caps (rows, pairs, lags) and record them in their evidence (F-35, F-39).
- **Parallelism:** checks at the same DAG level run concurrently, using threads for checks that mostly run in compiled code and a process pool for pure-Python checks. Default: one worker per core, capped. Results are deterministic regardless of worker count.
  How it works in `dsbf.engine` (`execute(..., workers=None)`):
  - `None` is one worker per core the process may use, at most 8; the checks of a level start in plan order as workers free up, and a level waits for the one before it. A check's timeout runs from the moment it starts, not from when it was queued. A timed-out thread frees its worker when it is abandoned, so more threads than `workers` can exist.
  - A check declares `runs_in="process"` (default `"thread"`) when it loops in pure Python. It then runs in a worker process whatever `workers` is, and a timeout terminates the process (the note says it was killed). A worker is one process per check from a `forkserver` context (`spawn` where there is none), preloaded with the engine so it starts in about 10 ms. Polars' thread pool makes a plain fork unsafe. The frame is written once per run to an uncompressed Arrow file that workers memory-map through pyarrow (no copy, so no per-worker 400 MB for 1M × 50); the check travels by name, so it must be declared at the top level of an importable module, and one that is not runs on a thread with a `ran_in_thread` note. A worker that exits without a result is an `error` with a `worker_died` note, and `SystemExit` in a worker is that, not a stop of the run.
  - The calling thread does all the waiting and emits every event, so hooks need no locking: per check `check_started`, its notes, `check_finished`; across checks the order of completion (plan order with one worker), with plan positions. `Run.results` is in plan order, and a run is the same, timing aside, for any worker count.
  - Timing: a thread check's CPU and peak memory are the whole process's, so `Timing.shared_with` counts the checks that overlapped on threads; a worker's are its own (`shared_with` is 0), and a killed worker has no CPU figure.
- **Caching:** results are keyed by `(dataset.fingerprint, check.id, check.version, params_hash, sampling)`. Rerunning after changing one threshold recomputes only the affected checks and their dependents. The cache is local disk and pluggable; it is on only when asked for (`cache=True` in `dsbf.profile()`).
  How it works in `dsbf.engine` (`execute(..., cache=None)`; off unless given, so a library call writes nothing it was not asked to):
  - The key hashes the dataset fingerprint, the sampling record, a digest of the profile (checks read it, and the profile code can change while the data does not), the check's id, version and parameters, and the digest of each upstream result the check may read (`requires` and `after`; a result's digest leaves out its timing). Because an upstream result is in the key, changing one check's parameters re-runs it and the checks that read its changed result; a dependent whose upstream re-ran to the same result is still served, and one whose soft upstream failed last time is not.
  - Only `ok` results are stored: an error, timeout, skip or missing extra depends on the machine and the limits, and a `not_applicable` result costs nothing to recompute. A hit is looked up after the extra and requirement rules and before the run budget, so it is served even when the budget is gone. It has no `CheckStarted`; its notes and `CheckFinished` come as usual, and `Timing.cached` is true with the lookup's wall time and no CPU or memory.
  - `ResultCache` is a two-method protocol (`get`, `put`). `DiskCache(root)` writes one JSON file per entry, `<root>/<check id>/<digest>.json`, by a temporary file and a rename; `DiskCache.default()` is `<DSBF_HOME>/cache/results`. An entry that cannot be read, is of another format or holds another key is a miss. A cache that raises is logged once and ignored for the rest of the run; it never changes the `Run`.
- **Events:** structured events (`check_started`, `check_finished`, `note`) go to registered hooks. The CLI renders a progress bar, the service streams progress, and OpenTelemetry export is optional. Logging is library-friendly: a `dsbf` logger hierarchy with no global changes and no markup in data (F-19).
- **Resource accounting:** wall time and CPU time per check, peak memory by sampling (labelled as approximate), recorded in `timing` (F-24).

### 6.5 Plugins

- Third-party checks and packs are discovered through the `dsbf.checks` / `dsbf.packs` **entry-point groups** (`pip install dsbf-healthcare`).
- Local directories are still supported (`plugins: [./my_checks/]`). They're imported under unique module names and are reload-safe (F-23).
- Plugins get the same contracts, parameter validation, isolation and versioning as core checks.
- A `Registry` holds `Check` objects by id and is built explicitly; nothing registers on import. `dsbf.checks` entry points load a `Check` or an iterable of them, and `dsbf.packs` entry points load a `Pack`. `load_directory` imports a folder as a package with a name unique to its path, so plugin files can use relative imports and two folders can each have a `utils.py`. A source that was loaded is not loaded again, in this registry or another in the same process, so editing a plugin file needs a new process.
- A plugin that fails to import, declares nothing, or reuses a taken id never raises: the registry records a `Note` (`plugin_load_failed`, `plugin_invalid`, `plugin_empty`, `check_id_conflict`) for the run to report.

---

## 7. Failure handling, health and policy (L5)

### 7.1 Default behaviour (report-only)

- A run **always completes** and **always returns a Run** (except unrecoverable setup errors, which still return a Run when invoked through the CLI).
- `Run.health` summarises outcomes:

```
DSBF run 7f3c…  titanic  (891 × 15, full scan)   61 checks
  ok 54 · not applicable 3 · unavailable 1 · skipped 2 · error 1
  ✖ error        fuzzy_duplicates      Comparison budget exceeded after 300s → partial results not stored. Hint: lower max_rows or install dsbf[text].
  ↷ skipped      ml_readiness          Requires 'skewness' (error).
  ⦸ unavailable  stationarity          Install dsbf[timeseries].
  ⚠ 14 findings at warn or above → see findings.parquet / dashboard
```

- The CLI prints this summary at the end of every run. The dashboard shows it as a **Run health** panel. The data is the same `RunHealth` object in the library.
- **Exit code is 0 by default**, even when checks error.

### 7.2 User-defined policies (opt-in)

Declarative YAML is evaluated over findings and health. There's **no `eval`** (it replaces the
current `eval` in the recommendation engine, F-31).

```yaml
# policy.yaml
version: 1
rules:
  - name: block-on-leakage
    when: { dimension: leakage, severity: ">=error" }
    then: fail                    # exit code 1, recorded in Run.policy
  - name: sparse-columns
    when: { code: high_null_rate, metric: { null_rate: ">0.3" } }
    then: warn
  - name: check-errors-are-notes
    when: { health.status: error }
    then: annotate                # default; never fails unless a rule says so
  - name: tell-the-team
    when: { severity: error, new_since: previous_run }
    then: { notify: slack:#data-quality }
```

Actions are `annotate | warn | fail | notify`. `fail` affects only the exit code and policy outcome;
the run still completes and writes all sinks. Policy outcomes are part of the Run, so a dashboard
or downstream system sees *why* a pipeline was failed.

The schema is `dsbf.model.policy` (`Policy`, `Rule`, `Matcher`, `PolicyAction`, `PolicyOutcome`):

- A `when` holds at least one condition, and every condition given must hold. Finding conditions
  are `dimension`, `code`, `check`, `column`, `tag`, `severity`, `confidence`, `metric` and
  `new_since: previous_run`. `health.status` selects checks and may only be narrowed by `check`.
- `severity`, `confidence` and each `metric` entry take one comparison: `==` (the default), `!=`,
  `>=`, `>`, `<=` or `<`. Quote a value that starts with an operator in YAML. Severity and confidence
  compare by rank, and metric operands are numbers (or text and booleans for `==` and `!=`).
- `then` is one action or a list, and defaults to `annotate`. `notify` takes a `scheme:destination`
  target; the other actions take none.
- `Run.policy` lists every rule in order with the finding ids and check ids it matched, plus `failed`,
  so the exit code (1 only when a triggered rule has a `fail` action) can be explained.
- Not yet expressible: a range on one setting (`>0.1` and `<0.5`).

---

## 8. Outputs, storage and service (L6–L8)

### 8.1 Sinks

| Sink | Purpose | Phase |
|---|---|---|
| In-memory `Run` | Library users | 3 |
| `run.json` + `findings.ndjson` | Simple file integration, CI artifacts | 5 |
| `findings.parquet`, `profile.parquet`, `results.parquet` | Load into warehouses or BI tools (**mode 1 with your own dashboard**) | 5 |
| Static HTML report | Share without any server | 5 |
| Store (SQLite / Postgres) | History and the bundled dashboard | 6 / 8 |
| Notifiers (webhook, Slack, email), OpenLineage facets | Pipeline and observability ecosystems | 8 |

Bulky evidence (fuzzy pairs, contingency tables, full pairwise matrices) goes to **artifacts**,
written as compressed Parquet or JSON and referenced by `details_ref`. Findings stay small (F-35).

### 8.2 Store

- `RunRepository` interface (save, get, list, query findings, history by dataset, compare, delete, gc). It's implemented with SQLAlchemy Core and **Alembic migrations** (no more ad-hoc `ALTER TABLE`, F-25).
  - SQLite implementation in Phase 6, Postgres in Phase 8.
  - Tables: `datasets` (stable identity: name plus namespace plus source fingerprint lineage), `runs`, `check_results`, `findings` (indexed by dataset, run, dimension, severity, code, subject), `profiles`, `artifacts`.
- `ArtifactStore` interface backed by **fsspec**: local `DSBF_HOME` in Phase 6, S3/GCS/Azure in Phase 8.
- Each run stores a **Parquet snapshot of the data as profiled** (sampled where applicable, size-capped and configurable), so the dashboard's sample and pair-plot views show what was actually analysed (F-15).
- `dsbf runs list | show | diff | delete | gc` (F-25).

### 8.3 Service

- **API v2** (FastAPI, OpenAPI-first, generated TS client):
  - `GET /v2/datasets`, `/v2/datasets/{id}/runs`, `/v2/datasets/{id}/history?code=…` (trend of a finding over time)
  - `GET /v2/runs/{id}` (manifest plus health), `/v2/runs/{id}/findings?dimension=&severity=&subject=&page=`, `/v2/runs/{id}/profile/{column}`, `/v2/runs/{id}/checks/{check}` (result without bulky evidence), `/v2/artifacts/{ref}` (evidence on demand, paginated)
  - `GET /v2/runs/compare?base=&head=` (finding-level diff: new, resolved, changed)
  - `POST /v2/jobs` (profile a source with a plan or config; Phase 6 local thread runner, Phase 8 queue), `GET /v2/jobs/{id}` with progress events
- Responses are small and paginated. **The dashboard never downloads everything at once** (F-35: from about 8 MB to under 200 KB on open).
- Non-finite numbers can't reach the API, because they're excluded at the schema level (F-36).
- Auth: none and bound to localhost in Phase 6; pluggable token/OIDC in Phase 8. CORS is configurable.

### 8.4 Dashboard

- It consumes **only** API v2 through generated types, with no hand-written response shapes. Today's 94 hard-coded `tasks.<name>.data` paths across 22 components go away.
- Layout: Overview (health, dimensions, top findings), **Findings explorer** (N-A5), Column view (profile charts from real bins), Relationships, Quality, ML readiness, Time series, Run compare / history, and Settings (edit config overlay → submit job).
- ECharts only. Each tab loads its own data. Pinia store. ESLint, Prettier and Vitest, plus one Playwright smoke test (F-29).
- Shipped inside the `dsbf[server]` wheel and served by `dsbf dashboard` (no Node required for users).

---

## 9. Integrations

All of these are thin wrappers over the library, so they stay cheap to maintain.

```python
# Library (mode 1 / 3)
import dsbf

run = dsbf.profile(
    df,
    packs=["core"],
    checks=["+fuzzy_duplicates", "-normality"],
    target="churned",
    time_index="event_ts",
    sample=dsbf.Sample(rows=1_000_000),
)
run.render_health()  # the §7.1 text; its header needs the run, not just the health
run.findings.filter(severity=">=warn").to_polars()
run.write("out/", sinks=["parquet", "html"])
dsbf.store.save(run)  # optional: make it visible in the dashboard
```

```bash
# CLI (any orchestrator, CI, cron)
dsbf check s3://bucket/events/date=2026-10-15/ --pack core --pack timeseries \
     --config dsbf.yaml --policy policy.yaml --out out/ --sink parquet --sink store
dsbf plan data.parquet --pack core          # dry run: what would run and why
dsbf dashboard                              # local server + UI on the store
dsbf runs diff <base> <head>
dsbf schema export --out schema/            # JSON Schema for downstream consumers
```

| Integration | Phase | Notes |
|---|---|---|
| Python library | 3–5 | Primary surface |
| CLI v2 | 5 | Report-only exit codes; policies opt-in |
| Static HTML report | 5 | Zero-infrastructure sharing |
| `dsbf dashboard` / `dsbf serve` | 6 | Local single-user |
| GitHub Action, pre-commit hook | 8 | Wraps the CLI |
| Airflow operator, Prefect task, Dagster asset check | 8 | Wrap the library; return the Run and push findings to the orchestrator's UI where supported |
| dbt | 8 | Post-hook running `dsbf check` on model outputs, plus documented warehouse-side consumption of `findings` tables |
| Docker image, Helm chart | 8 | Team deployment |
| OpenLineage facets, webhooks, Slack | 8 | Observability ecosystems |
| MCP server | 8 | Agents query runs and findings, and can submit jobs |

---

## 10. Statistical rigor (cross-cutting)

These are delivered as part of the check migrations in Phase 4, not as a separate phase.

- **Evidence is mandatory:** `n_used`, method and provenance on every finding. Confidence intervals where they're defined (Fisher z for correlations, bootstrap for η² and Cramér's V).
- **Bias-corrected Cramér's V;** associations ranked by strength band within type and never by raw value across types (F-25).
- **Multivariate checks choose their own missing-data strategy** and report what they dropped (F-32).
- **Leakage is redefined around a single `target` setting** (F-26). Feature–feature near-duplication moves to Redundancy. Without a target, Leakage reports `not_applicable` with a note rather than green.
- **Heavy checks are redesigned with explicit budgets** and honest partial-coverage reporting: fuzzy duplicates with rapidfuzz and IDF/MinHash blocking (F-35); stationarity on the resampled series with capped lags (F-39).
- **Calibration suite:** planted-truth datasets (leakage, MAR/MNAR, bimodality, near-duplicates, drift) measure false-positive and false-negative rates per check version, run nightly, and show trends in CI artifacts.

---

## 11. Quality engineering (cross-cutting)

| Layer | What | Gate |
|---|---|---|
| Contracts | JSON Schema snapshot; breaking-change detector | Every PR |
| Unit | Each check against small fixtures, **plus the adversarial frame** (odd names, all-null, inf, tz, unicode) | Every PR |
| Golden runs | Validation archetypes plus titanic plus kitchen5k → normalised `findings` snapshot; diffs reviewed and approved per PR | Every PR |
| Parity (migration only) | Legacy task vs. migrated check on the same data; differences must be listed in the PR | Every migration PR |
| Performance | Practice-run benchmarks (`tests/perf/`): 5k, 50k, 1M rows; per-check budgets; fails on a >25% regression | Nightly, plus label-triggered on PRs |
| Calibration | FP/FN rates per check | Nightly |
| Install smoke test | Build the wheel → install in a clean venv → run `dsbf check` **from a temp directory** → `dsbf dashboard` health check | Every PR (fast) |
| Frontend | Vitest, ESLint, `vite build`, one Playwright test against a fixture store | Every PR |
| Types and lint | ruff (`E,W,F,I,UP,B,BLE,SIM,PERF,PTH,RET,ARG,S`), pyrefly (`default` preset, baseline for legacy code) | Every PR |

### Performance budgets (targets, to be confirmed by Phase 2 benchmarks)

| Scenario | Today (measured) | Target |
|---|---|---|
| titanic, standard plan | 80 s | < 5 s |
| 52k × 60, standard plan | 707 s, 2.1 GB peak | < 60 s, < 1 GB |
| 1M × 50, standard plan | not measured | < 3 min on a laptop |
| Profile pass, 52.5k × 60 (type inference, dataset profile, stage; without loading) | 0.44 s, peak 1.06× the loaded data (CI, 2026-10-18) | < 2 s, peak < 2.5× the loaded data (`tests/perf`, `profile-kitchen50k`) |
| Profile pass, 1M × 50 floats | 9.5 s, peak 1.12× the loaded data (CI, 2026-10-18) | peak < 2.5× the loaded data; time recorded (`profile-wide1m`) |
| Dashboard: open a run | 8.2 MB (titanic) | < 200 KB |
| `import dsbf` | 2.0 s | < 0.5 s (lazy imports of heavy extras) |

---

## 12. Phased plan

Each phase ships on its own and leaves `main` releasable. Sizes are relative: S ≈ days,
M ≈ 1–2 weeks, L ≈ 3–5 weeks of focused solo work.

### Phase 0: Stabilise and baseline (S–M)
The goal is a trustworthy baseline to measure the refactor against. This is the minimum on the
**old** code.
- CI green with pinned ruff; Node 24 actions; Dependabot backlog merged (F-01).
- Toolchain aligned with the maintainer's other projects (D11): Poetry → uv, pytest tier markers with a fast gate, pyrefly with a legacy baseline. The frontend moves to pnpm with lint and tests in Phase 1.
- Test isolation through autouse fixtures (F-02). Fix the 5 failing tests (F-03).
- Shared config blocks pass validation (F-04). Revert the NFL config (F-05).
- Fix the package-data path in `reco_engine` with `importlib.resources`, so runs work from any directory (F-31).
- Crash guards only: `summarize_modes`, `normality_tests` log line, VIF zero-row guard (F-33, F-32 crash part).
- Escape `v-html` (F-14). Remove or fix the percentile histogram (F-12). Park the viz branch (A3).
- Clean up repo clutter (F-28). README corrections (F-27).
- **Capture baselines:** golden findings snapshots for all archetypes plus titanic and kitchen5k; `tests/perf/` from the practice-run scripts; the adversarial frame as a fixture.
- **Write ADRs** for decisions D1–D11 ([§13](#13-decisions-adrs)), now in [`docs/adr/`](adr/README.md).

**Exit:** CI green, baselines committed, ADRs merged.

### Phase 1: Contracts (M)
- `dsbf.model`: Subject, Evidence, Finding, Note, CheckResult, Run, RunHealth, Profile models, Config and Params base, Policy schema.
- JSON Schema export, TS generation, and the schema snapshot gate.
- `src/` layout move, `DSBF_HOME`, `importlib.resources`, and the extras skeleton in `pyproject.toml`.

**Exit:** the schema is published in the repo, and a toy check can produce a valid Run.
**Resolves by construction:** F-10, F-36 (schema side), F-20 (status model).

### Phase 2: Data and profile (L)
- `Source` adapters for T1. `Dataset` with fingerprint, sampling provenance and a value-array cache.
- Profiling pass (§5.3), including real histogram bins and sample-based semantic inference.
- Parity tests against today's `summarize_*` and `infer_types` outputs on the baselines.

**Exit:** the profile matches baselines within tolerance (or documented differences), and the perf benchmark shows the profile pass under 2 s at 52k × 60.
**Resolves:** F-08, F-09, F-12, F-30, F-37, F-32 (per-column statistics).

### Phase 3: Engine and check API (L)
- `@check` interface, registry with entry points, packs, planner (`dsbf plan`), executor (isolation, statuses, notes, timeouts, parallelism, cache, events), library-friendly logging.
- `LegacyTaskCheck` adapter: **the full existing task set runs on the new engine** and produces Findings.
- `dsbf.profile()` library entry point (in-memory Run).

**Exit:** the golden runs reproduce through the adapter. A forced exception in any check yields a complete Run with a note. Changing one parameter re-executes only the affected checks.
**Resolves:** F-06, F-07, F-19, F-21, F-22, F-23, F-24, F-34, F-16, F-17, F-18 (core does no I/O).

### Phase 4: Migrate checks (L, 19 units)
Revised in U4-00 (2026-10-19); the units and the full migration map are in [`ROADMAP.md`](../ROADMAP.md#phase-4-migrate-checks).
- **Parallel tracks:** core checks are built in a `core` pack beside the intact legacy pack. A core check names the legacy tasks it supersedes, and those leave the default selection. A superseded task still runs, with its findings dropped, when a remaining legacy task hard-requires it or reads its result undeclared, so no shim feeds legacy scorers from core output. Run time therefore rose until the legacy scorers left the default selection, which `quality_dimensions` and `ml_readiness` made possible in U4-15: they read the core checks' findings through soft `after` edges, and the default selection is the `core` pack alone.
- **Retired, not migrated** (chosen by the user on 2026-10-19): the 13 tasks that restate the profile (the `summarize_*` tasks, `categorical_length_stats`, `missingness_heatmap`, `infer_types`, `sample_head`, `sample_tail`), whose guidance moves to rule checks; the engine-metadata tasks; and the plot tasks. The ML-prep suggestion tasks and `schema_validation` are migrated as checks.
- **4a Rule checks (~20):** move onto the profile. Expected outcome is a large speed-up with near-zero findings diff.
- **4b Analysis checks:** outliers, normality and QQ, associations, contingency, group tests, bimodality, missingness mechanism, mutual information, encoded/regex/mixed types, string anomalies. Apply the statistical fixes (F-11, F-13, confidence intervals, Cramér's V correction) with intended diffs documented per PR.
- **4c Heavy checks, redesigned:** fuzzy duplicates (rapidfuzz, blocking, coverage, top-N artifact; F-35), time-series suite (resampling, capped lags, budgets; F-39), drift checks against reference *profiles*.
- **4d Synthesis:** quality dimensions over findings (report-only by design), ML readiness, target-based leakage (F-26).
- **Calibration suite:** planted-truth datasets with false-positive and false-negative rates per check version (§10); `tests/validation/` is rebuilt on the new engine.
- **Cutover** (chosen by the user on 2026-10-19): a stopgap `dsbf profile` runs `dsbf.profile()` and writes `run.json`. Then the whole legacy stack is deleted: the adapter, the legacy engine and tasks, `plot_factory`, the old store and API v1, and their dependencies (N-A2, F-38). The base install shrinks to the core set. The dashboard does not run from then until Phase 6 rebuilds it.

**Exit:** no legacy tasks remain; the perf budgets for titanic and 52k × 60 are met; calibration suite baselines are recorded; `pip install dsbf` without extras runs `dsbf profile`.

### Phase 5: Policy, sinks, CLI v2. **First release for mode 1** (M)
- Policy engine (§7.2), default report-only.
- JSON/NDJSON, Parquet and HTML sinks.
- CLI v2: `profile`, `check`, `plan`, `schema export`, and the run health summary at the end of every command.
- Docs: "Use DSBF in your pipeline", "Consume DSBF findings in your own dashboard" (Parquet schema guide), "Write a check", "Write a policy".

**Release:** `1.0.0a1` on PyPI.
**Exit:** the clean-venv smoke test passes from a temp directory; an example repo shows DSBF in a GitHub Actions workflow writing Parquet findings.

### Phase 6: Store, service and dashboard v2. **Mode 2, local** (L)
- RunRepository on SQLite with Alembic; ArtifactStore on local fsspec; per-run data snapshot; `dsbf runs …`.
- API v2 plus the local job runner. Generated TS client.
- Dashboard rebuilt tab by tab on API v2 (reusing `DsbfChart`, `chartTheme` and existing card designs where they fit); Run health panel; Findings explorer; Settings → rerun; run compare.
- `dsbf dashboard` serves the bundled frontend. API v1, the old storage and the old DB were already deleted in Phase 4, so nothing is migrated.

**Release:** `1.0.0`.
**Exit:** `pip install "dsbf[server]" && dsbf check data.csv --sink store && dsbf dashboard` works on a clean machine. Opening a run transfers under 200 KB.

### Phase 7: Scale and sources (L)
- T2 larger-than-memory and object storage; T3 Ibis sources with pushdown (start with DuckDB and Postgres, then Snowflake and BigQuery); T4 recurring datasets, history views and profile-based drift; `dsbf watch`/schedule recipes (cron, orchestrators).

**Exit:** 1M × 50 within budget; a 100M-row Parquet glob profiled with a sample at read time and provenance shown; one warehouse source verified end to end.

### Phase 8: Team and cloud (L)
- Postgres repository; S3/GCS/Azure artifacts; pluggable token/OIDC auth; job queue (start with a database-backed queue to avoid new infrastructure, with an adapter interface for Celery, arq or cloud batch).
- Docker image, Helm chart, GitHub Action, Airflow/Prefect/Dagster adapters, dbt recipe, notifiers, OpenLineage, MCP server.

**Exit:** a documented team deployment; each adapter has an example and a CI smoke test.

### Phase 9 and beyond: Modalities and workflow phases
- T5 modalities (text corpora first: it reuses string profiling), domain packs, then modelling-prep (export ML guidance `actions` as sklearn pipelines) and modelling diagnostics (README vision).

### Dependency view

```
P0 ─► P1 ─► P2 ─► P3 ─► P4 ─► P5 (1.0.0a1, mode 1) ─► P6 (1.0.0, mode 2 local) ─► P7 ─► P8
   P4 ends by deleting the legacy stack, so P5 starts after it (revised 2026-10-19)
```

### Where the review fixes land

| Fixes | Planned for | How it stands after Phase 3 (2026-10-19) |
|---|---|---|
| F-01, F-02, F-03, F-04, F-05, F-12, F-14, F-27, F-28, F-31, F-33 | Phase 0 (on current code) | Done, except F-12's dashboard display (Phase 6), F-28's two root scripts and F-31's `eval` and install smoke test (Phase 4 and 5) |
| F-10, F-20, F-36 | Phase 1 (by schema) | Done in the contracts; legacy pair keys and API v1 go in Phase 4 |
| F-08, F-09, F-30, F-32, F-37 | Phase 2 | Done in the data layer and profile; VIF `n_used` (F-32) is a Phase 4 check |
| F-06, F-07, F-16, F-17, F-18, F-19, F-21, F-22, F-23, F-24, F-34 | Phase 3 | Done in the engine; F-07's per-task keys go as each task migrates |
| F-11, F-13, F-26, F-35, F-38, F-39 | Phase 4 | Open; each has a unit |
| F-15, F-25, F-29 | Phase 6 | F-15's run identity is done; the old store and API v1 are deleted in Phase 4, so the rest are design inputs for Phase 6 |

The status of every finding, with its evidence, is recorded in
[`REVIEW_FINDINGS.md`](REVIEW_FINDINGS.md).

Fixes **not** listed for Phase 0 are deliberately **not** done on the old code. They disappear when
that code is replaced. Doing them twice is wasted effort, unless the refactor is abandoned, in which
case the incremental approach in `REVIEW_FINDINGS.md` still applies.

---

## 13. Decisions (ADRs)

All eleven decisions were accepted on 2026-10-16 (D11 was decided on 2026-10-15), without amendment.
Each is recorded in [`docs/adr/`](adr/README.md). The table summarises them; the ADR holds the context, alternatives and consequences.

| # | Decision | Accepted | Main alternative considered | ADR |
|---|---|---|---|---|
| D1 | Layering | Library-first pure core; everything else optional | Keep the engine coupled to storage and files | [0001](adr/0001-library-first-layering.md) |
| D2 | Primary output | `Finding` with typed evidence; bulky outputs as artifacts | Free-form per-task `data` dicts (today) | [0002](adr/0002-findings-as-primary-output.md) |
| D3 | Data representation | Polars/Arrow internally, NumPy at the statistics boundary, pandas at edges | Standardise on pandas (the right call for incremental fixes, not for a rebuild) | [0003](adr/0003-polars-arrow-data-representation.md) |
| D4 | Warehouse pushdown | Ibis behind the `Source`/profile interface (T3) | Hand-written SQL per dialect; DuckDB-only federation | [0004](adr/0004-ibis-warehouse-pushdown.md) |
| D5 | Contracts | pydantic v2, JSON Schema, generated TS, semver `schema_version` | Dataclasses plus hand-written TS | [0005](adr/0005-pydantic-contracts.md) |
| D6 | Failure semantics | Report-only, isolated checks, explicit statuses and notes; opt-in policies | Fail-fast; built-in default gate | [0006](adr/0006-report-only-failure-semantics.md) |
| D7 | Plugins | Entry points plus local dirs; same contract as core | Import side effects (today) | [0007](adr/0007-entry-point-plugins.md) |
| D8 | Storage | SQLAlchemy Core plus Alembic; SQLite then Postgres; fsspec artifacts | Raw sqlite3 (today); DuckDB as the store | [0008](adr/0008-sqlalchemy-alembic-fsspec-storage.md) |
| D9 | Packaging | Single distribution with extras, `src/` layout | Multiple packages (`dsbf-core`, `dsbf-server`) | [0009](adr/0009-single-distribution-with-extras.md) |
| D10 | Naming | Rename to "check" outright; no alias, since there are no external users (A2) | Keep "task" | [0010](adr/0010-check-naming.md) |
| D11 | Toolchain | uv (dependencies and builds), pytest tier markers, pyrefly, pnpm with ESLint and Vitest. Decided 2026-10-15 to match the maintainer's other projects | Poetry and npm, no type checker | [0011](adr/0011-toolchain.md) |

---

## 14. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Statistical nuance lost in migration | Silent quality regressions, which is DSBF's core value | Phase 0 golden runs; parity tests; intended diffs listed per PR; calibration suite |
| Scope too large for one maintainer | Stalls halfway, leaving two systems | Strangler pattern: each phase ships; the legacy adapter keeps everything running; Phase 5 delivers user value before the server work |
| The old dashboard breaks before the new one exists | Mode 2 users lose the UI | Accepted on 2026-10-19: there are no external users (A2), so the old stack is deleted at the end of Phase 4. Until Phase 6, the stopgap `dsbf profile` (Phase 4) and the static HTML report (Phase 5) are the ways to read a run |
| Polars or Ibis API churn | Maintenance load | Pin minor versions; adapters isolate them; Ibis only behind `dsbf[sql]` |
| Over-engineering team features early | Slows local delivery | Interfaces only until Phase 8; no Postgres or auth code before then |
| The schema evolves too fast for mode-1 consumers | Downstream breakage | Semver plus a schema diff gate; additive-only within 1.x; deprecation window of one minor release |
| Performance targets unmet for some checks | Slow standard runs | Cost classes; heavy checks off in `quick`/`standard` presets; per-check budgets with partial-coverage notes |

---

## 15. Execution

This plan is executed as **units**, each a single branch, PR and commit that can be finished in
one session. The backlog is [`ROADMAP.md`](../ROADMAP.md), the active unit lives in
[`PLAN.md`](../PLAN.md), and the workflow is in [`CLAUDE.md`](../CLAUDE.md). Phases 0–3 were
broken down in full and are finished. Phase 4 was broken down in full by U4-00, and Phase 5 into outline units.
Each later phase starts with a planning unit that breaks it down using what the earlier phases
taught. Finished programs are summarised in [`HISTORY.md`](HISTORY.md).
