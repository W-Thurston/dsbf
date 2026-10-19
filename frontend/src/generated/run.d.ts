/* eslint-disable */
/**
 * Generated from schema/run.schema.json by `dsbf schema export`.
 * Do not edit by hand: change the models in dsbf.model and regenerate.
 */

/**
 * How one check's execution ended. Every status but `ok` comes with a note.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "CheckStatus".
 */
export type CheckStatus = "ok" | "not_applicable" | "unavailable" | "skipped" | "error" | "timeout";
/**
 * The aspect of the data a finding speaks to.
 *
 * The first five are the quality dimensions; the rest describe the shape of
 * the data.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Dimension".
 */
export type Dimension =
  "completeness" | "validity" | "usability" | "redundancy" | "leakage" | "distribution" | "relationship" | "temporal";
/**
 * How much a finding matters, from informational to blocking.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Severity".
 */
export type Severity = "info" | "notice" | "warn" | "error";
/**
 * How firmly a check stands behind a finding; each check caps its own.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Confidence".
 */
export type Confidence = "low" | "moderate" | "high";
/**
 * What a rule does when it matches.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "ActionKind".
 */
export type ActionKind = "annotate" | "warn" | "fail" | "notify";
/**
 * How a comparison relates the observed value to the one in the rule.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Operator".
 */
export type Operator = "==" | "!=" | ">=" | ">" | "<=" | "<";

/**
 * Everything one execution produced, in one serialisable object.
 */
export interface Run {
  schema_version: string;
  run_id: string;
  started_at: string;
  finished_at: string;
  dataset: DatasetManifest;
  plan: PlanManifest;
  profile: DatasetProfile;
  /**
   * One result per planned check, keyed by check id.
   */
  results: {
    [k: string]: CheckResult;
  };
  health: RunHealth;
  /**
   * What happened to the run as a whole rather than to one check: a plugin that failed to load, an empty plan, a result cache that failed.
   */
  notes: Note[];
  /**
   * What the user's policy decided; absent when none was applied.
   */
  policy: PolicyOutcome | null;
}
/**
 * The identity of the data a run looked at.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "DatasetManifest".
 */
export interface DatasetManifest {
  name: string;
  modality: "tabular";
  source: SourceRef;
  /**
   * A content hash: equal data, equal fingerprint.
   */
  fingerprint: string;
  /**
   * Rows in the source, before any sampling.
   */
  n_rows: number;
  n_columns: number;
  /**
   * Absent when the whole dataset was used.
   */
  sampling: SamplingInfo | null;
}
/**
 * Where the data came from. An in-memory frame has no location or format.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "SourceRef".
 */
export interface SourceRef {
  kind: "dataframe" | "file" | "glob" | "sql" | "table";
  /**
   * A path, glob, URL or table name.
   */
  location: string | null;
  /**
   * For example `csv`.
   */
  format: string | null;
}
/**
 * How a sample was drawn from the data. No sampling is the absence of this.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "SamplingInfo".
 */
export interface SamplingInfo {
  strategy: "head" | "random" | "stratified";
  original_rows: number;
  sampled_rows: number;
  seed: number | null;
  stratify_by: string[];
}
/**
 * The checks a run was asked to perform, and what produced the plan.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "PlanManifest".
 */
export interface PlanManifest {
  dsbf_version: string;
  /**
   * The depth preset, if any.
   */
  depth: string | null;
  checks: PlannedCheck[];
}
/**
 * One check the plan includes, with the parameters it will run under.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "PlannedCheck".
 */
export interface PlannedCheck {
  check_id: string;
  check_version: string;
  pack: string | null;
  /**
   * The resolved parameters.
   */
  params: {
    [k: string]: unknown;
  };
  /**
   * The planned checks whose results this one needs.
   */
  depends_on: string[];
}
/**
 * The profile of a whole dataset: its shape and one profile per column.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "DatasetProfile".
 */
export interface DatasetProfile {
  /**
   * Rows profiled, after any sampling.
   */
  n_rows: number;
  approx_memory_bytes: number | null;
  /**
   * Rows equal to an earlier row; None if not computed.
   */
  duplicate_rows: number | null;
  rows_with_any_null: number | null;
  /**
   * Null co-occurrence; None if not computed.
   */
  missingness: Missingness | null;
  columns: (
    | NumericProfile
    | CategoricalProfile
    | BooleanProfile
    | DatetimeProfile
    | TextProfile
    | IdentifierProfile
    | UnknownProfile
  )[];
}
/**
 * Which columns go missing together.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Missingness".
 */
export interface Missingness {
  columns_with_nulls: number;
  /**
   * Columns paired: the ones with the most nulls, when there are more than the pairing limit.
   */
  columns_considered: number;
  /**
   * The strongest co-occurring pairs, most similar first; only pairs that share at least one null row.
   */
  pairs: NullPair[];
}
/**
 * Two columns that are null in the same rows.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "NullPair".
 */
export interface NullPair {
  /**
   * The earlier column in the dataset's column order.
   */
  a: string;
  b: string;
  /**
   * Rows where both columns are null.
   */
  both_null: number;
  /**
   * Rows null in both over rows null in either: 1 means the columns are null in exactly the same rows.
   */
  jaccard: number;
}
/**
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "NumericProfile".
 */
export interface NumericProfile {
  name: string;
  /**
   * Zero-based place in the dataset.
   */
  position: number;
  /**
   * The storage type, as a string.
   */
  physical_dtype: string;
  /**
   * How firmly the semantic type is held.
   */
  semantic_confidence: Confidence;
  /**
   * Rows the semantic type was inferred from.
   */
  semantic_sample_rows: number | null;
  counts: ColumnCounts;
  semantic_type: "numeric";
  zeros: number;
  /**
   * NaN and infinite values, counted apart from nulls.
   */
  non_finite: number;
  /**
   * None when there is no finite value to summarise.
   */
  stats: NumericStats | null;
  histogram: Histogram | null;
}
/**
 * How many rows a column has, how many are null, and how many values differ.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "ColumnCounts".
 */
export interface ColumnCounts {
  /**
   * Rows profiled, after any sampling.
   */
  rows: number;
  nulls: number;
  /**
   * Distinct non-null values.
   */
  distinct: number;
  /**
   * False when `distinct` is an estimate.
   */
  distinct_exact: boolean;
}
/**
 * Summary statistics of a numeric column's finite, non-null values.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "NumericStats".
 */
export interface NumericStats {
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
  skewness: number | null;
  kurtosis: number | null;
  /**
   * Statistics that were NaN or infinite, so they are null here.
   */
  non_finite_stats: string[];
  /**
   * In increasing order of `p`.
   */
  quantiles: Quantile[];
  quantile_method: ("exact" | "tdigest") | null;
}
/**
 * The value below which a share ``p`` of the non-null values fall.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Quantile".
 */
export interface Quantile {
  p: number;
  value: number;
}
/**
 * Equal-width bins over a numeric column, with the rule that chose them.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Histogram".
 */
export interface Histogram {
  /**
   * One more edge than bins.
   */
  edges: number[];
  /**
   * Values per bin; the last is closed.
   */
  counts: number[];
  /**
   * The rule that proposed the number of bins; a count over the profile's bin limit is clipped to it.
   */
  rule: "fd" | "sturges" | "scott" | "sqrt" | "fixed";
}
/**
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "CategoricalProfile".
 */
export interface CategoricalProfile {
  name: string;
  /**
   * Zero-based place in the dataset.
   */
  position: number;
  /**
   * The storage type, as a string.
   */
  physical_dtype: string;
  /**
   * How firmly the semantic type is held.
   */
  semantic_confidence: Confidence;
  /**
   * Rows the semantic type was inferred from.
   */
  semantic_sample_rows: number | null;
  counts: ColumnCounts;
  semantic_type: "categorical";
  top_values: ValueCount[];
  lengths: LengthStats | null;
  empty_strings: number;
  /**
   * Present for string columns only.
   */
  shape: TextShape | null;
}
/**
 * One value and how many rows hold it.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "ValueCount".
 */
export interface ValueCount {
  value: boolean | number | string;
  count: number;
}
/**
 * Lengths, in characters, of a column's non-null strings.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "LengthStats".
 */
export interface LengthStats {
  min: number;
  max: number;
  mean: number;
}
/**
 * What the characters of a string column's non-empty values look like.
 *
 * Every count is of non-null, non-empty values, so none exceeds the number of those.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "TextShape".
 */
export interface TextShape {
  /**
   * Values with at least one letter.
   */
  with_letters: number;
  /**
   * Values with at least one digit.
   */
  with_digits: number;
  /**
   * Values with a space or tab.
   */
  with_whitespace: number;
  /**
   * Values with a character that is not a word or space.
   */
  with_symbols: number;
  /**
   * Values that start or end with whitespace.
   */
  edge_whitespace: number;
  /**
   * Distinct pattern signatures.
   */
  n_patterns: number;
  /**
   * The most common signatures, by decreasing count.
   */
  patterns: PatternCount[];
}
/**
 * A pattern signature and how many values have it.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "PatternCount".
 */
export interface PatternCount {
  /**
   * Each run of uppercase letters is `A`, of lowercase letters `a` and of digits `9`; other characters stand for themselves.
   */
  pattern: string;
  count: number;
}
/**
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "BooleanProfile".
 */
export interface BooleanProfile {
  name: string;
  /**
   * Zero-based place in the dataset.
   */
  position: number;
  /**
   * The storage type, as a string.
   */
  physical_dtype: string;
  /**
   * How firmly the semantic type is held.
   */
  semantic_confidence: Confidence;
  /**
   * Rows the semantic type was inferred from.
   */
  semantic_sample_rows: number | null;
  counts: ColumnCounts;
  semantic_type: "boolean";
  true_count: number;
  false_count: number;
}
/**
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "DatetimeProfile".
 */
export interface DatetimeProfile {
  name: string;
  /**
   * Zero-based place in the dataset.
   */
  position: number;
  /**
   * The storage type, as a string.
   */
  physical_dtype: string;
  /**
   * How firmly the semantic type is held.
   */
  semantic_confidence: Confidence;
  /**
   * Rows the semantic type was inferred from.
   */
  semantic_sample_rows: number | null;
  counts: ColumnCounts;
  semantic_type: "datetime";
  min: string | null;
  max: string | null;
  timezone: string | null;
  /**
   * An offset alias such as `D`, when the spacing is regular.
   */
  inferred_frequency: string | null;
  n_gaps: number | null;
  largest_gap_seconds: number | null;
}
/**
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "TextProfile".
 */
export interface TextProfile {
  name: string;
  /**
   * Zero-based place in the dataset.
   */
  position: number;
  /**
   * The storage type, as a string.
   */
  physical_dtype: string;
  /**
   * How firmly the semantic type is held.
   */
  semantic_confidence: Confidence;
  /**
   * Rows the semantic type was inferred from.
   */
  semantic_sample_rows: number | null;
  counts: ColumnCounts;
  semantic_type: "text";
  top_values: ValueCount[];
  lengths: LengthStats | null;
  /**
   * Mean characters per whitespace-delimited word.
   */
  avg_word_length: number | null;
  empty_strings: number;
  shape: TextShape | null;
}
/**
 * A column that names rows (for example a key); only the common fields apply.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "IdentifierProfile".
 */
export interface IdentifierProfile {
  name: string;
  /**
   * Zero-based place in the dataset.
   */
  position: number;
  /**
   * The storage type, as a string.
   */
  physical_dtype: string;
  /**
   * How firmly the semantic type is held.
   */
  semantic_confidence: Confidence;
  /**
   * Rows the semantic type was inferred from.
   */
  semantic_sample_rows: number | null;
  counts: ColumnCounts;
  semantic_type: "identifier";
}
/**
 * A column whose meaning could not be inferred.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "UnknownProfile".
 */
export interface UnknownProfile {
  name: string;
  /**
   * Zero-based place in the dataset.
   */
  position: number;
  /**
   * The storage type, as a string.
   */
  physical_dtype: string;
  /**
   * How firmly the semantic type is held.
   */
  semantic_confidence: Confidence;
  /**
   * Rows the semantic type was inferred from.
   */
  semantic_sample_rows: number | null;
  counts: ColumnCounts;
  semantic_type: "unknown";
}
/**
 * What running one check produced.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "CheckResult".
 */
export interface CheckResult {
  check_id: string;
  check_version: string;
  status: CheckStatus;
  /**
   * Always present when the status is not `ok`.
   */
  notes: Note[];
  findings: Finding[];
  /**
   * The check's own output; bulky parts live elsewhere.
   */
  output: {
    [k: string]: unknown;
  } | null;
  /**
   * Keys of `output` holding a NaN, infinite or NaT value, now null.
   */
  non_finite_outputs: string[];
  timing: Timing;
  /**
   * The resolved parameters, for reproducibility.
   */
  params: {
    [k: string]: unknown;
  };
}
/**
 * An operational message: why a check was skipped, failed or degraded.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Note".
 */
export interface Note {
  /**
   * Machine-readable, for example `dependency_failed`.
   */
  code: string;
  /**
   * What happened, in a sentence.
   */
  message: string;
  severity: Severity;
  /**
   * What the reader can do about it.
   */
  hint: string | null;
  /**
   * The checks whose outcome caused a skip.
   */
  blocked_by: string[];
  exception_type: string | null;
  /**
   * Technical detail, for example a traceback summary.
   */
  detail: string | null;
}
/**
 * One statement about the data: what, where, how much, and why to believe it.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Finding".
 */
export interface Finding {
  /**
   * Stable across runs for the same check, subject and code.
   */
  id: string;
  check_id: string;
  check_version: string;
  /**
   * Machine-readable kind, for example `high_null_rate`.
   */
  code: string;
  subject: Subject;
  dimension: Dimension;
  severity: Severity;
  confidence: Confidence;
  title: string;
  /**
   * Self-contained, with the observed values.
   */
  message: string;
  evidence: Evidence;
  guidance: Guidance | null;
  tags: string[];
}
/**
 * The thing a finding or result refers to.
 *
 * Columns are kept as an ordered tuple of opaque strings, never joined into a
 * single key, so names containing ``|``, brackets, unicode or nothing at all
 * stay unambiguous.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Subject".
 */
export interface Subject {
  kind: "dataset" | "column" | "column_pair" | "column_group" | "row_set";
  columns: string[];
}
/**
 * The numbers behind a finding: what was measured, on how much data, and how.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Evidence".
 */
export interface Evidence {
  metrics: {
    [k: string]:
      | boolean
      | number
      | string
      | (boolean | number | string | null)[]
      | {
          [k: string]: boolean | number | string | null;
        }
      | null;
  };
  /**
   * Metrics whose value was NaN, infinite or NaT, so they are null in `metrics`.
   */
  non_finite_metrics: string[];
  n_used: number;
  n_total: number;
  /**
   * For example `spearman_r` or `vif`.
   */
  method: string;
  assumptions: AssumptionCheck[];
  interval: Interval | null;
  sampled: SamplingInfo | null;
  /**
   * Pointer to bulky evidence, fetched on demand.
   */
  details_ref: string | null;
}
/**
 * Whether a statistical assumption behind a result held.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "AssumptionCheck".
 */
export interface AssumptionCheck {
  name: string;
  /**
   * None when the assumption could not be assessed.
   */
  satisfied: boolean | null;
  method: string | null;
  detail: string | null;
}
/**
 * A confidence interval for a statistic.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Interval".
 */
export interface Interval {
  low: number;
  high: number;
  /**
   * Coverage, for example 0.95.
   */
  level: number;
  method: string | null;
}
/**
 * What to do about a finding, from one point of view.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Guidance".
 */
export interface Guidance {
  /**
   * `eda` describes the data as it is; `ml` says what to do before modelling.
   */
  phase: "eda" | "ml";
  actions: Action[];
}
/**
 * A step a person or an agent can take, in a form a program can read.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Action".
 */
export interface Action {
  /**
   * The verb, for example `transform`.
   */
  action: string;
  /**
   * The columns the step applies to; empty for the dataset.
   */
  columns: string[];
  /**
   * For example `log1p`.
   */
  method: string | null;
  /**
   * When the step applies, in words.
   */
  condition: string | null;
  /**
   * What the step does, in words.
   */
  detail: string | null;
  /**
   * Anything else the step needs, for example `from_dtype`.
   */
  params: {
    [k: string]: boolean | number | string | null;
  };
}
/**
 * Wall and CPU time and memory for one check, as measured by the executor.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Timing".
 */
export interface Timing {
  /**
   * Time the executor spent on the check, from its start.
   */
  wall_seconds: number;
  /**
   * CPU time while the check ran; approximate. For a check on a thread this is the whole process's, so work on other threads counts too, and `shared_with` says whether checks overlapped. For a check in a worker process it is the worker's own, and missing if the worker was killed or died.
   */
  cpu_seconds: number | null;
  /**
   * Peak resident memory while the check ran, found by sampling; approximate, and a short spike between samples is missed. For a check on a thread it is the whole process's, including the loaded data, and shared with any check that overlapped (`shared_with`). For a check in a worker process it is the worker's own, including what it mapped of the data.
   */
  peak_memory_bytes: number | null;
  /**
   * How many other checks on threads ran at the same time as this one. While any did, `cpu_seconds` and `peak_memory_bytes` are the whole process's and cannot be told apart per check. Always 0 for a check in a worker process.
   */
  shared_with: number;
  /**
   * True when the result came from the result cache: the check did not run in this run, and the timing is that of reading the entry.
   */
  cached: boolean;
  started_at: string | null;
}
/**
 * Counts and reasons by status, so a failure is never found by reading logs.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "RunHealth".
 */
export interface RunHealth {
  counts: StatusCounts;
  /**
   * Checks that errored, timed out, were skipped or were unavailable.
   */
  issues: HealthIssue[];
  findings_at_warn_or_above: number;
}
/**
 * How many checks ended in each status.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "StatusCounts".
 */
export interface StatusCounts {
  ok: number;
  not_applicable: number;
  unavailable: number;
  skipped: number;
  error: number;
  timeout: number;
}
/**
 * One check that did not complete normally, and why.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "HealthIssue".
 */
export interface HealthIssue {
  check_id: string;
  status: CheckStatus;
  message: string;
  hint: string | null;
  blocked_by: string[];
}
/**
 * Why a pipeline was or was not failed: every rule and what it matched.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "PolicyOutcome".
 */
export interface PolicyOutcome {
  /**
   * One entry per policy rule, in policy order, fired or not.
   */
  rules: RuleOutcome[];
  /**
   * A triggered rule has a `fail` action.
   */
  failed: boolean;
}
/**
 * What one rule matched in a run.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "RuleOutcome".
 */
export interface RuleOutcome {
  rule: Rule;
  /**
   * The findings a finding rule matched.
   */
  finding_ids: string[];
  /**
   * The checks a health rule matched.
   */
  check_ids: string[];
}
/**
 * When findings or checks match, take these actions.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Rule".
 */
export interface Rule {
  /**
   * Unique within the policy.
   */
  name: string;
  when: Matcher;
  /**
   * One action or a list. Written `fail`, `{notify: scheme:destination}` or a list of those; stored as `{kind, target}` entries.
   *
   * @minItems 1
   */
  then: [PolicyAction, ...PolicyAction[]];
}
/**
 * The ``when`` of a rule: every key given must hold, and at least one is needed.
 *
 * Finding keys select findings; ``health.status`` selects checks that ended in that
 * status and may be narrowed with ``check`` only. A bare value means equality;
 * ``severity``, ``confidence`` and ``metric`` values may carry an operator
 * (``">=error"``, ``">0.3"``), which must be quoted in YAML.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "Matcher".
 */
export interface Matcher {
  dimension: Dimension | null;
  /**
   * A finding's machine-readable kind, for example `high_null_rate`.
   */
  code: string | null;
  /**
   * The id of the check concerned.
   */
  check: string | null;
  /**
   * A column the finding's subject names; a pair matches on either.
   */
  column: string | null;
  /**
   * A tag the finding carries.
   */
  tag: string | null;
  severity: SeverityComparison | null;
  confidence: ConfidenceComparison | null;
  /**
   * Evidence metric name to comparison; every entry must hold.
   */
  metric: {
    [k: string]: MetricComparison;
  } | null;
  /**
   * Only findings absent from the previous run; needs one to compare.
   */
  new_since: "previous_run" | null;
  /**
   * Only checks that ended in this status.
   */
  "health.status": CheckStatus | null;
}
/**
 * Compare a finding's severity, for example ``>=error``.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "SeverityComparison".
 */
export interface SeverityComparison {
  op: Operator;
  value: Severity;
}
/**
 * Compare a finding's confidence, for example ``>=moderate``.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "ConfidenceComparison".
 */
export interface ConfidenceComparison {
  op: Operator;
  value: Confidence;
}
/**
 * Compare one number in a finding's evidence, for example ``>0.3``.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "MetricComparison".
 */
export interface MetricComparison {
  op: Operator;
  value: boolean | number | string;
}
/**
 * One step a matched rule takes. Only `fail` affects the exit code.
 *
 * This interface was referenced by `Run`'s JSON-Schema
 * via the `definition` "PolicyAction".
 */
export interface PolicyAction {
  kind: ActionKind;
  /**
   * Where `notify` sends it, as `scheme:destination`.
   */
  target: string | null;
}
