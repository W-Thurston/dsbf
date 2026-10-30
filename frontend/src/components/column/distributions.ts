// The Distributions tab's figures, as plain data so a test can read them without a DOM.
import { formatCount } from '../../format'
import type { components } from '../../generated/api'
import { severityRank, type SeverityName } from '../../generated/ranks'

type ColumnSummary = components['schemas']['ColumnSummary']
export type Profile =
  | components['schemas']['NumericProfile']
  | components['schemas']['CategoricalProfile']
  | components['schemas']['BooleanProfile']
  | components['schemas']['DatetimeProfile']
  | components['schemas']['TextProfile']
  | components['schemas']['IdentifierProfile']
  | components['schemas']['UnknownProfile']
type Cell = components['schemas']['RelationshipCell']

const TYPE_ORDER = [
  'numeric',
  'categorical',
  'boolean',
  'datetime',
  'text',
  'identifier',
  'unknown',
]

export interface BrowserColumn {
  name: string
  nullShare: number
  warned: boolean
}
export interface BrowserGroup {
  type: string
  columns: BrowserColumn[]
}

/**
 * The columns whose name holds `search`, grouped by semantic type in a fixed order, each
 * with its null share and whether a finding of `warn` or worse is about it.
 */
export function browserGroups(
  columns: ColumnSummary[],
  warnedColumns: Set<string>,
  search: string,
): BrowserGroup[] {
  const needle = search.trim().toLowerCase()
  const groups = new Map<string, BrowserColumn[]>()
  for (const c of columns) {
    if (!c.name.toLowerCase().includes(needle)) continue
    const list = groups.get(c.semantic_type) ?? []
    list.push({
      name: c.name,
      nullShare: c.counts.rows ? c.counts.nulls / c.counts.rows : 0,
      warned: warnedColumns.has(c.name),
    })
    groups.set(c.semantic_type, list)
  }
  return [...groups]
    .sort(([a], [b]) => TYPE_ORDER.indexOf(a) - TYPE_ORDER.indexOf(b))
    .map(([type, list]) => ({ type, columns: list }))
}

/** Columns a finding of `warn` or worse names. */
export function warnedColumns(
  findings: { severity: SeverityName; subject: { columns: string[] } }[],
): Set<string> {
  const warned = new Set<string>()
  for (const f of findings) {
    if (severityRank(f.severity) < severityRank('warn')) continue
    for (const column of f.subject.columns) warned.add(column)
  }
  return warned
}

function figure(value: number | null | undefined): string {
  if (value === null || value === undefined) return '–'
  return Number.isInteger(value) ? formatCount(value) : String(Number(value.toPrecision(4)))
}

function percent(value: number): string {
  return `${(value * 100).toFixed(value > 0 && value < 0.001 ? 2 : 1)}%`
}

export interface Stat {
  label: string
  value: string
}

/**
 * The strip of a column's headline statistics: a numeric column's moments and spread, a
 * categorical one's cardinality and top value. `outliers` is the IQR count when known.
 */
export function statStrip(profile: Profile, outliers: number | null): Stat[] {
  const { counts } = profile
  const nullShare = counts.rows ? counts.nulls / counts.rows : 0
  const stat = (label: string, value: string): Stat => ({ label, value })
  if (profile.semantic_type === 'numeric') {
    const s = profile.stats
    const q = (p: number) => s?.quantiles.find((x) => x.p === p)?.value
    const iqr = q(0.75) !== undefined && q(0.25) !== undefined ? q(0.75)! - q(0.25)! : null
    return [
      stat('Count', formatCount(counts.rows - counts.nulls)),
      stat('Mean', figure(s?.mean)),
      stat('Median', figure(q(0.5))),
      stat('Std', figure(s?.std)),
      stat('Min', figure(s?.min)),
      stat('Max', figure(s?.max)),
      stat('Skew', figure(s?.skewness)),
      stat('Kurtosis', figure(s?.kurtosis)),
      stat('IQR', figure(iqr)),
      stat('Null', percent(nullShare)),
      stat('Outliers', outliers === null ? '–' : formatCount(outliers)),
    ]
  }
  const top = 'top_values' in profile ? profile.top_values[0] : undefined
  const present = counts.rows - counts.nulls
  return [
    stat('Count', formatCount(present)),
    stat('Unique', formatCount(counts.distinct)),
    stat('Top value', top ? String(top.value) : '–'),
    stat('Top share', top && present ? percent(top.count / present) : '–'),
    stat('Null', percent(nullShare)),
  ]
}

/** A numeric column's quantiles as table rows, lowest first. */
export function percentileRows(profile: components['schemas']['NumericProfile']) {
  return (profile.stats?.quantiles ?? []).map((q) => ({
    label: `${Number((q.p * 100).toFixed(1))}%`,
    value: figure(q.value),
  }))
}

/** A categorical column's listed values with their counts and shares of the non-null rows. */
export function valueCountRows(profile: components['schemas']['CategoricalProfile']) {
  const present = profile.counts.rows - profile.counts.nulls
  return profile.top_values.map((v) => ({
    label: String(v.value),
    count: formatCount(v.count),
    share: present ? percent(v.count / present) : '–',
  }))
}

export interface OutlierColumn {
  n_used: number
  iqr: {
    multiplier: number
    count: number
    share: number
    q1: number
    q3: number
    lower_fence: number
    upper_fence: number
  } | null
  zscore: { count: number; share: number; threshold: number; max_score: number } | null
  mad: { count: number; share: number; threshold: number; max_score: number } | null
  methods_flagging: string[]
  consensus: boolean
}

export interface OutlierRow {
  method: string
  count: number
  share: string
  rule: string
}

/** One row per method that ran, with the rule it applied. */
export function outlierRows(column: OutlierColumn): OutlierRow[] {
  const rows: OutlierRow[] = []
  if (column.iqr) {
    const { iqr } = column
    rows.push({
      method: 'IQR',
      count: iqr.count,
      share: percent(iqr.share),
      rule: `outside ${figure(iqr.lower_fence)} to ${figure(iqr.upper_fence)} (${figure(iqr.multiplier)} × IQR)`,
    })
  }
  if (column.zscore) {
    rows.push({
      method: 'Z-score',
      count: column.zscore.count,
      share: percent(column.zscore.share),
      rule: `|z| above ${figure(column.zscore.threshold)} (largest ${figure(column.zscore.max_score)})`,
    })
  }
  if (column.mad) {
    rows.push({
      method: 'MAD',
      count: column.mad.count,
      share: percent(column.mad.share),
      rule: `modified z above ${figure(column.mad.threshold)} (largest ${figure(column.mad.max_score)})`,
    })
  }
  return rows
}

/** The consensus banner: how many methods agree. */
export function outlierConsensus(column: OutlierColumn): string {
  const flagging = column.methods_flagging.length
  if (column.consensus) return `${flagging} methods flag outliers in this column.`
  if (flagging === 1) return 'One method flags outliers; the others do not agree.'
  return 'No method flags outliers.'
}

export interface NormalityColumn {
  n_used: number
  primary: { test: string; statistic: number; p_value: number; rejects: boolean }
  jarque_bera: { test: string; statistic: number; p_value: number; rejects: boolean }
  non_normal: boolean
}

function pValue(p: number): string {
  return p < 0.001 ? '< 0.001' : p.toFixed(3)
}

/** The verdict and each test's p-value. */
export function normalitySummary(column: NormalityColumn) {
  const tests = [column.primary, column.jarque_bera].map((t) => ({
    name: t.test.replace(/_/g, ' '),
    p: pValue(t.p_value),
    rejects: t.rejects,
  }))
  return {
    verdict: column.non_normal ? 'Not normal' : 'Close to normal',
    tests,
  }
}

export interface BimodalColumn {
  n_finite: number
  bic_delta: number
  separation: number
  means: [number, number]
  stds: [number, number]
  weights: [number, number]
  bimodal: boolean
}

/** The callout's sentence for a bimodal column. */
export function bimodalText(column: BimodalColumn): string {
  const [m1, m2] = column.means.map(figure)
  const [w1, w2] = column.weights.map((w) => percent(w))
  return `Two peaks, around ${m1} (${w1} of values) and ${m2} (${w2}); a second component improves the fit by ${figure(column.bic_delta)} BIC, and the peaks are ${figure(column.separation)} deviations apart.`
}

/**
 * The two fitted components as expected counts at each bin centre, to draw over the
 * histogram: weight × values × bin width × normal density.
 */
export function componentCurves(
  column: BimodalColumn,
  edges: number[],
  counts: number[],
): [number, number][][] {
  const total = counts.reduce((a, b) => a + b, 0)
  return [0, 1].map((k) => {
    const mean = column.means[k]!
    const std = column.stds[k]!
    const weight = column.weights[k]!
    return counts.map((_, i) => {
      const low = edges[i]!
      const high = edges[i + 1]!
      const centre = (low + high) / 2
      const z = (centre - mean) / std
      const density = Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI))
      return [centre, weight * total * (high - low) * density] as [number, number]
    })
  })
}

/** A column's strongest associations with the others, strongest first. */
export function topCorrelations(cells: Cell[], column: string, limit = 5) {
  return cells
    .filter((c) => c.first === column || c.second === column)
    .map((c) => ({
      other: c.first === column ? c.second : c.first,
      statistic: c.statistic,
      value: c.value,
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, limit)
}
