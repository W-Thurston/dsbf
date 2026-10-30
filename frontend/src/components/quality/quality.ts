// What the quality and ML-readiness outputs say, as the words and badge classes the tab
// shows. A dimension that was not assessed has no level and its own neutral state: it is
// never drawn as green.

import { severityRank, type SeverityName } from '../../generated/ranks'

export type Level = 'green' | 'blue' | 'amber' | 'red'
export type Gate = 'ready' | 'needs_work' | 'not_ready' | 'not_assessed'
export type InputState = 'assessed' | 'nothing_to_check' | 'needs_setting' | 'not_run' | 'failed'

export interface InputReport {
  check_id: string
  state: InputState
  findings: number
  reason: string | null
}

export interface FindingRef {
  id: string
  check_id: string
  code: string
  severity: string
  columns: string[]
  title: string
  actions?: { action: string; method?: string | null; columns?: string[] }[]
}

export interface QualityDimension {
  assessed: boolean
  level: Level | null
  coverage: number
  inputs: InputReport[]
  affected_columns: string[]
  affected_count: number
  pct_affected: number
  findings: FindingRef[]
}

export interface QualityOutput {
  total_columns: number
  dimensions: Record<string, QualityDimension>
  clean_columns: string[]
}

export interface ReadinessDimension {
  label: string
  assessed: boolean
  level: 'green' | 'amber' | 'red' | null
  coverage: number
  inputs: InputReport[]
  action_columns: string[]
  advisory_columns: string[]
  findings: FindingRef[]
}

export interface ReadinessOutput {
  gate: Gate
  complete: boolean
  unassessed: string[]
  total_columns: number
  dimensions: Record<string, ReadinessDimension>
  action_columns: string[]
  advisory_columns: string[]
  clean_columns: string[]
}

const BADGE: Record<Level, string> = {
  green: 'badge-ok',
  blue: 'badge-notice',
  amber: 'badge-warn',
  red: 'badge-error',
}

const LEVEL_WORDS: Record<Level, string> = {
  green: 'All clear',
  blue: 'Minor',
  amber: 'Warning',
  red: 'Serious',
}

export function levelClass(level: Level | null): string {
  return level ? BADGE[level] : 'badge-info'
}

export function levelWords(level: Level | null): string {
  return level ? LEVEL_WORDS[level] : 'Not assessed'
}

const GATE_WORDS: Record<Gate, string> = {
  ready: 'Ready for modelling',
  needs_work: 'Needs work before modelling',
  not_ready: 'Not ready for modelling',
  not_assessed: 'Not assessed',
}

const GATE_LEVEL: Record<Gate, Level | null> = {
  ready: 'green',
  needs_work: 'amber',
  not_ready: 'red',
  not_assessed: null,
}

export function gateWords(gate: Gate): string {
  return GATE_WORDS[gate]
}

export function gateClass(gate: Gate): string {
  return levelClass(GATE_LEVEL[gate])
}

/** Why a gate of `ready` or `needs_work` may be short of the whole picture. */
export function gateCaveat(output: ReadinessOutput, labels: Record<string, string>): string {
  if (output.gate === 'not_assessed') return 'No readiness dimension could be assessed.'
  if (output.complete) return ''
  const names = output.unassessed.map((key) => labels[key] ?? key).join(', ')
  return `Holds only for the dimensions that were assessed; not assessed: ${names}.`
}

const INPUT_WORDS: Record<InputState, string> = {
  assessed: 'Read',
  nothing_to_check: 'Nothing to check',
  needs_setting: 'Needs a setting',
  not_run: 'Not run',
  failed: 'Did not complete',
}

export function inputWords(state: InputState): string {
  return INPUT_WORDS[state]
}

/** The explanation for a dimension with no level, from the inputs that could not be read. */
export function unassessedReason(inputs: InputReport[]): string {
  const reasons = inputs.filter((i) => i.state !== 'assessed' && i.reason).map((i) => i.reason)
  return reasons.length
    ? (reasons[0] as string)
    : 'None of the checks this dimension reads could be assessed.'
}

export function percent(share: number): string {
  return `${Math.round(share * 100)}%`
}

/** One step an `ml` action asks for, in words. */
export function actionWords(action: NonNullable<FindingRef['actions']>[number]): string {
  const where = action.columns?.length ? ` ${action.columns.join(', ')}` : ''
  return `${action.action}${action.method ? ` (${action.method})` : ''}${where}`
}

const LEVEL_ORDER: Level[] = ['green', 'blue', 'amber', 'red']

export interface Trust {
  level: Level | null
  headline: string
  detail: string
}

const TRUST_WORDS: Record<Level, string> = {
  green: 'Looking good',
  blue: 'Minor issues',
  amber: 'Worth investigating',
  red: 'Needs attention',
}

/** The worst level among the assessed dimensions, in words, with the counts behind it. */
export function trustOf(output: QualityOutput): Trust {
  const dimensions = Object.values(output.dimensions)
  const assessed = dimensions.filter((d) => d.assessed && d.level)
  const flagged = assessed.filter((d) => d.level !== 'green').length
  const level = assessed.reduce<Level | null>(
    (worst, d) =>
      !worst || LEVEL_ORDER.indexOf(d.level as Level) > LEVEL_ORDER.indexOf(worst)
        ? (d.level as Level)
        : worst,
    null,
  )
  const counts = `${output.clean_columns.length} of ${output.total_columns} columns fully clean · ${flagged} of ${assessed.length} assessed dimensions flagged`
  return {
    level,
    headline: level ? TRUST_WORDS[level] : 'Not assessed',
    detail: level ? counts : 'No quality dimension could be assessed.',
  }
}

/** How many columns a dimension flags, as the card's summary line. */
export function dimensionSummary(dimension: QualityDimension, total: number): string {
  if (!dimension.assessed) return unassessedReason(dimension.inputs)
  if (!dimension.findings.length) return 'No findings.'
  const columns = dimension.affected_count
  const share = `${percent(dimension.pct_affected)} of ${total}`
  return columns
    ? `${columns} ${columns === 1 ? 'column' : 'columns'} affected (${share}).`
    : `${dimension.findings.length} dataset-level ${dimension.findings.length === 1 ? 'finding' : 'findings'}.`
}

/** The kinds of finding a dimension holds, with how many of each, most common first. */
export function issueChips(findings: FindingRef[]): { code: string; count: number }[] {
  const counts = new Map<string, number>()
  for (const f of findings) counts.set(f.code, (counts.get(f.code) ?? 0) + 1)
  return [...counts]
    .map(([code, count]) => ({ code, count }))
    .sort((a, b) => b.count - a.count || a.code.localeCompare(b.code))
}

export type FindingSort = 'severity' | 'column' | 'finding'

/** The findings in the order a column header asks for; severity puts the worst first. */
export function sortFindings(
  findings: FindingRef[],
  by: FindingSort,
  descending: boolean,
): FindingRef[] {
  const key = (f: FindingRef): string | number =>
    by === 'severity'
      ? severityRank(f.severity as SeverityName)
      : by === 'column'
        ? f.columns.join(', ')
        : f.title
  const sign = descending ? -1 : 1
  return [...findings].sort((a, b) => {
    const x = key(a)
    const y = key(b)
    const order = typeof x === 'number' ? x - (y as number) : x.localeCompare(y as string)
    return sign * order || a.id.localeCompare(b.id)
  })
}

export interface Predictor {
  column: string
  effect: number
  effect_name: 'rank_biserial' | 'cramers_v'
  p_adjusted: number
  flagged: boolean
}

/** A predictor's bar: the size of the effect, and the sign only where the effect has one. */
export function effectBar(p: Pick<Predictor, 'effect' | 'effect_name'>): {
  width: number
  text: string
} {
  const size = Math.min(Math.abs(p.effect), 1)
  const sign = p.effect_name === 'rank_biserial' && p.effect < 0 ? '−' : ''
  return { width: size, text: `${sign}${size.toFixed(2)}` }
}
