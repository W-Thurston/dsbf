// What the quality and ML-readiness outputs say, as the words and badge classes the tab
// shows. A dimension that was not assessed has no level and its own neutral state: it is
// never drawn as green.

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
