// What the health bar shows for each quality dimension, and which dimensions matter on
// which tab. A dimension nobody assessed is its own grey state, never green.
import {
  levelWords,
  type Level,
  type QualityDimension,
  type QualityOutput,
} from '../quality/quality'

export type DimensionName = 'completeness' | 'validity' | 'usability' | 'redundancy' | 'leakage'
export type Mark = Level | 'none'

export const DIMENSIONS: { name: DimensionName; label: string; definition: string }[] = [
  {
    name: 'completeness',
    label: 'Completeness',
    definition: 'Values that are missing, or columns that are mostly empty.',
  },
  {
    name: 'validity',
    label: 'Validity',
    definition: 'Values that are impossible, malformed or outside their expected range.',
  },
  {
    name: 'usability',
    label: 'Usability',
    definition: 'Columns that carry little or no usable signal, such as constants and identifiers.',
  },
  {
    name: 'redundancy',
    label: 'Redundancy',
    definition: 'Columns, or rows, that repeat information held elsewhere.',
  },
  {
    name: 'leakage',
    label: 'Leakage',
    definition: 'Columns that give away the target. Needs a target to be assessed.',
  },
]

export type Tab =
  'overview' | 'quality' | 'distributions' | 'relationships' | 'ml-readiness' | 'time-series'

/** The tabs in the order a person uses them, with the route each one lives at. */
export const TABS: { tab: Tab; label: string; path: string }[] = [
  { tab: 'overview', label: 'Overview', path: '' },
  { tab: 'quality', label: 'Quality', path: '/quality' },
  { tab: 'distributions', label: 'Distributions', path: '/distributions' },
  { tab: 'relationships', label: 'Relationships', path: '/relationships' },
  { tab: 'ml-readiness', label: 'ML Readiness', path: '/ml-readiness' },
  { tab: 'time-series', label: 'Time Series', path: '/time-series' },
]

const ALL: DimensionName[] = DIMENSIONS.map((d) => d.name)

/** The dimensions a tab speaks to; the health bar dims the others. */
export const RELEVANT: Record<Tab, DimensionName[]> = {
  overview: ALL,
  quality: ALL,
  distributions: ['completeness', 'validity', 'usability'],
  relationships: ['redundancy', 'leakage'],
  'ml-readiness': ALL,
  'time-series': ['completeness', 'validity'],
}

export interface DimensionView {
  name: DimensionName
  label: string
  definition: string
  mark: Mark
  /** "All clear", "N cols" or "Not assessed". */
  summary: string
  /** The hover: the definition and how many columns are affected. */
  detail: string
  relevant: boolean
}

function summaryOf(dimension: QualityDimension): string {
  if (!dimension.assessed) return 'Not assessed'
  if (dimension.level === 'green') return 'All clear'
  // A finding about the dataset as a whole names no column, so say its level instead.
  if (dimension.affected_count === 0) return levelWords(dimension.level)
  return `${dimension.affected_count} ${dimension.affected_count === 1 ? 'col' : 'cols'}`
}

export function healthBar(output: QualityOutput, tab: Tab): DimensionView[] {
  return DIMENSIONS.map(({ name, label, definition }) => {
    const dimension = output.dimensions[name]
    const relevant = RELEVANT[tab].includes(name)
    if (!dimension) {
      return {
        name,
        label,
        definition,
        mark: 'none',
        summary: 'Not assessed',
        detail: `${definition} This run did not report it.`,
        relevant,
      }
    }
    const share = `${dimension.affected_count} of ${output.total_columns} columns (${Math.round(
      dimension.pct_affected * 100,
    )}%)`
    return {
      name,
      label,
      definition,
      mark: dimension.assessed && dimension.level ? dimension.level : 'none',
      summary: summaryOf(dimension),
      detail: dimension.assessed
        ? `${definition} ${share} affected.`
        : `${definition} Not assessed: no check could read it in this run.`,
      relevant,
    }
  })
}
