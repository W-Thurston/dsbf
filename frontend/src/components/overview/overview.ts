// What the Overview derives from a run's columns and dataset facts. Pure, so a test reads
// them without a page.
import type { components } from '../../generated/api'
import { formatCount } from '../../format'
import type { Mark } from '../shell/health'

type Column = components['schemas']['ColumnSummary']
type SemanticType = Column['semantic_type']

/** A table cell's tone: `none` is plain, the others are the level colours. */
export type Tone = 'none' | 'amber' | 'red'

/** Missing share: amber from 5%, red from 20% (the old dashboard's cut-offs). */
export function missingTone(share: number): Tone {
  if (share >= 0.2) return 'red'
  if (share >= 0.05) return 'amber'
  return 'none'
}

/** Share held by the most frequent value: amber from 70%, red from 90%. */
export function modeTone(share: number | null): Tone {
  if (share === null) return 'none'
  if (share >= 0.9) return 'red'
  if (share >= 0.7) return 'amber'
  return 'none'
}

/** Skewness beyond ±1 is marked, beyond ±2 is serious. */
export function skewTone(skew: number | null): Tone {
  if (skew === null) return 'none'
  if (Math.abs(skew) > 2) return 'red'
  if (Math.abs(skew) > 1) return 'amber'
  return 'none'
}

export function nullShare(column: Column): number {
  const { nulls, rows } = column.counts
  return rows === 0 ? 0 : nulls / rows
}

export function percent(share: number): string {
  if (share === 0) return '0%'
  if (share < 0.001) return '<0.1%'
  return `${(share * 100).toFixed(share < 0.1 ? 1 : 0)}%`
}

const BYTES = ['B', 'KB', 'MB', 'GB', 'TB']

export function formatBytes(bytes: number): string {
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < BYTES.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${unit === 0 ? value : value.toFixed(value < 10 ? 1 : 0)} ${BYTES[unit]}`
}

/** A number as the table shows it: grouped when whole, four figures otherwise. */
export function formatFigure(value: number | null): string {
  if (value === null) return '–'
  if (Number.isInteger(value)) return formatCount(value)
  return Number(value.toPrecision(4)).toString()
}

export interface Bar {
  label: string
  /** The bar's height as a share of the tallest, 0 to 1. */
  height: number
}

/** The bars of a row's inline chart: the histogram bins, or the top values' counts. */
export function sketchBars(column: Column): Bar[] {
  const { bins, top_values } = column.sketch
  if (bins.length) {
    const tallest = Math.max(...bins)
    return bins.map((count, i) => ({
      label: `${formatCount(count)} values in bin ${i + 1} of ${bins.length}`,
      height: tallest === 0 ? 0 : count / tallest,
    }))
  }
  if (top_values.length) {
    const tallest = Math.max(...top_values.map((v) => v.count))
    return top_values.map((v) => ({
      label: `${String(v.value)}: ${formatCount(v.count)}`,
      height: tallest === 0 ? 0 : v.count / tallest,
    }))
  }
  return []
}

export type ColumnSort = 'position' | 'name' | 'type' | 'missing' | 'unique' | 'mode' | 'skew'

/** Value a column sorts by; `null` sorts last whatever the direction. */
export function sortKey(column: Column, by: ColumnSort): string | number | null {
  switch (by) {
    case 'position':
      return column.position
    case 'name':
      return column.name.toLowerCase()
    case 'type':
      return column.semantic_type
    case 'missing':
      return nullShare(column)
    case 'unique':
      return column.counts.distinct
    case 'mode':
      return column.sketch.mode_share
    case 'skew':
      return column.sketch.skewness === null ? null : Math.abs(column.sketch.skewness)
  }
}

export function sortColumns(columns: Column[], by: ColumnSort, descending: boolean): Column[] {
  const direction = descending ? -1 : 1
  return [...columns].sort((a, b) => {
    const x = sortKey(a, by)
    const y = sortKey(b, by)
    if (x === null || y === null) return x === y ? 0 : x === null ? 1 : -1
    return x < y ? -direction : x > y ? direction : a.position - b.position
  })
}

/** Columns per semantic type, most common first. */
export function typeCounts(columns: Column[]): { label: SemanticType; count: number }[] {
  const counts = new Map<SemanticType, number>()
  for (const column of columns)
    counts.set(column.semantic_type, (counts.get(column.semantic_type) ?? 0) + 1)
  return [...counts]
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
}

/** The level colour a severity takes in a stacked bar; `info` is neutral. */
export const SEVERITY_MARK: Record<string, Mark> = {
  info: 'none',
  notice: 'blue',
  warn: 'amber',
  error: 'red',
}

/** The level colour a check status takes in the run-health bar. */
export const STATUS_MARK: Record<string, Mark> = {
  ok: 'green',
  not_applicable: 'none',
  unavailable: 'amber',
  skipped: 'amber',
  error: 'red',
  timeout: 'red',
}

export const MAX_HEAT_COLUMNS = 12
