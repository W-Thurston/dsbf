import type { EChartsOption } from 'echarts'
import type { components } from '../../generated/api'

type Matrix = components['schemas']['RelationshipMatrix']
export type Cell = components['schemas']['RelationshipCell']
export type Statistic = Cell['statistic']
export type PairPoints = components['schemas']['PairPoints']

export const STATISTIC_LABEL: Record<Statistic, string> = {
  pearson_r: 'Pearson r',
  spearman_r: 'Spearman ρ',
  kendalls_tau: 'Kendall τ',
  eta: 'η',
  cramers_v_corrected: 'Cramér’s V (corrected)',
  omega_squared: 'ω²',
}

/**
 * The statistics that share one panel. The three correlations are signed and on one
 * scale; η, V and ω² each measure something else, so each has a panel of its own and
 * no cell is coloured against another statistic's.
 */
export const PANELS: { title: string; statistics: Statistic[]; signed: boolean }[] = [
  { title: 'Correlation', statistics: ['pearson_r', 'spearman_r', 'kendalls_tau'], signed: true },
  { title: 'Numeric by category (η)', statistics: ['eta'], signed: false },
  { title: 'Category by category (V)', statistics: ['cramers_v_corrected'], signed: false },
  { title: 'Group differences (ω²)', statistics: ['omega_squared'], signed: false },
]

export interface Panel {
  title: string
  signed: boolean
  cells: Cell[]
  columns: string[]
  /** Columns the panel's pairs span before any cap; `columns.length` when none was applied. */
  total: number
}

/** The most columns one heatmap panel draws; wider panels keep the strongest. */
export const PANEL_COLUMN_CAP = 30

/** The panels that have a tested pair, with the columns each one draws in dataset order. */
export function panelsOf(matrix: Matrix): Panel[] {
  const position = new Map(matrix.columns.map((name, index) => [name, index]))
  return PANELS.flatMap((panel) => {
    const cells = matrix.cells.filter((c) => panel.statistics.includes(c.statistic))
    if (!cells.length) return []
    const names = new Set(cells.flatMap((c) => [c.first, c.second]))
    const columns = [...names].sort((a, b) => (position.get(a) ?? 0) - (position.get(b) ?? 0))
    return [{ title: panel.title, signed: panel.signed, cells, columns, total: columns.length }]
  })
}

/**
 * A panel cut to the `limit` columns whose strongest cell is largest in absolute value,
 * kept in dataset order, and the cells between them. A panel within the limit is returned
 * as it is.
 */
export function capPanel(panel: Panel, limit = PANEL_COLUMN_CAP): Panel {
  if (panel.columns.length <= limit) return panel
  const reach = new Map<string, number>()
  for (const c of panel.cells)
    for (const name of [c.first, c.second])
      reach.set(name, Math.max(reach.get(name) ?? 0, Math.abs(c.value)))
  const keep = new Set(
    [...panel.columns].sort((a, b) => (reach.get(b) ?? 0) - (reach.get(a) ?? 0)).slice(0, limit),
  )
  return {
    ...panel,
    columns: panel.columns.filter((name) => keep.has(name)),
    cells: panel.cells.filter((c) => keep.has(c.first) && keep.has(c.second)),
  }
}

/** The height a panel's heatmap needs: a row per column, plus the axis and scale. */
export const panelHeight = (panel: Panel) => Math.max(panel.columns.length * 22 + 140, 260)

/**
 * A heatmap of one panel. A pair is drawn in both halves; a pair the check did not test
 * has no data point, so its square stays empty rather than reading as zero.
 */
export function matrixOption(panel: Panel): EChartsOption {
  const index = new Map(panel.columns.map((name, i) => [name, i]))
  const data = panel.cells.flatMap((c) => {
    const x = index.get(c.first)
    const y = index.get(c.second)
    const point = (a: number, b: number) => ({
      value: [a, b, c.value],
      statistic: STATISTIC_LABEL[c.statistic],
      n: c.n_used,
    })
    return x === undefined || y === undefined
      ? []
      : x === y
        ? [point(x, y)]
        : [point(x, y), point(y, x)]
  })
  return {
    grid: { left: 110, right: 24, top: 12, bottom: 100 },
    tooltip: {
      formatter: (p: unknown) => {
        const { value, statistic, n } = (
          p as { data: { value: number[]; statistic: string; n: number } }
        ).data
        const [x = 0, y = 0, v = 0] = value
        return `${panel.columns[y]} × ${panel.columns[x]}<br/>${statistic} = ${v.toFixed(3)} (n = ${n})`
      },
    },
    xAxis: { type: 'category', data: panel.columns, axisLabel: { rotate: 45 } },
    yAxis: { type: 'category', data: panel.columns, inverse: true },
    visualMap: {
      min: panel.signed ? -1 : 0,
      max: 1,
      calculable: false,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      inRange: { color: panel.signed ? ['#2563eb', '#e5e7eb', '#dc2626'] : ['#e5e7eb', '#7c3aed'] },
    },
    series: [{ type: 'heatmap', data }],
  }
}

/** Cells grouped by statistic in the API's order, then by |value|, strongest first. */
export function sortedCells(matrix: Matrix): Cell[] {
  const rank = new Map(matrix.statistics.map((s, i) => [s, i]))
  return [...matrix.cells].sort(
    (a, b) =>
      (rank.get(a.statistic) ?? 0) - (rank.get(b.statistic) ?? 0) ||
      Math.abs(b.value) - Math.abs(a.value),
  )
}

export type PlotKind = 'scatter' | 'groups' | 'counts'

/** What to draw for two columns, from their semantic types. */
export function plotKind(first: string, second: string): PlotKind {
  const numeric = (type: string) => type === 'numeric'
  if (numeric(first) && numeric(second)) return 'scatter'
  return numeric(first) || numeric(second) ? 'groups' : 'counts'
}

const text = (value: unknown) => String(value)

/** The pair plot's option, drawn from the snapshot's values and nothing recomputed. */
export function pairOption(
  kind: PlotKind,
  points: PairPoints,
  numericFirst: boolean,
): EChartsOption {
  const rows = points.values as unknown as (string | number | null)[][]
  if (kind === 'scatter') {
    return {
      grid: { left: 64, right: 24, top: 16, bottom: 44 },
      tooltip: { trigger: 'item' },
      xAxis: {
        type: 'value',
        scale: true,
        name: points.first,
        nameLocation: 'middle',
        nameGap: 28,
      },
      yAxis: { type: 'value', scale: true, name: points.second },
      series: [{ type: 'scatter', symbolSize: 4, data: rows.map((r) => [r[0], r[1]]) }],
    }
  }
  if (kind === 'groups') {
    const [groupName, valueName] = numericFirst
      ? [points.second, points.first]
      : [points.first, points.second]
    const groups = [...new Set(rows.map((r) => text(r[numericFirst ? 1 : 0])))].sort()
    const slot = new Map(groups.map((g, i) => [g, i]))
    return {
      grid: { left: 64, right: 24, top: 16, bottom: 44 },
      tooltip: { trigger: 'item' },
      xAxis: {
        type: 'category',
        data: groups,
        name: groupName,
        nameLocation: 'middle',
        nameGap: 28,
      },
      yAxis: { type: 'value', scale: true, name: valueName },
      series: [
        {
          type: 'scatter',
          symbolSize: 4,
          data: rows.map((r) => [slot.get(text(r[numericFirst ? 1 : 0])), r[numericFirst ? 0 : 1]]),
        },
      ],
    }
  }
  const xs = [...new Set(rows.map((r) => text(r[0])))].sort()
  const ys = [...new Set(rows.map((r) => text(r[1])))].sort()
  const counts = new Map<string, number>()
  for (const r of rows) {
    const key = `${text(r[0])}\u0000${text(r[1])}`
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  return {
    grid: { left: 110, right: 24, top: 12, bottom: 80 },
    tooltip: { trigger: 'item' },
    xAxis: { type: 'category', data: xs, name: points.first, nameLocation: 'middle', nameGap: 40 },
    yAxis: { type: 'category', data: ys, name: points.second, inverse: true },
    visualMap: {
      min: 0,
      max: Math.max(1, ...counts.values()),
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
    },
    series: [
      {
        type: 'heatmap',
        data: [...counts].map(([key, n]) => {
          const [x = '', y = ''] = key.split('\u0000')
          return [xs.indexOf(x), ys.indexOf(y), n]
        }),
      },
    ],
  }
}

export interface VifRow {
  column: string
  vif: number
  tier: string
  partners: { column: string; r: number }[]
}

export interface LeakageRow {
  feature: string
  metric: string
  value: number
  n_used: number
  flagged: boolean
}

/** Rows of a check's stored output, or none when the output lacks the list. */
export function rowsOf<T>(output: Record<string, unknown>, key: string): T[] {
  const rows = output[key]
  return Array.isArray(rows) ? (rows as T[]) : []
}

/** The stored record of a pair, whichever way round the check named its columns. */
export function pairIn<T extends Record<string, unknown>>(
  rows: T[],
  keys: [string, string],
  a: string,
  b: string,
): T | undefined {
  return rows.find(
    (r) => (r[keys[0]] === a && r[keys[1]] === b) || (r[keys[0]] === b && r[keys[1]] === a),
  )
}

export const STRENGTHS = ['strong', 'moderate', 'weak', 'negligible'] as const
export type Strength = (typeof STRENGTHS)[number]

/** Tested cells per strength band, and those whose check names no band. */
export function strengthCounts(cells: Cell[]): {
  counts: Record<Strength, number>
  unbanded: number
} {
  const counts: Record<Strength, number> = { strong: 0, moderate: 0, weak: 0, negligible: 0 }
  let unbanded = 0
  for (const c of cells) {
    if (c.strength !== null && c.strength in counts) counts[c.strength as Strength] += 1
    else unbanded += 1
  }
  return { counts, unbanded }
}

export interface Association {
  other: string
  cell: Cell
}

/** Every tested pair that includes `column`, strongest first. */
export function associationsOf(matrix: Matrix, column: string): Association[] {
  return matrix.cells
    .filter((c) => c.first === column || c.second === column)
    .map((cell) => ({ other: cell.first === column ? cell.second : cell.first, cell }))
    .sort((a, b) => Math.abs(b.cell.value) - Math.abs(a.cell.value))
}

/** Per column, how many tested pairs are banded strong. */
export function strongCounts(matrix: Matrix): Map<string, number> {
  const counts = new Map<string, number>()
  for (const c of matrix.cells) {
    if (c.strength !== 'strong') continue
    for (const name of new Set([c.first, c.second])) counts.set(name, (counts.get(name) ?? 0) + 1)
  }
  return counts
}

export interface Chip {
  id: 'collinearity' | 'duplicates' | 'leakage'
  label: string
  columns: string[]
}

/**
 * The warning chips, each with the columns it concerns: columns with a VIF of 10 or more,
 * columns equal to another, and features the leakage check flagged. A chip with no column
 * is left out.
 */
export function chipsOf(outputs: {
  collinearity?: Record<string, unknown> | null
  duplicate_columns?: Record<string, unknown> | null
  target_leakage?: Record<string, unknown> | null
}): Chip[] {
  const high = rowsOf<VifRow>(outputs.collinearity ?? {}, 'columns')
    .filter((r) => r.vif >= 10)
    .map((r) => r.column)
  const groups = rowsOf<string[]>(outputs.duplicate_columns ?? {}, 'groups').flat()
  const leaks = [
    ...new Set(
      rowsOf<LeakageRow>(outputs.target_leakage ?? {}, 'features')
        .filter((r) => r.flagged)
        .map((r) => r.feature),
    ),
  ]
  const chips: Chip[] = [
    { id: 'collinearity', label: 'Collinear (VIF ≥ 10)', columns: high },
    { id: 'duplicates', label: 'Duplicate columns', columns: groups },
    { id: 'leakage', label: 'Possible leakage', columns: leaks },
  ]
  return chips.filter((c) => c.columns.length)
}

/** Whether a pair involves a column in `columns`, or any pair when `columns` is null. */
export const touches = (cell: Cell, columns: Set<string> | null) =>
  columns === null || columns.has(cell.first) || columns.has(cell.second)
