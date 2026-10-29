import type { EChartsOption } from 'echarts'

// What the three time series checks say, as the words and chart options the tab shows.

export const TIME_SERIES_CHECKS = ['stationarity', 'autocorrelation', 'seasonal_decomposition']

/** Rows the API returns per request of an artifact. */
export const ARTIFACT_PAGE = 500

export interface SeriesInfo {
  periods: number
  periods_filled: number
  rows_used: number
  series_start: string
}

export interface SeriesOutput {
  index: string
  frequency: string
  frequency_source: 'configured' | 'profile'
  aggregate: string
  max_periods: number
  periods_total: number
  truncated: boolean
  timezone: string | null
  skipped: Record<string, string>
}

export interface UnitRootTest {
  test: 'adf' | 'kpss'
  statistic: number
  p_value: number
  p_value_limit: 'at_most' | 'at_least' | null
  lags: number
  rejects: boolean
}

export interface ColumnStationarity extends SeriesInfo {
  adf: UnitRootTest | null
  kpss: UnitRootTest | null
  adf_trend: UnitRootTest | null
  kpss_trend: UnitRootTest | null
  errors: Record<string, string>
  assessment: string
  confidence: 'moderate' | 'low'
}

export interface StationarityOutput extends SeriesOutput {
  alpha: number
  columns: Record<string, ColumnStationarity>
}

export interface ColumnAutocorrelation extends SeriesInfo {
  max_lags: number
  acf: number[]
  acf_band: number[]
  pacf: number[]
  pacf_band: number
  ljung_box_p: number
  significant_acf: number[]
  significant_pacf: number[]
  flagged_acf: number[]
  flagged_pacf: number[]
  strongest_lag: number | null
}

export interface AutocorrelationOutput extends SeriesOutput {
  alpha: number
  min_abs: number
  columns: Record<string, ColumnAutocorrelation>
}

export interface ColumnDecomposition extends SeriesInfo {
  period: number
  robust: boolean
  trend_strength: number
  seasonal_strength: number
  remainder_share: number
  seasonal_floor: number
  trend_floor: number
}

export interface DecompositionOutput extends SeriesOutput {
  period: number | null
  period_source: string
  min_strength: number
  columns: Record<string, ColumnDecomposition>
}

/** The four series of a decomposition, one value per period. */
export interface Components {
  observed: number[]
  trend: number[]
  seasonal: number[]
  remainder: number[]
}

const ASSESSMENT: Record<string, string> = {
  consistent_with_stationary: 'Stationary',
  consistent_with_trend_stationary: 'Stationary around a trend',
  consistent_with_non_stationary: 'Not stationary',
  ambiguous_trend_stationary_or_structural_break: 'Trend or structural break',
  inconclusive: 'Inconclusive',
  indeterminate: 'Could not be tested',
}

export function assessmentWords(assessment: string): string {
  return ASSESSMENT[assessment] ?? assessment.replaceAll('_', ' ')
}

/** The badge class of an assessment: only a clean pass is green; a drifting level is a warning. */
export function assessmentClass(assessment: string): string {
  switch (assessment) {
    case 'consistent_with_stationary':
      return 'badge-ok'
    case 'consistent_with_trend_stationary':
      return 'badge-info'
    case 'consistent_with_non_stationary':
    case 'ambiguous_trend_stationary_or_structural_break':
      return 'badge-warn'
    default:
      return 'badge-notice'
  }
}

/** A test's p-value, with the bound KPSS reports when the table runs out. */
export function pValueWords(test: UnitRootTest | null): string {
  if (!test) return 'not run'
  const bound =
    test.p_value_limit === 'at_most' ? '≤ ' : test.p_value_limit === 'at_least' ? '≥ ' : ''
  return `${bound}${test.p_value < 0.001 ? '< 0.001' : test.p_value.toFixed(3)}`
}

/** Strength of a trend or season as the check labels it, against the noise floor. */
export function strengthWords(strength: number, floor: number): string {
  if (strength < floor) return 'within noise'
  if (strength >= 0.7) return 'strong'
  if (strength >= 0.4) return 'moderate'
  return 'weak'
}

/**
 * Where a column's rows start in the decomposition artifact: it is the columns' series
 * one after the other in the output's column order, each as long as its periods.
 */
export function rowRange(
  columns: Record<string, { periods: number }>,
  column: string,
): { offset: number; count: number } | null {
  let offset = 0
  for (const [name, info] of Object.entries(columns)) {
    if (name === column) return { offset, count: info.periods }
    offset += info.periods
  }
  return null
}

/** The rows of the decomposition artifact as four series, or why they cannot be drawn. */
export function componentsOf(
  names: readonly string[],
  rows: readonly (readonly unknown[])[],
  column: string,
): Components {
  const at = (name: string) => {
    const index = names.indexOf(name)
    if (index < 0) throw new Error(`the decomposition has no ${name} column`)
    return index
  }
  const [c, o, t, s, r] = ['column', 'observed', 'trend', 'seasonal', 'remainder'].map(at) as [
    number,
    number,
    number,
    number,
    number,
  ]
  const result: Components = { observed: [], trend: [], seasonal: [], remainder: [] }
  for (const row of rows) {
    if (row[c] !== column) throw new Error(`the rows read are not those of ${column}`)
    result.observed.push(row[o] as number)
    result.trend.push(row[t] as number)
    result.seasonal.push(row[s] as number)
    result.remainder.push(row[r] as number)
  }
  return result
}

// Both edges of a band share one muted colour: they are a reference, not two series.
const BAND_STYLE = { type: 'dashed' as const, color: '#94a3b8' }

const PANELS: { key: keyof Components; label: string }[] = [
  { key: 'observed', label: 'Observed' },
  { key: 'trend', label: 'Trend' },
  { key: 'seasonal', label: 'Seasonal' },
  { key: 'remainder', label: 'Remainder' },
]

/** Four panels stacked on one period axis, zoomed together. */
export function stlOption(components: Components): EChartsOption {
  const height = 17
  const top = 8
  const gap = 5
  return {
    axisPointer: { link: [{ xAxisIndex: 'all' }] },
    tooltip: { trigger: 'axis' },
    dataZoom: [{ type: 'inside', xAxisIndex: [0, 1, 2, 3] }],
    grid: PANELS.map((_, i) => ({
      left: 72,
      right: 24,
      top: `${top + i * (height + gap)}%`,
      height: `${height}%`,
    })),
    xAxis: PANELS.map((_, i) => ({
      type: 'category' as const,
      gridIndex: i,
      data: components.observed.map((_v, p) => p),
      axisLabel: { show: i === PANELS.length - 1 },
      name: i === PANELS.length - 1 ? 'period' : '',
      nameLocation: 'middle' as const,
      nameGap: 26,
    })),
    yAxis: PANELS.map((panel, i) => ({
      type: 'value' as const,
      gridIndex: i,
      name: panel.label,
      nameGap: 44,
      nameLocation: 'middle' as const,
      scale: true,
      splitNumber: 2,
    })),
    series: PANELS.map((panel, i) => ({
      type: 'line' as const,
      name: panel.label,
      xAxisIndex: i,
      yAxisIndex: i,
      showSymbol: false,
      data: components[panel.key],
    })),
  }
}

/** Autocorrelation at lags 1 up: stems, with the confidence band drawn as lines. */
export function acfOption(column: ColumnAutocorrelation): EChartsOption {
  const lags = column.acf.map((_v, k) => k).slice(1)
  return correlogram(
    lags,
    column.acf.slice(1),
    column.acf_band.slice(1),
    new Set(column.flagged_acf),
    'ACF',
  )
}

/** Partial autocorrelation at lags 1 up, with its constant band. */
export function pacfOption(column: ColumnAutocorrelation): EChartsOption {
  const lags = column.pacf.map((_v, k) => k).slice(1)
  const band = lags.map(() => column.pacf_band)
  return correlogram(lags, column.pacf.slice(1), band, new Set(column.flagged_pacf), 'PACF')
}

function correlogram(
  lags: number[],
  values: number[],
  band: number[],
  flagged: Set<number>,
  name: string,
): EChartsOption {
  return {
    grid: { left: 64, right: 24, top: 24, bottom: 40 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: lags, name: 'lag', nameLocation: 'middle', nameGap: 26 },
    yAxis: { type: 'value', min: -1, max: 1, name, nameGap: 36, nameLocation: 'middle' },
    series: [
      {
        type: 'bar',
        name,
        barWidth: 3,
        data: values.map((value, i) => ({
          value,
          itemStyle: flagged.has(lags[i] ?? -1) ? { opacity: 1 } : { opacity: 0.45 },
        })),
      },
      {
        type: 'line',
        name: 'upper band',
        showSymbol: false,
        itemStyle: { color: '#94a3b8' },
        lineStyle: BAND_STYLE,
        data: band,
      },
      {
        type: 'line',
        name: 'lower band',
        showSymbol: false,
        itemStyle: { color: '#94a3b8' },
        lineStyle: BAND_STYLE,
        data: band.map((b) => -b),
      },
    ],
  }
}
