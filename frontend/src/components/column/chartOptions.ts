import type { EChartsOption } from 'echarts'
import type { components } from '../../generated/api'
import { histogramBars, topValues } from '../charts/options'

type Theme = 'dark' | 'light'

type Histogram = components['schemas']['Histogram']
type ValueCount = components['schemas']['ValueCount']

/** The stored Q-Q data of one column, as the `normality` check keeps it. */
export interface QQData {
  theoretical: number[]
  empirical: number[]
  reference_x: [number, number]
  reference_y: [number, number]
}

/**
 * The Q-Q data of `column` in a `normality` result's output, or why there is none: the
 * check keeps quantiles only for the columns it flags, and says why it skipped the rest.
 */
export function qqOf(
  output: Record<string, unknown> | null,
  column: string,
): { qq: QQData } | { reason: string } {
  const columns = (output?.columns ?? {}) as Record<string, { qq?: QQData | null }>
  const skipped = (output?.skipped ?? {}) as Record<string, string>
  const tested = columns[column]
  if (tested?.qq) return { qq: tested.qq }
  if (tested) return { reason: 'The column passed the normality tests, so no Q-Q plot was kept.' }
  if (column in skipped)
    return { reason: `The normality check skipped this column: ${skipped[column]}` }
  return { reason: 'The normality check did not test this column.' }
}

/** Bars on a value axis: each spans its stored edges, so the axis reads in the column's units. */
export function histogramOption(histogram: Histogram | null, theme: Theme = 'dark'): EChartsOption {
  return histogram ? histogramBars(histogram.edges, histogram.counts, theme) : {}
}

/** The most common values as horizontal bars, largest first. */
export function topValuesOption(values: ValueCount[], theme: Theme = 'dark'): EChartsOption {
  return topValues(
    values.map((v) => ({ label: String(v.value), count: v.count })),
    theme,
  )
}

/** The column's quantiles against a fitted normal's, with the line through Q1 and Q3. */
export function qqOption(qq: QQData): EChartsOption {
  return {
    grid: { left: 56, right: 24, top: 16, bottom: 40 },
    tooltip: { trigger: 'item' },
    xAxis: {
      type: 'value',
      scale: true,
      name: 'normal quantile',
      nameLocation: 'middle',
      nameGap: 26,
    },
    yAxis: { type: 'value', scale: true, name: 'column quantile' },
    series: [
      { type: 'scatter', symbolSize: 5, data: qq.theoretical.map((x, i) => [x, qq.empirical[i]]) },
      {
        type: 'line',
        symbol: 'none',
        silent: true,
        data: [
          [qq.reference_x[0], qq.reference_y[0]],
          [qq.reference_x[1], qq.reference_y[1]],
        ],
      },
    ],
  }
}
