import type { EChartsOption } from 'echarts'
import type { components } from '../../generated/api'

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
export function histogramOption(histogram: Histogram | null): EChartsOption {
  if (!histogram) return {}
  const { edges, counts } = histogram
  const bins = counts.map((count, i) => [edges[i], edges[i + 1], count])
  return {
    grid: { left: 56, right: 24, top: 16, bottom: 36 },
    tooltip: {
      trigger: 'item',
      formatter: (p: unknown) => {
        const [low, high, count] = (p as { value: number[] }).value
        return `${low} to ${high}<br/>${count} values`
      },
    },
    xAxis: { type: 'value', min: edges[0], max: edges[edges.length - 1], scale: true },
    yAxis: { type: 'value', name: 'values' },
    series: [
      {
        type: 'custom',
        data: bins,
        encode: { x: [0, 1], y: 2 },
        renderItem: (_params, api) => {
          const [x0 = 0, y0 = 0] = api.coord([api.value(0), api.value(2)])
          const [x1 = 0, y1 = 0] = api.coord([api.value(1), 0])
          return {
            type: 'rect',
            shape: { x: x0, y: y0, width: Math.max(x1 - x0 - 1, 1), height: y1 - y0 },
            style: api.style(),
          }
        },
      },
    ],
  }
}

/** The most common values as horizontal bars, largest first. */
export function topValuesOption(values: ValueCount[]): EChartsOption {
  return {
    grid: { left: 120, right: 24, top: 8, bottom: 24 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', inverse: true, data: values.map((v) => String(v.value)) },
    series: [{ type: 'bar', data: values.map((v) => v.count) }],
  }
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
