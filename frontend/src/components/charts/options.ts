// ECharts options for the dashboard's shared charts. Each builder takes plain arrays and
// the theme, so a test can read the option without a canvas.
import type { EChartsOption } from 'echarts'
import { MARKS } from '../../chartTheme'

type Theme = 'dark' | 'light'

export interface ReferenceLine {
  value: number
  label: string
}

/** Index of the bin that holds `value`; the first or last bin when it lies outside. */
export function binIndex(edges: number[], value: number): number {
  const last = edges.length - 2
  for (let i = 0; i <= last; i++) if (value < (edges[i + 1] as number)) return i
  return last
}

/**
 * Bars on a value axis: each spans its stored edges, so the axis reads in the column's
 * units. A reference line (mean, median, a threshold) is drawn at its value.
 */
export function histogramBars(
  edges: number[],
  counts: number[],
  theme: Theme,
  references: ReferenceLine[] = [],
): EChartsOption {
  const marks = MARKS[theme]
  const bins = counts.map((count, i) => [edges[i], edges[i + 1], count])
  return {
    grid: { left: 56, right: 24, top: 24, bottom: 36 },
    tooltip: {
      trigger: 'item',
      formatter: (p: unknown) => {
        const [low, high, count] = (p as { value: number[] }).value
        return `${low} to ${high}<br/>${count} values`
      },
    },
    xAxis: {
      type: 'value',
      min: edges[0],
      max: edges[edges.length - 1],
      scale: true,
      axisLabel: { formatter: (v: number) => String(Number(v.toPrecision(3))) },
    },
    yAxis: { type: 'value', name: 'values' },
    series: [
      {
        type: 'custom',
        data: bins,
        encode: { x: [0, 1], y: 2 },
        itemStyle: { color: marks.primary },
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
      {
        // The custom series cannot carry reference lines, so a hidden one does.
        type: 'line',
        data: [],
        silent: true,
        markLine: {
          symbol: 'none',
          animation: false,
          lineStyle: { color: marks.reference, type: 'dashed', width: 1.5 },
          label: { formatter: '{b}', color: marks.reference },
          // Neighbouring lines (a mean beside a median) put their labels on opposite sides.
          data: references.map((r, i) => ({
            xAxis: r.value,
            name: r.label,
            label: { position: i % 2 ? 'insideEndBottom' : 'insideEndTop' },
          })),
        },
      },
    ],
  }
}

export interface BoxRow {
  label: string
  min: number
  q1: number
  median: number
  q3: number
  max: number
}

/** One horizontal box per row, on a shared axis so spreads compare. */
export function boxPlot(rows: BoxRow[], theme: Theme): EChartsOption {
  const marks = MARKS[theme]
  return {
    grid: { left: 120, right: 24, top: 8, bottom: 24 },
    tooltip: { trigger: 'item' },
    xAxis: { type: 'value', scale: true },
    yAxis: { type: 'category', inverse: true, data: rows.map((r) => r.label) },
    series: [
      {
        type: 'boxplot',
        data: rows.map((r) => [r.min, r.q1, r.median, r.q3, r.max]),
        layout: 'horizontal',
        itemStyle: { color: 'transparent', borderColor: marks.primary, borderWidth: 1.5 },
      },
    ],
  }
}

export interface CountItem {
  label: string
  count: number
}

/** The most common values as horizontal bars, in the order given. */
export function topValues(items: CountItem[], theme: Theme): EChartsOption {
  return {
    grid: { left: 120, right: 24, top: 8, bottom: 24 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', inverse: true, data: items.map((v) => v.label) },
    series: [
      { type: 'bar', data: items.map((v) => v.count), itemStyle: { color: MARKS[theme].primary } },
    ],
  }
}

export interface HeatCell {
  x: number
  y: number
  value: number
}

/**
 * A matrix of values. `signed` centres the scale on zero (a correlation); otherwise it
 * runs from `min` to `max`.
 */
export function heatmap(
  xLabels: string[],
  yLabels: string[],
  cells: HeatCell[],
  theme: Theme,
  scale: { min: number; max: number; signed?: boolean },
): EChartsOption {
  const marks = MARKS[theme]
  const colors = scale.signed
    ? [marks.secondary, 'transparent', marks.primary]
    : ['transparent', marks.primary]
  return {
    grid: { left: 110, right: 24, top: 12, bottom: 90 },
    tooltip: {
      formatter: (p: unknown) => {
        const [x = 0, y = 0, v = 0] = (p as { value: number[] }).value
        return `${yLabels[y]} × ${xLabels[x]} = ${v.toFixed(3)}`
      },
    },
    xAxis: { type: 'category', data: xLabels, axisLabel: { rotate: 45 } },
    yAxis: { type: 'category', data: yLabels, inverse: true },
    visualMap: {
      min: scale.min,
      max: scale.max,
      calculable: false,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      inRange: { color: colors },
    },
    series: [{ type: 'heatmap', data: cells.map((c) => [c.x, c.y, c.value]) }],
  }
}
