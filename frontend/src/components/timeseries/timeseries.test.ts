import { describe, expect, it } from 'vitest'
import {
  acfOption,
  assessmentClass,
  assessmentWords,
  componentsOf,
  pacfOption,
  pValueWords,
  rowRange,
  stlOption,
  strengthWords,
  type ColumnAutocorrelation,
  type UnitRootTest,
} from './timeseries'

const test = (p_value: number, p_value_limit: UnitRootTest['p_value_limit'] = null) =>
  ({ test: 'kpss', statistic: 0, p_value, p_value_limit, lags: 0, rejects: false }) as UnitRootTest

const COLUMN = {
  acf: [1, 0.5, -0.1, 0.3],
  acf_band: [0, 0.2, 0.2, 0.2],
  pacf: [1, 0.5, -0.05, 0.1],
  pacf_band: 0.2,
  flagged_acf: [1, 3],
  flagged_pacf: [1],
} as ColumnAutocorrelation

describe('the stationarity words', () => {
  it('is green only for a series that is stationary', () => {
    expect(assessmentClass('consistent_with_stationary')).toBe('badge-ok')
    expect(assessmentClass('consistent_with_non_stationary')).toBe('badge-warn')
    expect(assessmentClass('inconclusive')).not.toBe('badge-ok')
    expect(assessmentClass('indeterminate')).not.toBe('badge-ok')
  })

  it('names an assessment and falls back to its spaced code', () => {
    expect(assessmentWords('consistent_with_trend_stationary')).toBe('Stationary around a trend')
    expect(assessmentWords('something_new')).toBe('something new')
  })

  it('shows the bound KPSS gives instead of an estimate', () => {
    expect(pValueWords(test(0.1, 'at_least'))).toBe('≥ 0.100')
    expect(pValueWords(test(0.01, 'at_most'))).toBe('≤ 0.010')
    expect(pValueWords(test(0.0004))).toBe('< 0.001')
    expect(pValueWords(null)).toBe('not run')
  })

  it('calls a strength within noise when it is under the floor', () => {
    expect(strengthWords(0.3, 0.32)).toBe('within noise')
    expect(strengthWords(0.5, 0.3)).toBe('moderate')
    expect(strengthWords(0.8, 0.3)).toBe('strong')
    expect(strengthWords(0.35, 0.3)).toBe('weak')
  })
})

describe('the decomposition rows', () => {
  const columns = { a: { periods: 3 }, b: { periods: 2 }, c: { periods: 4 } }

  it('finds a column after those before it, in the output order', () => {
    expect(rowRange(columns, 'a')).toEqual({ offset: 0, count: 3 })
    expect(rowRange(columns, 'c')).toEqual({ offset: 5, count: 4 })
    expect(rowRange(columns, 'missing')).toBeNull()
  })

  const names = ['column', 'period', 'observed', 'trend', 'seasonal', 'remainder']

  it('reads four series from the rows by column name', () => {
    const rows = [
      ['b', 0, 1, 2, 3, 4],
      ['b', 1, 5, 6, 7, 8],
    ]
    expect(componentsOf(names, rows, 'b')).toEqual({
      observed: [1, 5],
      trend: [2, 6],
      seasonal: [3, 7],
      remainder: [4, 8],
    })
  })

  it('refuses rows that belong to another column', () => {
    expect(() => componentsOf(names, [['a', 0, 1, 2, 3, 4]], 'b')).toThrow(/not those of b/)
  })

  it('says when the table lacks a series', () => {
    expect(() => componentsOf(['column'], [], 'a')).toThrow(/no observed column/)
  })
})

describe('the charts', () => {
  it('draws the ACF from lag 1 with the band either side of zero', () => {
    const option = acfOption(COLUMN) as { series: { data: unknown[] }[]; xAxis: { data: number[] } }
    expect(option.xAxis.data).toEqual([1, 2, 3])
    expect(option.series[1]?.data).toEqual([0.2, 0.2, 0.2])
    expect(option.series[2]?.data).toEqual([-0.2, -0.2, -0.2])
  })

  it('fades the lags that are not flagged', () => {
    const bars = (acfOption(COLUMN) as { series: { data: { itemStyle: { opacity: number } }[] }[] })
      .series[0]!.data
    expect(bars.map((b) => b.itemStyle.opacity)).toEqual([1, 0.45, 1])
  })

  it('gives the PACF its constant band', () => {
    const option = pacfOption(COLUMN) as { series: { data: number[] }[] }
    expect(option.series[1]?.data).toEqual([0.2, 0.2, 0.2])
  })

  it('stacks four panels on a shared axis', () => {
    const option = stlOption({
      observed: [1, 2],
      trend: [1, 1],
      seasonal: [0, 1],
      remainder: [0, 0],
    }) as { series: { name: string }[]; grid: unknown[] }
    expect(option.series.map((s) => s.name)).toEqual(['Observed', 'Trend', 'Seasonal', 'Remainder'])
    expect(option.grid).toHaveLength(4)
  })
})
