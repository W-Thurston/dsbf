import { describe, expect, it } from 'vitest'
import type { components } from '../../generated/api'
import { matrixOption, pairIn, pairOption, panelsOf, plotKind, sortedCells } from './relationships'

type Matrix = components['schemas']['RelationshipMatrix']
type Cell = components['schemas']['RelationshipCell']

const cell = (
  first: string,
  second: string,
  statistic: Cell['statistic'],
  value: number,
): Cell => ({
  first,
  second,
  check_id: 'associations',
  statistic,
  value,
  strength: null,
  n_used: 10,
  interval: null,
  flagged: null,
})

const MATRIX: Matrix = {
  columns: ['a', 'b', 'c', 'd'],
  statistics: ['pearson_r', 'eta', 'cramers_v_corrected'],
  cells: [
    cell('a', 'b', 'pearson_r', 0.2),
    cell('a', 'c', 'pearson_r', -0.9),
    cell('a', 'd', 'eta', 0.95),
    cell('c', 'd', 'cramers_v_corrected', 0.3),
  ],
  coverage: [],
}

describe('the relationship matrix', () => {
  it('gives each statistic its own panel, with only the columns it tested', () => {
    const panels = panelsOf(MATRIX)
    expect(panels.map((p) => p.title)).toEqual([
      'Correlation',
      'Numeric by category (η)',
      'Category by category (V)',
    ])
    expect(panels[0]!.columns).toEqual(['a', 'b', 'c'])
    expect(panels[0]!.signed).toBe(true)
  })

  it('draws a tested pair in both halves and leaves an untested one without data', () => {
    const [correlation] = panelsOf(MATRIX)
    const option = matrixOption(correlation!)
    const [series] = option.series as { data: { value: number[] }[] }[]
    const squares = series!.data.map((d) => d.value.join(','))
    expect(squares).toContain('0,1,0.2')
    expect(squares).toContain('1,0,0.2')
    // b and c were not tested together: neither square exists, so none reads as zero
    expect(squares.some((s) => s.startsWith('1,2,') || s.startsWith('2,1,'))).toBe(false)
    expect(series!.data).toHaveLength(4)
  })

  it('sorts by statistic group before value', () => {
    const order = sortedCells(MATRIX).map((c) => `${c.statistic}:${c.value}`)
    expect(order).toEqual([
      'pearson_r:-0.9',
      'pearson_r:0.2',
      'eta:0.95',
      'cramers_v_corrected:0.3',
    ])
  })
})

describe('the pair plot', () => {
  const points = {
    first: 'x',
    second: 'g',
    values: [
      [1, 'u'],
      [2, 'v'],
      [3, 'u'],
    ],
    rows_with_both: 3,
    snapshot_rows: 3,
    sampling: null,
  } as unknown as components['schemas']['PairPoints']

  it('picks a plot from the two types', () => {
    expect(plotKind('numeric', 'numeric')).toBe('scatter')
    expect(plotKind('categorical', 'numeric')).toBe('groups')
    expect(plotKind('categorical', 'text')).toBe('counts')
  })

  it('places a numeric value at its group on a category axis', () => {
    const option = pairOption('groups', points, true)
    const [series] = option.series as { data: unknown[][] }[]
    expect(series!.data).toEqual([
      [0, 1],
      [1, 2],
      [0, 3],
    ])
  })

  it('counts a categorical pair', () => {
    const [series] = pairOption('counts', points, false).series as { data: number[][] }[]
    expect(series!.data.reduce((sum, d) => sum + d[2]!, 0)).toBe(3)
  })

  it('finds a stored pair whichever way round it was named', () => {
    const rows = [{ first: 'a', second: 'b', v: 1 }]
    expect(pairIn(rows, ['first', 'second'], 'b', 'a')?.v).toBe(1)
    expect(pairIn(rows, ['first', 'second'], 'a', 'c')).toBeUndefined()
  })
})
