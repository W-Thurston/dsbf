import { describe, expect, it } from 'vitest'
import type { components } from '../../generated/api'
import {
  associationsOf,
  capPanel,
  chipsOf,
  matrixOption,
  pairIn,
  pairOption,
  panelHeight,
  panelsOf,
  plotKind,
  sortedCells,
  strengthCounts,
  strongCounts,
  touches,
} from './relationships'

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

describe('the heatmap cap', () => {
  const wide = (n: number): Matrix => {
    const names = Array.from({ length: n }, (_, i) => `c${i}`)
    return {
      columns: names,
      statistics: ['pearson_r'],
      coverage: [],
      cells: names.slice(1).map((name, i) => cell('c0', name, 'pearson_r', i === 4 ? 0.9 : 0.1)),
    }
  }

  it('keeps the strongest columns in dataset order and says how many there were', () => {
    const [panel] = panelsOf(wide(40))
    const capped = capPanel(panel!, 5)
    expect(capped.total).toBe(40)
    expect(capped.columns).toHaveLength(5)
    expect(capped.columns).toContain('c0')
    expect(capped.columns).toContain('c5')
    expect([...capped.columns].sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)))).toEqual(
      capped.columns,
    )
    expect(
      capped.cells.every(
        (c) => capped.columns.includes(c.first) && capped.columns.includes(c.second),
      ),
    ).toBe(true)
  })

  it('leaves a panel within the limit as it is', () => {
    const [panel] = panelsOf(wide(6))
    expect(capPanel(panel!, 30)).toBe(panel)
    expect(panelHeight(panel!)).toBeGreaterThanOrEqual(260)
  })
})

describe('strength and associations', () => {
  const banded = (strength: string | null, a = 'a', b = 'b'): Cell => ({
    ...cell(a, b, 'pearson_r', 0.5),
    strength,
  })

  it('counts the bands and the cells that name none', () => {
    const { counts, unbanded } = strengthCounts([
      banded('strong'),
      banded('strong'),
      banded('weak'),
      banded(null),
    ])
    expect(counts).toEqual({ strong: 2, moderate: 0, weak: 1, negligible: 0 })
    expect(unbanded).toBe(1)
  })

  it('lists a column’s pairs strongest first, whichever side it is on', () => {
    const rows = associationsOf(MATRIX, 'c')
    expect(rows.map((r) => r.other)).toEqual(['a', 'd'])
  })

  it('counts strong pairs per column', () => {
    const counts = strongCounts({
      ...MATRIX,
      cells: [banded('strong', 'a', 'b'), banded('strong', 'a', 'c'), banded('weak', 'b', 'c')],
    })
    expect(counts.get('a')).toBe(2)
    expect(counts.get('c')).toBe(1)
  })

  it('narrows pairs to those touching the chip’s columns', () => {
    const c = cell('a', 'b', 'pearson_r', 1)
    expect(touches(c, null)).toBe(true)
    expect(touches(c, new Set(['b']))).toBe(true)
    expect(touches(c, new Set(['z']))).toBe(false)
  })
})

describe('the warning chips', () => {
  it('names the columns each check concerns and drops a chip with none', () => {
    const chips = chipsOf({
      collinearity: {
        columns: [
          { column: 'a', vif: 25, tier: 'severe', partners: [] },
          { column: 'b', vif: 2, tier: 'low', partners: [] },
        ],
      },
      duplicate_columns: { groups: [['x', 'y']] },
      target_leakage: {
        features: [{ feature: 'f', metric: 'eta', value: 0.2, n_used: 5, flagged: false }],
      },
    })
    expect(chips.map((c) => [c.id, c.columns])).toEqual([
      ['collinearity', ['a']],
      ['duplicates', ['x', 'y']],
    ])
  })
})
