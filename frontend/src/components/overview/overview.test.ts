import { describe, expect, it } from 'vitest'
import {
  formatBytes,
  missingTone,
  modeTone,
  skewTone,
  sketchBars,
  sortColumns,
  typeCounts,
} from './overview'

function column(name: string, position: number, over: Record<string, unknown> = {}) {
  return {
    name,
    position,
    semantic_type: 'numeric',
    physical_dtype: 'Float64',
    counts: { distinct: 3, distinct_exact: true, nulls: 0, rows: 10 },
    sketch: {
      mode: null,
      mode_share: null,
      skewness: null,
      min: 0,
      max: 1,
      bins: [],
      top_values: [],
    },
    ...over,
  } as never
}

describe('tones', () => {
  it('marks missing share from 5% and 20%', () => {
    expect([0.04, 0.05, 0.19, 0.2].map(missingTone)).toEqual(['none', 'amber', 'amber', 'red'])
  })

  it('marks a dominant value from 70% and 90%, and skew beyond 1 and 2', () => {
    expect([null, 0.69, 0.7, 0.9].map(modeTone)).toEqual(['none', 'none', 'amber', 'red'])
    expect([null, 1, -1.5, 2.5].map(skewTone)).toEqual(['none', 'none', 'amber', 'red'])
  })
})

describe('sketchBars', () => {
  it('scales the bins to the tallest', () => {
    const c = column('a', 0, { sketch: { bins: [1, 4, 2], top_values: [] } })
    expect(sketchBars(c).map((b) => b.height)).toEqual([0.25, 1, 0.5])
  })

  it('uses the top values of a categorical column and is empty without either', () => {
    const c = column('a', 0, {
      sketch: {
        bins: [],
        top_values: [
          { value: 'x', count: 6 },
          { value: 'y', count: 3 },
        ],
      },
    })
    expect(sketchBars(c).map((b) => b.height)).toEqual([1, 0.5])
    expect(sketchBars(column('b', 1))).toEqual([])
  })
})

describe('sortColumns', () => {
  it('sorts by missing share and puts an absent figure last either way', () => {
    const a = column('a', 0, {
      counts: { distinct: 1, nulls: 5, rows: 10 },
      sketch: { skewness: -2, bins: [], top_values: [] },
    })
    const b = column('b', 1, { counts: { distinct: 1, nulls: 1, rows: 10 } })
    const c = column('c', 2, { sketch: { skewness: null, bins: [], top_values: [] } })
    expect(sortColumns([b, a], 'missing', true).map((x) => x.name)).toEqual(['a', 'b'])
    expect(sortColumns([c, a], 'skew', false).map((x) => x.name)).toEqual(['a', 'c'])
    expect(sortColumns([c, a], 'skew', true).map((x) => x.name)).toEqual(['a', 'c'])
  })
})

describe('summaries', () => {
  it('counts columns per type, most common first', () => {
    const cols = [column('a', 0), column('b', 1), column('c', 2, { semantic_type: 'categorical' })]
    expect(typeCounts(cols)).toEqual([
      { label: 'numeric', count: 2 },
      { label: 'categorical', count: 1 },
    ])
  })

  it('formats bytes', () => {
    expect([512, 2048, 94757, 5 * 1024 ** 2].map(formatBytes)).toEqual([
      '512 B',
      '2.0 KB',
      '93 KB',
      '5.0 MB',
    ])
  })
})
