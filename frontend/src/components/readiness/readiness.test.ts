import { describe, expect, it } from 'vitest'
import { actionNote } from './actionMeta'
import { skewBars, type ColumnTransforms } from './readiness'

describe('actionNote', () => {
  it('finds a method under its action, then on its own', () => {
    expect(actionNote('impute', 'median')?.what).toContain('median')
    expect(actionNote('transform', 'log1p')?.what).toContain('log')
    expect(actionNote('encode', 'frequency')).toEqual(actionNote('encode', 'frequency_encoding'))
  })

  it('normalises how a method is spelled', () => {
    expect(actionNote('transform', 'Yeo-Johnson')).toEqual(actionNote('transform', 'yeo_johnson'))
  })

  it('reads an action with no method by its action', () => {
    expect(actionNote('drop')?.what).toContain('Removes the column')
  })

  it('never reads an action with a method as the bare action', () => {
    expect(actionNote('stratified_split', 'time_ordered')).toBeNull()
    expect(actionNote('summarise', 'median')).toBeNull()
  })

  it('has nothing for guidance it does not know', () => {
    expect(actionNote('transform', 'pca')).toBeNull()
  })
})

describe('skewBars', () => {
  const column: ColumnTransforms = {
    skewness_before: 2.4,
    recommended: 'log1p',
    candidates: {
      log1p: { skewness_after: -0.1, lam: null, skipped: null },
      box_cox: { skewness_after: null, lam: null, skipped: 'requires all values > 0' },
    },
  }

  it('draws the skew before and after each applied candidate, starring the recommended one', () => {
    const { bars } = skewBars(column)
    expect(bars.map((b) => b.label)).toEqual(['before', 'log1p ★'])
    expect(bars.map((b) => b.text)).toEqual(['+2.40', '−0.10'])
    expect(bars.map((b) => b.count)).toEqual([2.4, 0.1])
  })

  it('lists a skipped candidate with its reason instead of a bar', () => {
    expect(skewBars(column).skipped).toEqual(['box_cox: requires all values > 0'])
  })
})
