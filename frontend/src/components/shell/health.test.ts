import { describe, expect, it } from 'vitest'
import { healthBar, TABS } from './health'
import type { QualityDimension, QualityOutput } from '../quality/quality'

const dimension = (over: Partial<QualityDimension>): QualityDimension => ({
  assessed: true,
  level: 'green',
  coverage: 1,
  inputs: [],
  affected_columns: [],
  affected_count: 0,
  pct_affected: 0,
  findings: [],
  ...over,
})

const output: QualityOutput = {
  total_columns: 10,
  clean_columns: [],
  dimensions: {
    completeness: dimension({
      level: 'red',
      affected_count: 4,
      pct_affected: 0.4,
      affected_columns: ['a', 'b', 'c', 'd'],
    }),
    validity: dimension({ level: 'amber', affected_count: 1, pct_affected: 0.1 }),
    usability: dimension({}),
    redundancy: dimension({ level: 'blue', affected_count: 0 }),
    leakage: dimension({ assessed: false, level: null, coverage: 0 }),
  },
}

describe('the health bar', () => {
  const bar = healthBar(output, 'overview')
  const by = (name: string) => bar.find((d) => d.name === name)!

  it('lists the five dimensions in the old order', () => {
    expect(bar.map((d) => d.name)).toEqual([
      'completeness',
      'validity',
      'usability',
      'redundancy',
      'leakage',
    ])
  })

  it('says how many columns a flagged dimension affects, and "All clear" for green', () => {
    expect(by('completeness').summary).toBe('4 cols')
    expect(by('validity').summary).toBe('1 col')
    expect(by('usability').summary).toBe('All clear')
    expect(by('redundancy').summary).toBe('Minor') // flagged, but no column named
    expect(by('completeness').detail).toContain('4 of 10 columns (40%)')
  })

  it('draws a dimension nobody assessed grey, never green', () => {
    expect(by('leakage').mark).toBe('none')
    expect(by('leakage').summary).toBe('Not assessed')
  })

  it('dims the dimensions a tab does not speak to', () => {
    const relationships = healthBar(output, 'relationships')
    expect(relationships.filter((d) => d.relevant).map((d) => d.name)).toEqual([
      'redundancy',
      'leakage',
    ])
  })

  it('draws a dimension the run did not report as unassessed', () => {
    const missing = healthBar({ ...output, dimensions: {} }, 'overview')
    expect(missing.every((d) => d.mark === 'none')).toBe(true)
  })
})

describe('the tabs', () => {
  it('are the six of the old dashboard', () => {
    expect(TABS.map((t) => t.label)).toEqual([
      'Overview',
      'Quality',
      'Distributions',
      'Relationships',
      'ML Readiness',
      'Time Series',
    ])
  })
})
