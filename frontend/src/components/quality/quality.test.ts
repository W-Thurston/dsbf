import { describe, expect, it } from 'vitest'
import {
  actionWords,
  dimensionSummary,
  effectBar,
  issueChips,
  sortFindings,
  trustOf,
  type FindingRef,
  type QualityOutput,
  gateCaveat,
  gateClass,
  gateWords,
  inputWords,
  levelClass,
  levelWords,
  percent,
  unassessedReason,
  type ReadinessOutput,
} from './quality'

const ready = (over: Partial<ReadinessOutput>): ReadinessOutput => ({
  gate: 'ready',
  complete: true,
  unassessed: [],
  total_columns: 3,
  dimensions: {},
  action_columns: [],
  advisory_columns: [],
  clean_columns: [],
  ...over,
})

describe('levels', () => {
  it('draws a dimension with no level as neutral, never as green', () => {
    expect(levelClass(null)).toBe('badge-info')
    expect(levelWords(null)).toBe('Not assessed')
    expect(levelClass(null)).not.toBe(levelClass('green'))
  })

  it('maps each level to its own badge', () => {
    expect([
      levelClass('green'),
      levelClass('blue'),
      levelClass('amber'),
      levelClass('red'),
    ]).toEqual(['badge-ok', 'badge-notice', 'badge-warn', 'badge-error'])
  })
})

describe('the readiness gate', () => {
  it('words each gate and colours not_assessed neutrally', () => {
    expect(gateWords('not_ready')).toBe('Not ready for modelling')
    expect(gateClass('needs_work')).toBe('badge-warn')
    expect(gateClass('not_assessed')).toBe('badge-info')
  })

  it('says a ready gate holds only for what was assessed', () => {
    const caveat = gateCaveat(ready({ complete: false, unassessed: ['leakage'] }), {
      leakage: 'Leakage risk',
    })
    expect(caveat).toContain('Leakage risk')
    expect(gateCaveat(ready({}), {})).toBe('')
  })
})

describe('inputs', () => {
  it('words every input state', () => {
    expect(inputWords('needs_setting')).toBe('Needs a setting')
    expect(inputWords('failed')).toBe('Did not complete')
  })

  it("uses the check's own reason, or a general one", () => {
    const base = { check_id: 'target_leakage', findings: 0 }
    expect(
      unassessedReason([{ ...base, state: 'needs_setting', reason: 'No target was given.' }]),
    ).toBe('No target was given.')
    expect(unassessedReason([{ ...base, state: 'not_run', reason: null }])).toContain('None of')
  })
})

describe('formatting', () => {
  it('rounds a share to a whole percent', () => {
    expect(percent(0.6666)).toBe('67%')
  })

  it('words an action with its method and columns', () => {
    expect(actionWords({ action: 'transform', method: 'log1p', columns: ['fare'] })).toBe(
      'transform (log1p) fare',
    )
    expect(actionWords({ action: 'review' })).toBe('review')
  })
})

const dim = (level: string | null, over: Record<string, unknown> = {}) =>
  ({
    assessed: level !== null,
    level,
    coverage: 1,
    inputs: [],
    affected_columns: [],
    affected_count: 0,
    pct_affected: 0,
    findings: [],
    ...over,
  }) as never

const ref = (id: string, severity: string, column: string, title: string): FindingRef => ({
  id,
  check_id: 'c',
  code: 'k',
  severity,
  columns: column ? [column] : [],
  title,
})

describe('trustOf', () => {
  const output = (dimensions: Record<string, unknown>, clean = 2): QualityOutput =>
    ({ total_columns: 5, clean_columns: ['a', 'b'].slice(0, clean), dimensions }) as never

  it('takes the worst assessed level and counts the flagged dimensions of those assessed', () => {
    const t = trustOf(output({ a: dim('green'), b: dim('amber'), c: dim('blue'), d: dim(null) }))
    expect(t.level).toBe('amber')
    expect(t.headline).toBe('Worth investigating')
    expect(t.detail).toContain('2 of 5 columns fully clean')
    expect(t.detail).toContain('2 of 3 assessed dimensions flagged')
  })

  it('is not assessed, never green, when no dimension was assessed', () => {
    const t = trustOf(output({ a: dim(null) }))
    expect(t.level).toBeNull()
    expect(t.headline).toBe('Not assessed')
  })
})

describe('dimensionSummary and issueChips', () => {
  it('says how many columns are affected, or that the findings are about the dataset', () => {
    const f = [ref('1', 'warn', 'x', 't')]
    expect(
      dimensionSummary(dim('amber', { findings: f, affected_count: 1, pct_affected: 0.2 }), 5),
    ).toBe('1 column affected (20% of 5).')
    expect(dimensionSummary(dim('blue', { findings: f }), 5)).toBe('1 dataset-level finding.')
    expect(dimensionSummary(dim('green'), 5)).toBe('No findings.')
  })

  it('counts the finding kinds, most common first', () => {
    const f = [ref('1', 'warn', 'x', 't'), ref('2', 'warn', 'y', 't')]
    f[1]!.code = 'z'
    f.push({ ...f[1]!, id: '3' })
    expect(issueChips(f)).toEqual([
      { code: 'z', count: 2 },
      { code: 'k', count: 1 },
    ])
  })
})

describe('sortFindings', () => {
  const f = [
    ref('1', 'notice', 'b', 'Beta'),
    ref('2', 'error', 'c', 'Alpha'),
    ref('3', 'warn', '', 'Gamma'),
  ]
  it('puts the worst first by severity and sorts text ascending', () => {
    expect(sortFindings(f, 'severity', true).map((x) => x.id)).toEqual(['2', '3', '1'])
    expect(sortFindings(f, 'column', false).map((x) => x.id)).toEqual(['3', '1', '2'])
    expect(sortFindings(f, 'finding', false).map((x) => x.id)).toEqual(['2', '1', '3'])
  })
})

describe('effectBar', () => {
  it('shows a sign only for a rank-biserial effect', () => {
    expect(effectBar({ effect: -0.4, effect_name: 'rank_biserial' })).toEqual({
      width: 0.4,
      text: '−0.40',
    })
    expect(effectBar({ effect: 0.89, effect_name: 'cramers_v' })).toEqual({
      width: 0.89,
      text: '0.89',
    })
  })
})
