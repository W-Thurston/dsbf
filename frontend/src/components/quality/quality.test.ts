import { describe, expect, it } from 'vitest'
import {
  actionWords,
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
