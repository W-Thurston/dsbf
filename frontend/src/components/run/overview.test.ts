// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import FindingCounts from './FindingCounts.vue'
import PolicyVerdict from './PolicyVerdict.vue'
import SamplingNotice from './SamplingNotice.vue'

describe('PolicyVerdict', () => {
  const rule = { name: 'no-broken-checks', when: {}, then: [{ kind: 'fail', target: null }] }

  it('shows a failed verdict and the rule that matched', () => {
    const policy = { failed: true, rules: [{ rule, finding_ids: [], check_ids: ['a_err'] }] }
    const panel = mount(PolicyVerdict, { props: { policy: policy as never } })

    expect(panel.find('.badge').text()).toBe('failed')
    expect(panel.text()).toContain('no-broken-checks')
    expect(panel.text()).toContain('1 check')
  })

  it('shows a passed verdict with no rules matched', () => {
    const panel = mount(PolicyVerdict, { props: { policy: { failed: false, rules: [] } } })

    expect(panel.find('.badge').text()).toBe('passed')
    expect(panel.text()).toContain('0 rules matched')
  })
})

describe('SamplingNotice', () => {
  it('says the numbers describe a sample', () => {
    const sampling = {
      strategy: 'random' as const,
      original_rows: 100_000,
      sampled_rows: 10_000,
      seed: 7,
      stratify_by: ['region'],
    }
    const text = mount(SamplingNotice, { props: { sampling } }).text()

    expect(text).toContain('10,000 of 100,000 rows')
    expect(text).toContain('seed 7')
    expect(text).toContain('stratified by region')
    expect(text).toContain('describe the sample')
  })
})

describe('FindingCounts', () => {
  it('sums a dimension by severity and blanks the empty cells', () => {
    const counts = [
      { dimension: 'completeness', severity: 'warn', count: 3 },
      { dimension: 'completeness', severity: 'info', count: 2 },
      { dimension: 'leakage', severity: 'error', count: 1 },
    ] as const
    const panel = mount(FindingCounts, { props: { counts: [...counts] } })

    const rows = panel.findAll('tbody tr').map((row) => row.findAll('th, td').map((c) => c.text()))
    expect(panel.findAll('thead th').map((c) => c.text())).toEqual([
      'Dimension',
      'error',
      'warn',
      'notice',
      'info',
    ])
    expect(rows).toEqual([
      ['completeness', '–', '3', '–', '2'],
      ['leakage', '1', '–', '–', '–'],
    ])
  })

  it('says when there are no findings', () => {
    expect(mount(FindingCounts, { props: { counts: [] } }).text()).toContain('No findings.')
  })
})
