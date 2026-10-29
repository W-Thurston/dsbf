// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import LineageTable from '../dataset/LineageTable.vue'
import DiffSections from './DiffSections.vue'

const finding = (id: string, severity: 'low' | 'high') => ({
  id,
  check_id: 'missingness',
  code: 'high_missing',
  title: `Finding ${id}`,
  severity,
})

function diff(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: '1',
    base_run_id: 'a',
    head_run_id: 'b',
    same_dataset: true,
    tolerance: 0.05,
    new: [finding('n1', 'high')],
    resolved: [],
    changed: [
      {
        id: 'c1',
        check_id: 'missingness',
        code: 'high_missing',
        title: 'Changed one',
        base_severity: 'low',
        head_severity: 'high',
        metrics: [{ name: 'fraction', base: 0.1, head: 0.33333 }],
      },
    ],
    status_changes: [{ check_id: 'zeros', base: 'ok', head: null }],
    ...overrides,
  }
}

describe('DiffSections', () => {
  it('lists new, resolved, changed and status changes', () => {
    const text = mount(DiffSections, { props: { diff: diff() as never } }).text()

    expect(text).toContain('1 new, 0 resolved, 1 changed findings; 1 check status change')
    expect(text).toContain('Finding n1')
    expect(text).toContain('fraction 0.1 → 0.3333')
    expect(text).toContain('not run')
  })

  it('says when the runs are of different datasets', () => {
    const wrapper = mount(DiffSections, { props: { diff: diff({ same_dataset: false }) as never } })
    expect(wrapper.find('[data-role="different-datasets"]').exists()).toBe(true)

    const same = mount(DiffSections, { props: { diff: diff() as never } })
    expect(same.find('[data-role="different-datasets"]').exists()).toBe(false)
  })
})

describe('LineageTable', () => {
  const entry = (run_id: string, location_changed: boolean, fingerprint_changed: boolean) => ({
    run_id,
    started_at: '2026-10-29T10:00:00Z',
    location: '/data/titanic.csv',
    fingerprint: 'f',
    n_rows: 891,
    location_changed,
    fingerprint_changed,
  })

  it('flags a changed location and changed content', () => {
    const wrapper = mount(LineageTable, {
      props: { entries: [entry('r2', true, false), entry('r1', false, false)] },
      global: { stubs: { 'router-link': true } },
    })
    const rows = wrapper.findAll('tbody tr')

    expect(rows[0]!.find('[data-flag="location"]').exists()).toBe(true)
    expect(rows[0]!.find('[data-flag="fingerprint"]').exists()).toBe(false)
    expect(rows[1]!.find('[data-flag]').exists()).toBe(false)
  })
})
