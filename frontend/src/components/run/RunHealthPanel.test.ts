// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import type { components } from '../../generated/api'
import RunHealthPanel from './RunHealthPanel.vue'

type Health = components['schemas']['RunHealth']

const COUNTS = { ok: 3, not_applicable: 1, unavailable: 1, skipped: 1, error: 1, timeout: 0 }

function health(issues: Health['issues'], counts = COUNTS): Health {
  return { counts, issues, findings_at_warn_or_above: 0 }
}

const ISSUES: Health['issues'] = [
  {
    check_id: 'a_na',
    status: 'not_applicable',
    message: 'No numeric columns.',
    hint: null,
    blocked_by: [],
  },
  { check_id: 'a_err', status: 'error', message: 'It raised.', hint: 'Report it.', blocked_by: [] },
  { check_id: 'a_skip', status: 'skipped', message: 'Skipped.', hint: null, blocked_by: ['a_err'] },
  {
    check_id: 'a_gone',
    status: 'unavailable',
    message: 'Needs scipy.',
    hint: 'pip install dsbf[stats]',
    blocked_by: [],
  },
]

describe('RunHealthPanel', () => {
  it('shows every status other than ok with its reason, failures first', () => {
    const panel = mount(RunHealthPanel, { props: { health: health(ISSUES), notes: [] } })

    const groups = panel.findAll('ul.issues').map((list) => list.attributes('data-status'))
    expect(groups).toEqual(['error', 'skipped', 'unavailable', 'not_applicable'])
    const text = panel.text()
    for (const reason of ['It raised.', 'Skipped.', 'Needs scipy.', 'No numeric columns.']) {
      expect(text).toContain(reason)
    }
    expect(text).toContain('Hint: Report it.')
    expect(text).toContain('Blocked by a_err')
  })

  it('lists the run notes', () => {
    const notes: components['schemas']['Note'][] = [
      {
        code: 'plugin_load_failed',
        message: 'A plugin did not import.',
        hint: 'Fix the file.',
        severity: 'warn',
        blocked_by: [],
        count: null,
        detail: null,
        exception_type: null,
      },
    ]
    const panel = mount(RunHealthPanel, { props: { health: health([]), notes } })

    expect(panel.find('ul[data-status="notes"]').text()).toContain('A plugin did not import.')
    expect(panel.text()).not.toContain('Every check ran to completion.')
  })

  it('says so when nothing needs a reason', () => {
    const counts = { ...COUNTS, not_applicable: 0, unavailable: 0, skipped: 0, error: 0 }
    const panel = mount(RunHealthPanel, { props: { health: health([], counts), notes: [] } })

    expect(panel.text()).toContain('Every check ran to completion.')
  })
})
